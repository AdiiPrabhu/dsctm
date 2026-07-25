"""Cross-rank prediction gathering with mandatory coverage validation.

Metrics are computed on rank 0 from the union of every rank's predictions. The union must
cover the split **exactly once**. This module refuses to produce metrics otherwise —
a duplicated or missing sample is a silent, plausible-looking corruption of macro-F1, and
on a 47-session test split a single duplicate moves the third decimal.

Every prediction record carries the fields the Gate 4 run contract requires:
``sample_id``, ``subject_id``, ``label``, ``logits``, ``probabilities``, ``rank``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist

from .errors import EvaluationCoverageError


@dataclass
class PredictionRecord:
    """One scored sample. ``sample_id`` is the split-global index, not the batch index."""

    sample_id: int
    subject_id: str
    label: int
    logits: list[float]
    probabilities: list[float]
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_records(sample_ids: Sequence[int], subject_ids: Sequence[Any],
                  labels: Sequence[int], logits: torch.Tensor, rank: int
                  ) -> list[PredictionRecord]:
    """Convert one batch of model output into records.

    ``logits`` is (B, C) on any device; it is detached, moved to CPU and softmaxed here so
    the caller never has to remember to do it.
    """
    logits = logits.detach().float().cpu()
    probs = torch.softmax(logits, dim=1)
    n = logits.shape[0]
    if not (len(sample_ids) == len(subject_ids) == len(labels) == n):
        raise ValueError(
            f"ragged batch: {len(sample_ids)} ids, {len(subject_ids)} subjects, "
            f"{len(labels)} labels, {n} logit rows"
        )
    return [
        PredictionRecord(
            sample_id=int(sample_ids[i]),
            subject_id=str(subject_ids[i]),
            label=int(labels[i]),
            logits=[float(v) for v in logits[i].tolist()],
            probabilities=[float(v) for v in probs[i].tolist()],
            rank=int(rank),
        )
        for i in range(n)
    ]


def gather_predictions(records: Sequence[PredictionRecord],
                       world_size: int | None = None) -> list[PredictionRecord]:
    """All-gather records from every rank and return them sorted by ``sample_id``.

    Every rank receives the full sorted list, so any rank can assert coverage even though
    only rank 0 writes it out.

    Implementation note: ``all_gather_object`` pickles. Fine at this scale — the largest
    split here is StudentLife at 2,160 windows x a handful of floats. If a future dataset
    makes this a bottleneck, switch logits to a padded tensor ``all_gather`` and keep the
    ids as objects; the coverage contract below does not change.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return sorted(records, key=lambda r: r.sample_id)
    ws = world_size or dist.get_world_size()
    buckets: list[list[PredictionRecord] | None] = [None] * ws
    dist.all_gather_object(buckets, list(records))
    merged: list[PredictionRecord] = []
    for bucket in buckets:
        if bucket:
            merged.extend(bucket)
    return sorted(merged, key=lambda r: r.sample_id)


def assert_exact_coverage(records: Sequence[PredictionRecord], expected_n: int,
                          expected_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Hard gate before any metric is computed.

    Raises ``EvaluationCoverageError`` on duplicates, on a count mismatch, or on an
    unexpected id set. Returns an audit dict for the run record on success.
    """
    ids = [r.sample_id for r in records]
    unique = set(ids)
    duplicates = sorted({i for i in ids if ids.count(i) > 1}) if len(unique) != len(ids) else []

    if duplicates:
        raise EvaluationCoverageError(
            f"{len(ids) - len(unique)} duplicate prediction(s) across ranks; "
            f"offending sample_ids (first 10): {duplicates[:10]}. "
            f"This is the DistributedSampler padding bug — evaluation must use "
            f"UnpaddedDistributedSampler."
        )
    if len(unique) != expected_n:
        missing = (sorted(set(expected_ids) - unique)[:10] if expected_ids is not None
                   else "unknown (no expected_ids supplied)")
        raise EvaluationCoverageError(
            f"gathered {len(unique)} unique predictions but the split has {expected_n}. "
            f"Missing (first 10): {missing}"
        )
    if expected_ids is not None and unique != set(expected_ids):
        raise EvaluationCoverageError(
            f"gathered id set does not match the split id set; "
            f"unexpected={sorted(unique - set(expected_ids))[:10]} "
            f"missing={sorted(set(expected_ids) - unique)[:10]}"
        )

    per_rank: dict[int, int] = {}
    for r in records:
        per_rank[r.rank] = per_rank.get(r.rank, 0) + 1
    return {
        "expected_n": expected_n,
        "gathered_n": len(ids),
        "unique_n": len(unique),
        "duplicates": 0,
        "per_rank_counts": dict(sorted(per_rank.items())),
        "covers_exactly_once": True,
    }


def records_to_arrays(records: Sequence[PredictionRecord]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, y_prob, subject_id) as arrays, in ``sample_id`` order."""
    ordered = sorted(records, key=lambda r: r.sample_id)
    y_true = np.array([r.label for r in ordered], dtype=np.int64)
    y_prob = np.array([r.probabilities for r in ordered], dtype=np.float64)
    subjects = np.array([r.subject_id for r in ordered])
    return y_true, y_prob.argmax(1), y_prob, subjects


def gather_and_validate(records: Sequence[PredictionRecord], expected_n: int,
                        expected_ids: Sequence[int] | None = None
                        ) -> tuple[list[PredictionRecord], dict[str, Any]]:
    """The only sanctioned path from per-rank records to metric inputs."""
    merged = gather_predictions(records)
    audit = assert_exact_coverage(merged, expected_n, expected_ids)
    return merged, audit
