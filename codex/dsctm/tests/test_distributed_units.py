"""Gate 2 — single-process unit tests for the distributed layer.

These run without spawning processes. The multi-process behaviour is covered by
tests/test_distributed_gloo.py.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from dsctm.distributed import (
    EarlyStopCoordinator,
    PredictionRecord,
    PreflightFailure,
    UnpaddedDistributedSampler,
    assert_exact_coverage,
    audit_run_directory,
    audit_sampler_partition,
    autocast_dtype,
    build_records,
    has_lazy_parameters,
    loader_kwargs_for_param,
    materialize_lazy_parameters,
    records_to_arrays,
    resolve_batch_semantics,
    write_json_atomic,
)
from dsctm.distributed.errors import EvaluationCoverageError
from dsctm.models.baselines import ITransformerBaseline


class _Len:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


# --------------------------------------------------------------------------- #
# Unpadded evaluation sampler — the DistributedSampler duplication bug
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,ws", [(47, 2), (47, 4), (56, 3), (275, 2), (2160, 4),
                                  (1, 2), (3, 4), (100, 1)])
def test_eval_partition_covers_each_sample_exactly_once(n, ws):
    shards = [list(UnpaddedDistributedSampler(_Len(n), ws, r)) for r in range(ws)]
    flat = [i for s in shards for i in s]
    assert len(flat) == n, f"emitted {len(flat)} indices for a {n}-sample split"
    assert len(set(flat)) == n, "duplicate indices across ranks"
    assert set(flat) == set(range(n))
    for a in range(ws):
        for b in range(a + 1, ws):
            assert not (set(shards[a]) & set(shards[b])), f"ranks {a},{b} overlap"


@pytest.mark.parametrize("n,ws", [(47, 2), (47, 4), (56, 3), (275, 4)])
def test_eval_shard_sizes_differ_by_at_most_one(n, ws):
    sizes = [len(UnpaddedDistributedSampler(_Len(n), ws, r)) for r in range(ws)]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced shards {sizes}"


def test_daicwoz_test_split_would_be_duplicated_by_the_padded_sampler():
    """The concrete DAIC-WOZ hazard, pinned with numbers."""
    audit = audit_sampler_partition(dataset_len=47, world_size=2)
    assert audit["covers_exactly_once"] is True
    assert audit["total_emitted"] == 47 and audit["unique_emitted"] == 47
    # Stock DistributedSampler(drop_last=False) would pad 47 -> 48 and score one twice.
    assert audit["padded_sampler_would_emit"] == 48
    assert audit["duplicates_avoided"] == 1


def test_eval_partition_is_deterministic():
    a = list(UnpaddedDistributedSampler(_Len(275), 4, 2))
    b = list(UnpaddedDistributedSampler(_Len(275), 4, 2))
    assert a == b


def test_eval_sampler_rejects_out_of_range_rank():
    with pytest.raises(ValueError):
        UnpaddedDistributedSampler(_Len(10), num_replicas=2, rank=2)


def test_eval_sampler_set_epoch_is_a_noop():
    s = UnpaddedDistributedSampler(_Len(20), 2, 0)
    before = list(s)
    s.set_epoch(7)
    assert list(s) == before


# --------------------------------------------------------------------------- #
# Global-batch semantics
# --------------------------------------------------------------------------- #
def test_global_batch_is_preserved_not_multiplied():
    s = resolve_batch_semantics(scientific_global_batch=64, world_size=2)
    assert s.per_rank_batch_size == 32
    assert s.effective_global_batch == 64
    assert s.matches_scientific_intent


@pytest.mark.parametrize("gb,ws,accum,per_rank", [(64, 2, 1, 32), (64, 4, 1, 16),
                                                  (64, 2, 2, 16), (8, 2, 1, 4),
                                                  (64, 1, 1, 64)])
def test_batch_split_arithmetic(gb, ws, accum, per_rank):
    s = resolve_batch_semantics(gb, ws, accum)
    assert s.per_rank_batch_size == per_rank
    assert s.per_rank_batch_size * ws * accum == gb


def test_indivisible_global_batch_is_refused_by_default():
    with pytest.raises(PreflightFailure, match="not divisible"):
        resolve_batch_semantics(scientific_global_batch=10, world_size=4)


def test_indivisible_global_batch_allowed_only_explicitly_and_is_recorded():
    s = resolve_batch_semantics(10, 4, allow_uneven=True)
    assert s.effective_global_batch == 8
    assert not s.matches_scientific_intent
    assert "NOT divisible" in s.note


def test_world_size_larger_than_batch_is_refused():
    with pytest.raises(PreflightFailure, match="cannot be split"):
        resolve_batch_semantics(scientific_global_batch=2, world_size=8)


# --------------------------------------------------------------------------- #
# Precision policy — V100 is sm_70: fp16 yes, bf16 no
# --------------------------------------------------------------------------- #
def test_fp32_disables_autocast():
    assert autocast_dtype("fp32", torch.device("cpu")) is None


def test_fp16_resolves_to_float16():
    assert autocast_dtype("fp16", torch.device("cpu")) is torch.float16


def test_unknown_precision_is_refused():
    with pytest.raises(PreflightFailure, match="unknown precision"):
        autocast_dtype("int4", torch.device("cpu"))


def test_bf16_is_refused_on_sm70(monkeypatch):
    """V100 has no bf16 tensor cores; silent emulation would be unexplainable numbers."""
    import dsctm.distributed.runtime as rt
    monkeypatch.setattr(rt, "compute_capability", lambda device=None: (7, 0))
    with pytest.raises(PreflightFailure, match="sm_70"):
        rt.autocast_dtype("bf16", torch.device("cuda", 0))


def test_bf16_allowed_on_ampere(monkeypatch):
    import dsctm.distributed.runtime as rt
    monkeypatch.setattr(rt, "compute_capability", lambda device=None: (8, 0))
    assert rt.autocast_dtype("bf16", torch.device("cuda", 0)) is torch.bfloat16


# --------------------------------------------------------------------------- #
# Prediction records and coverage validation
# --------------------------------------------------------------------------- #
def _records(ids, rank=0, n_classes=2):
    return [PredictionRecord(sample_id=int(i), subject_id=f"p{i}", label=int(i) % n_classes,
                             logits=[0.1] * n_classes, probabilities=[1 / n_classes] * n_classes,
                             rank=rank) for i in ids]


def test_build_records_shapes_and_softmax():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    recs = build_records([5, 6], ["a", "b"], [0, 1], logits, rank=1)
    assert [r.sample_id for r in recs] == [5, 6]
    assert [r.rank for r in recs] == [1, 1]
    for r in recs:
        assert pytest.approx(sum(r.probabilities), abs=1e-6) == 1.0


def test_build_records_rejects_ragged_batch():
    with pytest.raises(ValueError, match="ragged batch"):
        build_records([1, 2], ["a"], [0, 1], torch.zeros(2, 2), rank=0)


def test_coverage_accepts_an_exact_partition():
    audit = assert_exact_coverage(_records(range(47)), expected_n=47)
    assert audit["covers_exactly_once"] and audit["duplicates"] == 0
    assert audit["unique_n"] == 47


def test_coverage_rejects_duplicates_with_an_actionable_message():
    recs = _records(range(47)) + _records([0])
    with pytest.raises(EvaluationCoverageError, match="duplicate"):
        assert_exact_coverage(recs, expected_n=47)


def test_coverage_rejects_missing_samples():
    with pytest.raises(EvaluationCoverageError, match="but the split has"):
        assert_exact_coverage(_records(range(46)), expected_n=47)


def test_coverage_rejects_a_wrong_id_set():
    with pytest.raises(EvaluationCoverageError, match="does not match"):
        assert_exact_coverage(_records(range(1, 48)), expected_n=47,
                              expected_ids=list(range(47)))


def test_records_to_arrays_is_sorted_by_sample_id():
    recs = _records([3, 1, 2])
    y_true, y_pred, y_prob, subjects = records_to_arrays(recs)
    assert subjects.tolist() == ["p1", "p2", "p3"]
    assert y_prob.shape == (3, 2)
    assert len(y_true) == len(y_pred) == 3


# --------------------------------------------------------------------------- #
# Lazy parameter materialization (iTransformer / nn.LazyLinear vs DDP)
# --------------------------------------------------------------------------- #
def test_itransformer_starts_with_lazy_parameters():
    model = ITransformerBaseline(8, 3, d_model=16, nhead=2, layers=1, head_hidden=16)
    assert has_lazy_parameters(model), "expected nn.LazyLinear to be uninitialized"


def test_materialize_lazy_parameters_makes_the_model_ddp_wrappable():
    model = ITransformerBaseline(8, 3, d_model=16, nhead=2, layers=1, head_hidden=16)
    did_work = materialize_lazy_parameters(
        model, example_input=(torch.randn(2, 30, 8), None))
    assert did_work is True
    assert not has_lazy_parameters(model)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_materialize_is_a_noop_for_an_eager_model():
    model = nn.Linear(4, 4)
    assert materialize_lazy_parameters(model, example_input=(torch.randn(2, 4),)) is False


def test_materialize_refuses_without_an_input():
    model = ITransformerBaseline(8, 3, d_model=16, nhead=2, layers=1, head_hidden=16)
    with pytest.raises(PreflightFailure, match="lazy parameters"):
        materialize_lazy_parameters(model)


def test_materialize_preserves_training_mode():
    model = ITransformerBaseline(8, 3, d_model=16, nhead=2, layers=1, head_hidden=16)
    model.train()
    materialize_lazy_parameters(model, example_input=(torch.randn(2, 30, 8), None))
    assert model.training is True


# --------------------------------------------------------------------------- #
# Early stopping (single-process path; broadcast path is in the gloo suite)
# --------------------------------------------------------------------------- #
def test_early_stop_tracks_best_and_fires_after_patience():
    stopper = EarlyStopCoordinator(patience=2)
    assert stopper.step(0.50, 0).improved
    assert stopper.step(0.60, 1).improved
    d2 = stopper.step(0.55, 2)
    assert not d2.improved and not d2.should_stop and d2.patience == 1
    d3 = stopper.step(0.54, 3)
    assert d3.should_stop and d3.best_score == pytest.approx(0.60)


def test_early_stop_min_mode():
    stopper = EarlyStopCoordinator(patience=1, mode="min")
    assert stopper.step(1.0, 0).improved
    assert stopper.step(0.5, 1).improved
    assert stopper.step(0.7, 2).should_stop


def test_early_stop_state_roundtrip():
    a = EarlyStopCoordinator(patience=3)
    a.step(0.4, 0); a.step(0.3, 1)
    b = EarlyStopCoordinator(patience=3)
    b.load_state_dict(a.state_dict())
    assert b.best_score == a.best_score and b.counter == a.counter


def test_early_stop_rejects_bad_mode():
    with pytest.raises(ValueError):
        EarlyStopCoordinator(patience=1, mode="sideways")


# --------------------------------------------------------------------------- #
# Run-directory contract and atomic writes
# --------------------------------------------------------------------------- #
def test_run_directory_audit_reports_missing_files(tmp_path):
    audit = audit_run_directory(tmp_path)
    assert not audit["complete"]
    assert "metrics.json" in audit["missing"]


def test_run_directory_audit_passes_when_complete(tmp_path):
    from dsctm.distributed import REQUIRED_RUN_FILES
    for name in REQUIRED_RUN_FILES:
        (tmp_path / name).write_text("x")
    audit = audit_run_directory(tmp_path)
    assert audit["complete"] and not audit["missing"]


def test_atomic_json_write_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "metrics.json"
    write_json_atomic(target, {"macro_f1": 0.5})
    assert json.loads(target.read_text())["macro_f1"] == 0.5
    assert not list(tmp_path.glob("*.tmp"))


def test_loader_kwargs_do_not_oversubscribe_a_param_gpu_node():
    kwargs = loader_kwargs_for_param()
    # 40 cores / 2 ranks per node; 4 workers per rank leaves headroom for NCCL and Lustre.
    assert kwargs["num_workers"] == 4
    assert kwargs["persistent_workers"] is True
    assert kwargs["drop_last"] is False


def test_loader_kwargs_respect_zero_workers():
    kwargs = loader_kwargs_for_param(num_workers=0)
    assert kwargs["num_workers"] == 0
    assert "persistent_workers" not in kwargs
