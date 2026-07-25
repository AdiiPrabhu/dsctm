"""Phase 1 — data integrity (Gate 1, master-prompt §9): provenance manifests +
leakage audit. Records manuscript-vs-actual discrepancies rather than hiding them.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.splits import audit_folds, subject_grouped_kfold


def provenance_studentlife(ds):
    uniq, counts = np.unique(ds.subject_id, return_counts=True)
    cls, ccounts = np.unique(ds.y, return_counts=True)
    return {
        "dataset": ds.dataset,
        "N": int(ds.N), "T": int(ds.T), "F": int(ds.F),
        "n_subjects": int(len(uniq)),
        "class_counts": {int(k): int(v) for k, v in zip(cls, ccounts)},
        "windows_per_subject": {
            "min": int(counts.min()), "max": int(counts.max()), "median": int(np.median(counts))
        },
        "sensor_missingness_mean": round(float(np.mean(getattr(ds, "_missingness", np.zeros(1)))), 3),
        "data_version_hash": ds.data_version_hash(),
        "manuscript_claim": "48 subjects, F=8, T=60, 3-class stress",
        "actual": f"{len(uniq)} subjects (≥ min stress-EMA), F={ds.F}, T={ds.T}, 3-class",
    }


def leakage_studentlife(ds, n_splits=5, seed=0):
    folds, manifest = subject_grouped_kfold(ds.subject_id, ds.y, n_splits=n_splits, seed=seed)
    per_fold = audit_folds(folds, ds.subject_id)  # raises on any leakage
    val_counts = np.zeros(ds.N, int)
    for _, va in folds:
        val_counts[va] += 1
    return {
        "scheme": "subject_grouped_kfold",
        "n_splits": n_splits,
        "split_hash": manifest["split_hash"],
        "subject_overlap_across_folds": 0,
        "each_sample_in_exactly_one_val_fold": bool((val_counts == 1).all()),
        "leakage_free": True,
        "per_fold": per_fold,
    }


def provenance_daic(ds, manifest):
    cls, ccounts = np.unique(ds.y, return_counts=True)
    lengths = getattr(ds, "_lengths", np.zeros(0))
    return {
        "dataset": ds.dataset,
        "N": int(ds.N), "T": int(ds.T), "F": int(ds.F),
        "n_subjects": int(len(np.unique(ds.subject_id))),
        "class_counts": {int(k): int(v) for k, v in zip(cls, ccounts)},
        "official_split_counts": manifest["counts"],
        "true_length_frames": (
            {"min": int(lengths.min()), "max": int(lengths.max()), "median": int(np.median(lengths))}
            if len(lengths) else {}
        ),
        "data_version_hash": ds.data_version_hash(),
        "manuscript_claim": "189 sessions, 107/82 split, eGeMAPS 88-dim @0.5s openSMILE 3.0",
        "actual": (
            f"{ds.dataset} {len(np.unique(ds.subject_id))} sessions, official "
            f"{manifest['counts']}, F={ds.F}, version={ds.version}"
        ),
        "feature_provenance": manifest.get("feature_set", "23-dim eGeMAPS LLD, openSMILE 2.3.0"),
    }


def leakage_daic(ds, manifest):
    subs_by_split = {}
    for pid, s in manifest["split_of_subject"].items():
        subs_by_split.setdefault(s, set()).add(pid)
    splits = list(subs_by_split)
    overlaps = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlaps[f"{splits[i]}&{splits[j]}"] = len(subs_by_split[splits[i]] & subs_by_split[splits[j]])
    return {
        "scheme": "official_split",
        "counts": dict(Counter(manifest["split_of_subject"].values())),
        "cross_split_participant_overlap": overlaps,
        "leakage_free": all(v == 0 for v in overlaps.values()),
    }


def _write(res, out_root, name):
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gate1_{name}.json").write_text(json.dumps(res, indent=2, default=str))
    return out / f"gate1_{name}.json"


def run_gate1_studentlife(ds, out_root="artifacts/resubmission/gate1"):
    res = {"gate": "Gate 1 — data integrity", "dataset": "studentlife",
           "provenance": provenance_studentlife(ds), "leakage": leakage_studentlife(ds)}
    _write(res, out_root, "studentlife")
    return res


def run_gate1_daic(ds, manifest, out_root="artifacts/resubmission/gate1"):
    res = {"gate": "Gate 1 — data integrity", "dataset": "e-daic",
           "provenance": provenance_daic(ds, manifest), "leakage": leakage_daic(ds, manifest)}
    _write(res, out_root, "daic")
    return res
