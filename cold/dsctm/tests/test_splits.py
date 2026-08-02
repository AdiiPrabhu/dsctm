"""EXP-1.2 leakage assertions on the subject-grouped splitter."""
import numpy as np
import pytest

from dsctm.data.splits import (
    assert_no_subject_overlap,
    audit_folds,
    subject_grouped_kfold,
)
from dsctm.data.synthetic import make_synthetic


def test_subject_grouped_folds_have_no_leakage():
    ds = make_synthetic(n_subjects=15, windows_per_subject=20, seed=1)
    folds, manifest = subject_grouped_kfold(ds.subject_id, ds.y, n_splits=5, seed=1)
    report = audit_folds(folds, ds.subject_id)  # raises on any leakage
    assert len(report) == 5
    assert manifest["split_hash"]
    # every sample appears in exactly one validation fold
    val_counts = np.zeros(ds.N, int)
    for _, va in folds:
        val_counts[va] += 1
    assert (val_counts == 1).all()


def test_leakage_detector_catches_overlap():
    ds = make_synthetic(n_subjects=6, windows_per_subject=10, seed=2)
    idx = np.arange(ds.N)
    with pytest.raises(AssertionError):
        assert_no_subject_overlap(idx, idx, ds.subject_id)  # identical → overlap
