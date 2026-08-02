"""Leakage-safe splitting (master-prompt §7.1, EXP-1.2).

Subjects, not windows, are the unit of splitting: all windows of one subject stay
in a single fold, and subjects are split BEFORE any preprocessing/normalization.
Automated leakage assertions are provided for the Gate-1 audit.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _split_hash(subject_folds: dict) -> str:
    return hashlib.sha256(json.dumps(subject_folds, sort_keys=True).encode()).hexdigest()[:16]


def subject_grouped_kfold(subject_id, y, n_splits=5, seed=0, stratify=True):
    """Subject-grouped k-fold CV, stratified by subject-level label prevalence.

    Returns (folds, manifest):
      folds    - list of (train_idx, val_idx) numpy arrays into the sample axis
      manifest - dict with per-fold subject lists and a stable split_hash
    """
    subject_id = np.asarray(subject_id)
    y = np.asarray(y)
    subjects = np.unique(subject_id)
    if stratify:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            # Stratify on all window labels while grouping by participant. Reducing a
            # participant's distribution to round(mean(label)) is invalid for multiclass
            # labels and produced severely unbalanced folds on StudentLife.
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            fold_of_subject = np.empty(len(subjects), int)
            subject_pos = {s: i for i, s in enumerate(subjects)}
            for f, (_, va) in enumerate(splitter.split(np.zeros(len(y)), y, subject_id)):
                for s in np.unique(subject_id[va]):
                    fold_of_subject[subject_pos[s]] = f
        except Exception:
            rng = np.random.default_rng(seed)
            fold_of_subject = rng.permutation(len(subjects)) % n_splits
    else:
        rng = np.random.default_rng(seed)
        fold_of_subject = rng.permutation(len(subjects)) % n_splits

    folds, subject_folds = [], {}
    for f in range(n_splits):
        val_subjects = subjects[fold_of_subject == f]
        val_mask = np.isin(subject_id, val_subjects)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        folds.append((train_idx, val_idx))
        subject_folds[str(f)] = sorted(map(str, val_subjects.tolist()))

    manifest = {
        "scheme": "subject_grouped_kfold",
        "n_splits": n_splits,
        "seed": seed,
        "stratify": stratify,
        "stratification_unit": "window_labels_grouped_by_subject",
        "subject_folds": subject_folds,
        "split_hash": _split_hash(subject_folds),
    }
    return folds, manifest


def stratified_holdout_by_subject(subject_id, y, test_frac=0.2, seed=0):
    """StudentLife-style participant-stratified holdout (the paper's 80/20 protocol),
    kept available so EXP-2.1 can reproduce the ORIGINAL protocol alongside the
    corrected grouped-CV protocol."""
    subject_id = np.asarray(subject_id)
    y = np.asarray(y)
    subjects = np.unique(subject_id)
    subj_label = np.array([int(np.round(np.mean(y[subject_id == s]))) for s in subjects])
    try:
        from sklearn.model_selection import train_test_split

        tr_s, te_s = train_test_split(
            subjects, test_size=test_frac, random_state=seed, stratify=subj_label
        )
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(subjects))
        cut = int(round(len(subjects) * (1 - test_frac)))
        tr_s, te_s = subjects[perm[:cut]], subjects[perm[cut:]]
    te_mask = np.isin(subject_id, te_s)
    manifest = {
        "scheme": "stratified_holdout_by_subject",
        "test_frac": test_frac,
        "seed": seed,
        "test_subjects": sorted(map(str, np.asarray(te_s).tolist())),
        "split_hash": _split_hash({"test": sorted(map(str, np.asarray(te_s).tolist()))}),
    }
    return np.where(~te_mask)[0], np.where(te_mask)[0], manifest


# --------------------------------------------------------------------------- #
# Leakage assertions (EXP-1.2)
# --------------------------------------------------------------------------- #
def assert_no_subject_overlap(train_idx, val_idx, subject_id):
    subject_id = np.asarray(subject_id)
    inter = set(subject_id[train_idx].tolist()) & set(subject_id[val_idx].tolist())
    if inter:
        raise AssertionError(
            f"SUBJECT LEAKAGE: {len(inter)} subject(s) in both train and val: {sorted(inter)[:5]}"
        )
    return True


def assert_disjoint_indices(train_idx, val_idx):
    if set(np.asarray(train_idx).tolist()) & set(np.asarray(val_idx).tolist()):
        raise AssertionError("WINDOW LEAKAGE: overlapping sample indices between splits")
    return True


def audit_folds(folds, subject_id):
    """Run all leakage assertions across every fold; returns a per-fold report."""
    report = []
    for f, (tr, va) in enumerate(folds):
        assert_disjoint_indices(tr, va)
        assert_no_subject_overlap(tr, va, subject_id)
        report.append({"fold": f, "n_train": int(len(tr)), "n_val": int(len(va)),
                       "subject_overlap": 0, "index_overlap": 0})
    return report
