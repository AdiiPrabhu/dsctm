"""Synthetic multi-scale generator so the ENTIRE pipeline (model, splits, leakage
audit, training, stats) runs end-to-end before real StudentLife/DAIC data arrives.

Signals deliberately carry both fast (short-scale) and slow (long-scale) structure
whose mixture encodes the label, and a per-subject bias — so subject-grouped
evaluation and the multi-branch architecture are both exercised meaningfully.
This is clearly labelled synthetic; no synthetic number is ever a manuscript result.
"""
from __future__ import annotations

import numpy as np

from .contract import WindowedDataset


def make_synthetic(
    n_subjects: int = 12,
    windows_per_subject: int = 40,
    T: int = 60,
    F: int = 8,
    n_classes: int = 3,
    seed: int = 0,
) -> WindowedDataset:
    rng = np.random.default_rng(seed)
    X, y, sid, ts = [], [], [], []
    for s in range(n_subjects):
        subj_bias = rng.normal(0, 1, F)
        subj_mix = rng.normal(1, 0.2, F)
        for w in range(windows_per_subject):
            cls = int(rng.integers(0, n_classes))
            t = np.arange(T)
            fast = np.sin(2 * np.pi * t / 3 + rng.uniform(0, 2 * np.pi))
            slow = np.sin(2 * np.pi * t / 40 + rng.uniform(0, 2 * np.pi))
            amp_fast = 0.5 + 0.7 * (cls == 0)
            amp_slow = 0.5 + 0.7 * (cls == n_classes - 1)
            base = amp_fast * fast + amp_slow * slow
            feats = np.outer(base, subj_mix) + subj_bias + rng.normal(0, 0.3, (T, F))
            X.append(feats.astype(np.float32))
            y.append(cls)
            sid.append(f"S{s:03d}")
            ts.append(w)
    return WindowedDataset(
        X=np.asarray(X, np.float32),
        y=np.asarray(y),
        subject_id=np.asarray(sid),
        timestamp=np.asarray(ts),
        feature_names=[f"f{i}" for i in range(F)],
        label_type="multiclass" if n_classes > 2 else "binary",
        n_classes=n_classes,
        sampling_interval_s=60.0,
        dataset="synthetic",
        version=f"synth-v1-s{seed}",
    )


def make_delay_dependency(delay: int, n_subjects: int = 25, windows_per_subject: int = 24,
                          T: int = 256, seed: int = 0) -> WindowedDataset:
    """Controlled binary XOR task with a known temporal dependency length.

    Channel 0 contains two signed bits at ``T-1-delay`` and ``T-1``; the target is
    one iff their signs differ. Channel 1 marks the query timestep ``T-1``. Channels
    2–3 are nuisance Gaussian processes plus a subject-specific offset. All four bit
    combinations are cycled within every subject, preventing label imbalance and
    subject shortcuts. No target-derived input feature is present.
    """
    if not 1 <= delay < T:
        raise ValueError("delay must satisfy 1 <= delay < T")
    rng = np.random.default_rng(seed)
    X, y, sid, ts = [], [], [], []
    combinations = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    for subject in range(n_subjects):
        nuisance_offset = rng.normal(0.0, 0.15, size=2)
        order = rng.permutation(windows_per_subject)
        for window in range(windows_per_subject):
            a, b = combinations[int(order[window] % 4)]
            sample = rng.normal(0.0, 0.05, size=(T, 4))
            sample[:, 2:] += nuisance_offset
            sample[T - 1 - delay, 0] = 3.0 * a
            sample[T - 1, 0] = 3.0 * b
            sample[T - 1, 1] = 1.0
            X.append(sample.astype(np.float32))
            y.append(int(a != b))
            sid.append(f"S{subject:03d}")
            ts.append(window)
    return WindowedDataset(
        X=np.asarray(X), y=np.asarray(y), subject_id=np.asarray(sid),
        timestamp=np.asarray(ts), feature_names=["signal", "query", "noise1", "noise2"],
        label_type="binary", n_classes=2, sampling_interval_s=1.0,
        dataset="synthetic-delay", version=f"delay-xor-v1-d{delay}-s{seed}",
    )
