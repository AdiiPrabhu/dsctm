"""The canonical data contract.

Every data source — the synthetic generator now, and the real StudentLife /
DAIC-WOZ loaders once data arrives — produces a `WindowedDataset` of exactly this
shape. Enforcing the contract in one place means the model, training loop, splits,
and leakage audit are written once and never care where the data came from.

This file is also the target the user's data-gathering aims at: to plug real data
in, a loader only needs to emit these arrays.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WindowedDataset:
    X: np.ndarray                       # (N, T, F) float32 — windowed multivariate series
    y: np.ndarray                       # (N,) labels (int for classification, float for regression)
    subject_id: np.ndarray              # (N,) subject/session id (hashed; never raw PII in artifacts)
    timestamp: Optional[np.ndarray] = None  # (N,) window-start index — for chronological support/query
    feature_names: Optional[list] = None
    label_type: str = "binary"          # binary | multiclass | regression
    n_classes: int = 2
    sampling_interval_s: Optional[float] = None
    dataset: str = ""
    version: str = ""

    def __post_init__(self):
        self.X = np.asarray(self.X, np.float32)
        self.y = np.asarray(self.y)
        self.subject_id = np.asarray(self.subject_id)
        if self.X.ndim != 3:
            raise ValueError(f"X must be (N,T,F); got {self.X.shape}")
        if not (len(self.X) == len(self.y) == len(self.subject_id)):
            raise ValueError("X, y, subject_id must have equal length")
        if self.timestamp is not None:
            self.timestamp = np.asarray(self.timestamp)
            if len(self.timestamp) != len(self.X):
                raise ValueError("timestamp length mismatch")

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def T(self) -> int:
        return self.X.shape[1]

    @property
    def F(self) -> int:
        return self.X.shape[2]

    def summary(self) -> dict:
        uniq, counts = np.unique(self.subject_id, return_counts=True)
        cls, ccounts = np.unique(self.y, return_counts=True)
        return {
            "dataset": self.dataset,
            "version": self.version,
            "N": int(self.N),
            "T": int(self.T),
            "F": int(self.F),
            "n_subjects": int(len(uniq)),
            "windows_per_subject_min": int(counts.min()),
            "windows_per_subject_max": int(counts.max()),
            "label_type": self.label_type,
            "class_counts": {str(k): int(v) for k, v in zip(cls, ccounts)},
            "data_version_hash": self.data_version_hash(),
        }

    def data_version_hash(self) -> str:
        """Content hash used as `data_version` in the run registry."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.X).tobytes())
        h.update(np.ascontiguousarray(self.y).tobytes())
        h.update(np.ascontiguousarray(self.subject_id).tobytes())
        return h.hexdigest()[:16]


def to_torch_dataset(ds: WindowedDataset):
    """Wrap a WindowedDataset as a torch Dataset yielding (X[T,F], y, subject_idx)."""
    import torch
    from torch.utils.data import TensorDataset

    subj_to_idx = {s: i for i, s in enumerate(np.unique(ds.subject_id))}
    subj_idx = np.array([subj_to_idx[s] for s in ds.subject_id], dtype=np.int64)
    y_dtype = torch.float32 if ds.label_type == "regression" else torch.long
    return TensorDataset(
        torch.from_numpy(ds.X),
        torch.as_tensor(ds.y, dtype=y_dtype),
        torch.from_numpy(subj_idx),
    ), subj_to_idx
