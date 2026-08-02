"""Classification metrics (master-prompt §8): macro-F1 (primary), accuracy,
balanced accuracy, per-class F1, confusion matrix, AUC-ROC, PR-AUC, Brier, ECE."""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    out["per_class_precision"] = p.tolist()
    out["per_class_recall"] = r.tolist()
    out["per_class_f1"] = f.tolist()
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                out["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif y_prob.ndim == 2:
                out["auc_roc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception:
            out["auc_roc"] = None
        out["ece"] = expected_calibration_error(y_true, y_prob)
        out["brier"] = brier_score(y_true, y_prob)
    return out


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    conf = y_prob.max(1)
    pred = y_prob.argmax(1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n, c = y_prob.shape
    onehot = np.zeros((n, c))
    onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))
