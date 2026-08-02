import numpy as np

from dsctm.eval.metrics import classification_metrics


def test_binary_metrics_include_pr_auc_and_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
    m = classification_metrics(y, p.argmax(1), p)
    assert m["pr_auc"] == 1.0
    assert m["auc_roc"] == 1.0
    assert "brier" in m and "ece" in m
