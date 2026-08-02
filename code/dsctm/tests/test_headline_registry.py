from types import SimpleNamespace

import numpy as np

from dsctm.experiments import headline


def test_studentlife_headline_registers_every_completed_fold(monkeypatch, tmp_path):
    ds = SimpleNamespace(
        F=8, T=60, n_classes=3, N=4,
        subject_id=np.array([0, 1, 2, 3]), y=np.array([0, 1, 2, 0]),
        version="synthetic-v1", data_version_hash=lambda: "data-hash",
    )
    folds = [(np.array([1, 2, 3]), np.array([0]))] * 5
    monkeypatch.setattr(
        "dsctm.data.splits.subject_grouped_kfold",
        lambda *args, **kwargs: (folds, {"split_hash": "fixed-split"}),
    )
    monkeypatch.setattr(headline, "_builder", lambda *args: (lambda n: None, False))
    registered = []
    monkeypatch.setattr(headline, "write_completed_fit",
                        lambda **kwargs: registered.append(kwargs))

    def fake_cv(*args, on_fold_complete=None, **kwargs):
        for fold in range(5):
            on_fold_complete(fold, {"val_metrics": {"macro_f1": 0.5}})
        return {"pooled": {"macro_f1": 0.5},
                "per_fold_macro_f1": [0.5] * 5}

    monkeypatch.setattr(headline, "headline_cv", fake_cv)
    headline.run_studentlife_headline(
        ds, seeds=(0,), cfg={"batch_size": 2}, out_root=tmp_path, log=lambda *_: None,
    )
    assert len(registered) == len(headline.CORE_MODELS) * 5
    assert {r["condition"] for r in registered} == set(headline.CORE_MODELS)
    assert all(r["split_hash"] == "fixed-split" for r in registered)
