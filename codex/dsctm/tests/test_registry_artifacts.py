from types import SimpleNamespace

import numpy as np

from dsctm.registry import write_completed_fit, write_failed_fit


def test_completed_fit_writes_required_artifacts_without_subject_ids(tmp_path):
    ds = SimpleNamespace(dataset="synthetic", data_version_hash=lambda: "data123")
    result = {"val_metrics": {"macro_f1": 0.5}, "curve": [{"epoch": 0, "loss": 1.0}],
              "val_probs": np.array([[0.4, 0.6]]), "val_true": np.array([1])}
    path = write_completed_fit(experiment_id="EXP-9.9", condition="smoke", dataset=ds,
                               protocol="test", fold=0, seed=0, split_hash="split123",
                               config={"batch_size": 1}, result=result, root=tmp_path)
    required = {"run.json", "config_resolved.yaml", "environment.txt", "stdout.log",
                "stderr.log", "metrics.csv", "curve.csv", "predictions.npz",
                "checkpoint_reference.txt"}
    assert required <= {p.name for p in path.iterdir()}
    assert "subject" not in (path / "run.json").read_text().lower()


def test_completed_fit_is_resume_safe_and_failure_is_preserved(tmp_path):
    ds = SimpleNamespace(dataset="synthetic", data_version_hash=lambda: "data123")
    args = dict(experiment_id="EXP-9.8", condition="resume", dataset=ds, protocol="test",
                fold=0, seed=0, split_hash="split123", config={"batch_size": 1})
    result = {"val_metrics": {"macro_f1": 0.5}}
    first = write_completed_fit(**args, result=result, root=tmp_path)
    assert write_completed_fit(**args, result=result, root=tmp_path) == first
    failed = write_failed_fit(**{**args, "condition": "failure"},
                              error=RuntimeError("expected"), root=tmp_path)
    assert '"status": "model_failed"' in (failed / "run.json").read_text()
    assert "RuntimeError: expected" in (failed / "stderr.log").read_text()
