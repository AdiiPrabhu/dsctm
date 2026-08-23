import json

import pytest

from dsctm.experiments.headline import CORE_MODELS
from dsctm.experiments.result_audit import audit_studentlife_headline


def _valid_result():
    row = {
        "pooled_macro_f1_mean": 0.5, "pooled_macro_f1_std": 0.0,
        "fold_macro_f1_mean": 0.5, "fold_ci95": [0.4, 0.6],
        "per_fold_macro_f1_avg_over_seeds": [0.5] * 5,
    }
    return {
        "experiment": "EXP-4.1", "dataset": "studentlife",
        "protocol": "subject_grouped_5fold", "split_hash": "split",
        "data_hash": "data", "seeds": [0, 1, 2],
        "results": {name: dict(row) for name in CORE_MODELS},
        "dmstcn_vs_baselines": {name: {} for name in CORE_MODELS if name != "dmstcn"},
    }


def test_result_audit_accepts_complete_consistent_summary(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_valid_result()))
    audit = audit_studentlife_headline(
        path, expected_split_hash="split", expected_data_hash="data")
    assert audit["checks_passed"]
    assert len(audit["sha256"]) == 64


def test_result_audit_rejects_incomplete_or_inconsistent_summary(tmp_path):
    result = _valid_result()
    del result["results"]["lstm"]
    result["results"]["dmstcn"]["fold_macro_f1_mean"] = 0.7
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="model family.*dmstcn: fold mean"):
        audit_studentlife_headline(
            path, expected_split_hash="split", expected_data_hash="data")
