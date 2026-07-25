"""Mechanical audits for immutable experiment summary JSON files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .headline import CORE_MODELS


def audit_studentlife_headline(path, *, expected_split_hash: str,
                               expected_data_hash: str,
                               expected_seeds=(0, 1, 2)) -> dict:
    """Fail closed unless a final EXP-4.1 summary has its complete expected structure."""
    source = Path(path)
    raw = source.read_bytes()
    result = json.loads(raw)
    errors = []

    def require(ok, message):
        if not ok:
            errors.append(message)

    require(result.get("experiment") == "EXP-4.1", "wrong experiment identifier")
    require(result.get("dataset") == "studentlife", "wrong dataset identifier")
    require(result.get("protocol") == "subject_grouped_5fold", "wrong protocol")
    require(result.get("split_hash") == expected_split_hash, "split hash mismatch")
    require(result.get("seeds") == list(expected_seeds), "seed set/order mismatch")
    # Launch revisions predating the embedded field are accepted only because the caller
    # supplies the independently audited cache hash; a conflicting embedded hash fails.
    embedded_hash = result.get("data_hash")
    require(embedded_hash in (None, expected_data_hash), "data hash mismatch")

    results = result.get("results", {})
    require(set(results) == set(CORE_MODELS), "model family is incomplete or unexpected")
    for name in CORE_MODELS:
        if name not in results:
            continue
        row = results[name]
        folds = np.asarray(row.get("per_fold_macro_f1_avg_over_seeds", []), dtype=float)
        require(folds.shape == (5,), f"{name}: expected five fold values")
        if folds.shape == (5,):
            require(bool(np.isfinite(folds).all()), f"{name}: non-finite fold metric")
            require(bool(((folds >= 0) & (folds <= 1)).all()),
                    f"{name}: fold metric outside [0,1]")
            reported = row.get("fold_macro_f1_mean")
            require(reported is not None and np.isfinite(reported),
                    f"{name}: missing/non-finite fold mean")
            if reported is not None and np.isfinite(reported):
                require(bool(np.isclose(float(reported), folds.mean(), atol=1e-12)),
                        f"{name}: fold mean does not recompute")
        pooled = row.get("pooled_macro_f1_mean")
        require(pooled is not None and np.isfinite(pooled) and 0 <= pooled <= 1,
                f"{name}: invalid pooled macro-F1")
        ci = np.asarray(row.get("fold_ci95", []), dtype=float)
        require(ci.shape == (2,) and np.isfinite(ci).all() and ci[0] <= ci[1],
                f"{name}: invalid fold confidence interval")

    comparisons = result.get("dmstcn_vs_baselines", {})
    require(set(comparisons) == set(CORE_MODELS) - {"dmstcn"},
            "paired-comparison family is incomplete or unexpected")
    audit = {
        "source": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "expected_split_hash": expected_split_hash,
        "expected_data_hash": expected_data_hash,
        "embedded_data_hash": embedded_hash,
        "launch_revision_missing_embedded_data_hash": embedded_hash is None,
        "checks_passed": not errors,
        "errors": errors,
    }
    if errors:
        raise ValueError("EXP-4.1 result audit failed: " + "; ".join(errors))
    return audit
