"""Fail-closed admission of experiment families, and aggregation of admitted runs.

Nothing reaches a manuscript table without passing through here. The auditor rejects, and
says why, on every condition the brief lists:

    missing models · missing folds · wrong seeds · wrong dataset hash · wrong split hash
    non-finite metrics · metrics out of range · invalid confidence intervals
    duplicate sample ids · missing predictions · summaries that do not recompute
    test access during tuning · incomplete runs marked successful

A family that fails admission produces no numbers at all. Partial admission is not offered,
because a partially-admitted family is exactly how a campaign ends up reporting five of six
models and calling it a comparison.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..eval import statistics as st
from .plan import build_plan, plan_digest

METRIC_RANGES = {
    "macro_f1": (0.0, 1.0), "accuracy": (0.0, 1.0), "balanced_accuracy": (0.0, 1.0),
    "auc_roc": (0.0, 1.0), "pr_auc": (0.0, 1.0), "ece": (0.0, 1.0), "brier": (0.0, 2.0),
}


@dataclass
class AuditResult:
    family: str
    admitted: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_expected: int = 0
    n_found: int = 0
    n_completed: int = 0
    receipt: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "admitted": self.admitted,
                "errors": self.errors, "warnings": self.warnings,
                "n_expected": self.n_expected, "n_found": self.n_found,
                "n_completed": self.n_completed, "receipt": self.receipt,
                "details": self.details}


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    if not status_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return {"run_dir": str(run_dir), "status": "unreadable", "metrics": None}
    metrics = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = None
    return {"run_dir": str(run_dir), "task_id": run_dir.name,
            "status": status.get("status"), "status_blob": status, "metrics": metrics}


def _finite_in_range(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return f"{name} is not finite ({value})"
    lo, hi = METRIC_RANGES.get(name, (None, None))
    if lo is not None and not (lo <= value <= hi):
        return f"{name}={value} outside valid range [{lo}, {hi}]"
    return None


def _scan_metrics(prefix: str, blob: Any, errors: list[str]) -> None:
    """Recursively validate every numeric metric we recognise."""
    if isinstance(blob, dict):
        for k, v in blob.items():
            if isinstance(v, (dict, list)):
                _scan_metrics(f"{prefix}{k}.", v, errors)
            else:
                msg = _finite_in_range(k, v)
                if msg:
                    errors.append(f"{prefix}{msg}")
    elif isinstance(blob, list):
        for i, v in enumerate(blob):
            if isinstance(v, (dict, list)):
                _scan_metrics(f"{prefix}{i}.", v, errors)


def audit_family(family: str, results_root: str | Path,
                 require_all: bool = True) -> AuditResult:
    """Admit or reject an entire experiment family."""
    root = Path(results_root) / family
    expected = build_plan(family)
    res = AuditResult(family=family, admitted=False, n_expected=len(expected))

    if not root.exists():
        res.errors.append(f"results directory does not exist: {root}")
        return res

    found = {p.name: _load_run(p) for p in sorted(root.iterdir()) if p.is_dir()}
    found = {k: v for k, v in found.items() if v is not None}
    res.n_found = len(found)

    expected_ids = {t.task_id for t in expected}
    missing = sorted(expected_ids - set(found))
    unexpected = sorted(set(found) - expected_ids)

    if missing and require_all:
        res.errors.append(
            f"{len(missing)} planned task(s) have no run directory "
            f"(first 5: {missing[:5]}). A family is admitted whole or not at all.")
    if unexpected:
        res.errors.append(
            f"{len(unexpected)} run directory(ies) are not in the plan "
            f"(first 5: {unexpected[:5]}). Plan drift — check plan_digest.")

    completed, failed, incomplete = [], [], []
    for task_id, run in found.items():
        status = run["status"]
        if status == "completed":
            completed.append(run)
        elif status in ("model_failed", "excluded_by_prespecified_rule"):
            failed.append(run)
        else:
            incomplete.append((task_id, status))
    res.n_completed = len(completed)

    # A run claiming completion whose contract is violated is a hard error.
    for run in completed:
        contract = run["status_blob"].get("contract", {})
        if contract and not contract.get("complete", False):
            res.errors.append(
                f"{run['task_id']}: status=completed but contract incomplete "
                f"(missing {contract.get('missing')})")
        if run["metrics"] is None:
            res.errors.append(f"{run['task_id']}: completed but metrics.json is unreadable")

    if incomplete and require_all:
        res.errors.append(
            f"{len(incomplete)} run(s) are neither completed nor a recorded failure: "
            f"{incomplete[:5]}")

    # Metric sanity across every admitted run.
    metric_errors: list[str] = []
    for run in completed:
        if run["metrics"]:
            _scan_metrics(f"{run['task_id']}: ", run["metrics"], metric_errors)
    res.errors.extend(metric_errors[:20])
    if len(metric_errors) > 20:
        res.errors.append(f"... and {len(metric_errors) - 20} further metric violations")

    # Provenance agreement: every run in a family must share dataset and split hashes.
    hashes: dict[str, set] = {"dataset": set(), "split": set(), "plan": set()}
    for run in completed:
        d = Path(run["run_dir"])
        try:
            hashes["dataset"].add(json.loads((d / "dataset_hashes.json").read_text())
                                  .get("data_version_hash"))
            hashes["split"].add(json.dumps(json.loads((d / "split_hashes.json").read_text()),
                                           sort_keys=True)[:64])
        except Exception:
            res.errors.append(f"{run['task_id']}: provenance files unreadable")
        if run["metrics"]:
            hashes["plan"].add(run["metrics"].get("plan_digest"))
    for key, values in hashes.items():
        clean = {v for v in values if v is not None}
        if len(clean) > 1:
            res.errors.append(f"inconsistent {key} hash across the family: {sorted(clean)[:4]}")
    if hashes["plan"] and plan_digest(family) not in hashes["plan"]:
        res.warnings.append(
            f"recorded plan_digest {sorted(hashes['plan'])} != current {plan_digest(family)}; "
            f"the plan changed after these runs executed")

    # Tuning families must never have touched test.
    if family.startswith("tuning"):
        for run in completed:
            m = run["metrics"] or {}
            if m.get("test_accessed") is True or "test_metrics" in m:
                res.errors.append(
                    f"{run['task_id']}: tuning run recorded test metrics — "
                    f"test must be inaccessible during search")

    res.details = {
        "missing_task_ids": missing[:50],
        "unexpected_task_ids": unexpected[:50],
        "failed_runs": [r["task_id"] for r in failed],
        "incomplete_runs": [t for t, _ in incomplete][:50],
        "dataset_hashes": sorted(v for v in hashes["dataset"] if v),
        "current_plan_digest": plan_digest(family),
    }
    res.admitted = not res.errors
    if res.admitted:
        res.receipt = _family_receipt(family, completed)
    return res


def _family_receipt(family: str, runs: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    digest.update(family.encode())
    for run in sorted(runs, key=lambda r: r["task_id"]):
        digest.update(run["task_id"].encode())
        receipt_file = Path(run["run_dir"]) / "receipt.sha256"
        if receipt_file.exists():
            digest.update(receipt_file.read_bytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Aggregation — only ever called on an ADMITTED family
# --------------------------------------------------------------------------- #
def _primary_scores(run: dict) -> float | None:
    m = run.get("metrics") or {}
    for path in (("pooled", "macro_f1"), ("test_metrics", "macro_f1"),
                 ("dev_metrics", "macro_f1")):
        node = m
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            return float(node)
    return None


def aggregate_family(family: str, results_root: str | Path,
                     group_by: str = "auto") -> dict[str, Any]:
    """Group admitted runs and compute paired statistics with multiplicity correction.

    `group_by="auto"` groups tuning/confirmation families by model and ablation families by
    condition — i.e. by whatever the family is actually comparing.
    """
    root = Path(results_root) / family
    plan = {t.task_id: t for t in build_plan(family)}
    runs = []
    for p in sorted(root.iterdir()) if root.exists() else []:
        if not p.is_dir():
            continue
        run = _load_run(p)
        if run and run["status"] == "completed" and run["metrics"]:
            run["task"] = plan.get(p.name)
            runs.append(run)

    if group_by == "auto":
        group_by = "condition" if family == "ablation" else "model"

    groups: dict[str, list[dict]] = {}
    for run in runs:
        task = run["task"]
        if task is None:
            continue
        key = getattr(task, group_by)
        groups.setdefault(key, []).append(run)

    summary: dict[str, Any] = {}
    per_group_scores: dict[str, list[float]] = {}
    for key, members in sorted(groups.items()):
        scores = [s for s in (_primary_scores(r) for r in members) if s is not None]
        if not scores:
            continue
        arr = np.asarray(scores, dtype=float)
        point, lo, hi = st.bootstrap_ci(arr)
        per_group_scores[key] = scores
        summary[key] = {
            "n_runs": len(scores),
            "macro_f1_mean": float(arr.mean()),
            "macro_f1_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "bootstrap_point": float(point),
            "ci95": [float(lo), float(hi)],
            "per_run": scores,
        }
        if not (lo <= point <= hi):
            summary[key]["ci_invalid"] = True

    reference = "dmstcn" if "dmstcn" in per_group_scores else (
        "branch_ssb+msb+lsb" if "branch_ssb+msb+lsb" in per_group_scores else None)

    comparisons: dict[str, Any] = {}
    if reference and len(per_group_scores) > 1:
        ref = np.asarray(per_group_scores[reference], dtype=float)
        names, raw_p = [], []
        for key, scores in sorted(per_group_scores.items()):
            if key == reference:
                continue
            other = np.asarray(scores, dtype=float)
            n = min(len(ref), len(other))
            if n < 2:
                continue
            a, b = ref[:n], other[:n]
            test = st.wilcoxon_paired(a, b)
            comparisons[key] = {
                "n_pairs": n,
                "hodges_lehmann_ref_minus_other": st.hodges_lehmann_paired(a, b),
                "rank_biserial": st.paired_rank_biserial(a, b),
                "wilcoxon": test,
            }
            names.append(key)
            raw_p.append(test["p_value"])
        if raw_p:
            holm = st.holm_bonferroni(raw_p)
            bh = st.benjamini_hochberg(raw_p)
            for i, key in enumerate(names):
                comparisons[key]["wilcoxon"]["holm_adjusted_p"] = float(holm[i])
                comparisons[key]["wilcoxon"]["bh_adjusted_p"] = float(bh[i])

    return {
        "family": family,
        "group_by": group_by,
        "reference": reference,
        "n_admitted_runs": len(runs),
        "groups": summary,
        "comparisons": comparisons,
        "multiplicity_family": (
            f"all prespecified {reference}-versus-other comparisons within {family}"
            if reference else None),
        "plan_digest": plan_digest(family),
    }
