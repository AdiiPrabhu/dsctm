"""Immutable run registry (master-prompt §5).

Every run gets its own directory under artifacts/resubmission/runs/<run_id>/ and a
row appended to runs.csv carrying the full run-identity schema. Failures are
preserved, never overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

RUN_STATUSES = {
    "completed",
    "infrastructure_failed",
    "model_failed",
    "excluded_by_prespecified_rule",
    "cancelled",
}

RUN_IDENTITY_FIELDS = [
    "campaign_id", "phase", "experiment_id", "condition", "dataset", "protocol",
    "fold", "repeat", "seed", "config_hash", "split_hash", "data_version",
    "git_commit", "environment_hash", "host", "gpu_model", "gpu_count",
    "interconnect", "precision", "batch_definition", "start_time", "end_time",
    "status", "failure_class", "artifact_paths",
]

ARTIFACT_ROOT = Path("artifacts/resubmission")


@dataclass
class RunIdentity:
    campaign_id: str = "dmstcn-ieee-access-resub-2026"
    phase: str = ""
    experiment_id: str = ""
    condition: str = ""
    dataset: str = ""
    protocol: str = ""
    fold: Optional[int] = None
    repeat: Optional[int] = None
    seed: Optional[int] = None
    config_hash: str = ""
    split_hash: str = ""
    data_version: str = ""
    git_commit: str = ""
    environment_hash: str = ""
    host: str = ""
    gpu_model: str = ""
    gpu_count: int = 0
    interconnect: str = ""
    precision: str = ""
    batch_definition: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = ""
    failure_class: str = ""
    artifact_paths: str = ""

    def run_id(self) -> str:
        key = (
            f"{self.experiment_id}|{self.condition}|{self.dataset}|"
            f"{self.fold}|{self.repeat}|{self.seed}|{self.config_hash}"
        )
        h = hashlib.sha256(key.encode()).hexdigest()[:10]
        parts = [
            p
            for p in [
                self.experiment_id,
                self.condition,
                self.dataset,
                None if self.fold is None else f"f{self.fold}",
                None if self.seed is None else f"s{self.seed}",
            ]
            if p
        ]
        return "__".join(parts + [h])


def run_dir(run_id: str, root: Path = ARTIFACT_ROOT) -> Path:
    d = root / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_run(identity: RunIdentity, root: Path = ARTIFACT_ROOT) -> Path:
    if identity.status and identity.status not in RUN_STATUSES:
        raise ValueError(f"invalid run status {identity.status!r}")
    d = run_dir(identity.run_id(), root)
    (d / "run.json").write_text(json.dumps(asdict(identity), indent=2, default=str))
    _append_runs_csv(identity, root)
    return d


def _append_runs_csv(identity: RunIdentity, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "runs.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_IDENTITY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: getattr(identity, k) for k in RUN_IDENTITY_FIELDS})


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def write_completed_fit(*, experiment_id: str, condition: str, dataset, protocol: str,
                        fold: int | None, seed: int, split_hash: str, config: dict,
                        result: dict, precision: str = "float32",
                        root: Path = ARTIFACT_ROOT) -> Path:
    """Persist one fold/seed fit in the campaign's immutable artifact layout.

    Subject identifiers are deliberately excluded. Prediction rows retain only their
    within-evaluation-set order, labels, and probabilities.
    """
    from .repro import capture_environment

    now = datetime.now(timezone.utc).isoformat()
    env = capture_environment()
    config_hash = _stable_hash(config)
    identity = RunIdentity(
        phase=experiment_id.split(".")[0].replace("EXP-", "phase"),
        experiment_id=experiment_id, condition=condition, dataset=dataset.dataset,
        protocol=protocol, fold=fold, seed=seed, config_hash=config_hash,
        split_hash=split_hash, data_version=dataset.data_version_hash(),
        git_commit=_git_commit(), environment_hash=_stable_hash(env), host=platform.node(),
        gpu_model=str(env.get("gpu_model", "")),
        gpu_count=int(env.get("gpu_count", 0)),
        interconnect="single_device_pcie", precision=precision,
        batch_definition=f"samples_per_step={config.get('batch_size')}",
        start_time=result.get("start_time", ""), end_time=now, status="completed",
    )
    rid = identity.run_id()
    target = root / "runs" / rid
    if target.exists():
        # Resume-safe and immutable: accept an already-completed identical identity,
        # but never overwrite or reinterpret it.
        prior = json.loads((target / "run.json").read_text())
        if prior.get("status") == "completed":
            return target
        raise FileExistsError(f"immutable non-completed run directory already exists: {target}")
    target.mkdir(parents=True)

    (target / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (target / "environment.txt").write_text(json.dumps(env, indent=2, default=str))
    (target / "stdout.log").write_text(result.get("stdout", ""))
    (target / "stderr.log").write_text(result.get("stderr", ""))

    metrics = result.get("val_metrics") or result.get("test_metrics") or {}
    with (target / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, json.dumps(value) if isinstance(value, (list, dict)) else value])
    curve = result.get("curve", [])
    with (target / "curve.csv").open("w", newline="") as f:
        fields = sorted({k for row in curve for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(curve)
    probs = result.get("val_probs", result.get("test_probs"))
    truth = result.get("val_true", result.get("test_true"))
    if probs is not None and truth is not None:
        np.savez_compressed(target / "predictions.npz", probabilities=np.asarray(probs),
                            labels=np.asarray(truth))
    (target / "checkpoint_reference.txt").write_text(
        "No model checkpoint retained; best-development predictions and metrics are preserved.\n"
    )
    identity.artifact_paths = str(target)
    (target / "run.json").write_text(json.dumps(asdict(identity), indent=2, default=str))
    _append_runs_csv(identity, root)
    return target


def write_failed_fit(*, experiment_id: str, condition: str, dataset, protocol: str,
                     fold: int | None, seed: int, split_hash: str, config: dict,
                     error: BaseException, root: Path = ARTIFACT_ROOT) -> Path:
    """Persist a failed fit without deleting or silently retrying it."""
    from .repro import capture_environment

    env = capture_environment()
    identity = RunIdentity(
        phase=experiment_id.split(".")[0].replace("EXP-", "phase"),
        experiment_id=experiment_id, condition=condition, dataset=dataset.dataset,
        protocol=protocol, fold=fold, seed=seed, config_hash=_stable_hash(config),
        split_hash=split_hash, data_version=dataset.data_version_hash(),
        git_commit=_git_commit(), environment_hash=_stable_hash(env), host=platform.node(),
        gpu_model=str(env.get("gpu_model", "")), gpu_count=int(env.get("gpu_count", 0)),
        interconnect="single_device_pcie", precision="float32",
        batch_definition=f"samples_per_step={config.get('batch_size')}",
        end_time=datetime.now(timezone.utc).isoformat(), status="model_failed",
        failure_class=type(error).__name__,
    )
    target = root / "runs" / identity.run_id()
    if target.exists():
        return target
    target.mkdir(parents=True)
    (target / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (target / "environment.txt").write_text(json.dumps(env, indent=2, default=str))
    (target / "stdout.log").write_text("")
    (target / "stderr.log").write_text(f"{type(error).__name__}: {error}\n")
    identity.artifact_paths = str(target)
    (target / "run.json").write_text(json.dumps(asdict(identity), indent=2, default=str))
    _append_runs_csv(identity, root)
    return target
