"""Immutable run registry (master-prompt §5).

Every run gets its own directory under artifacts/resubmission/runs/<run_id>/ and a
row appended to runs.csv carrying the full run-identity schema. Failures are
preserved, never overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

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
