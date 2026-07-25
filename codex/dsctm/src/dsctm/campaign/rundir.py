"""Run-directory contract writer.

Gate 4 mandates 15 files per run. A run missing any of them is NOT complete, regardless of
exit code — that rule exists because the previous campaign produced numbers whose backing
artifacts no longer exist anywhere, making every one of them unverifiable
(`artifacts/gate0/OLD_RESULT_QUARANTINE.md`).

This module writes them, and `finalize` refuses to mark a run complete if any is missing.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "command.txt", "resolved_config.yaml", "environment.json", "git.json",
    "slurm.json", "hardware.json", "dataset_hashes.json", "split_hashes.json",
    "stdout.log", "stderr.log", "metrics.json", "predictions.parquet",
    "checkpoint.pt", "status.json", "receipt.sha256",
)

#: Files whose absence is tolerated ONLY with a written, recorded reason.
WAIVABLE = {"checkpoint.pt", "predictions.parquet"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_metadata() -> dict[str, Any]:
    return {
        "commit": _sh(["git", "rev-parse", "HEAD"]),
        "branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_sh(["git", "status", "--porcelain"])),
        "describe": _sh(["git", "describe", "--tags", "--always"]),
        "captured_utc": _utc(),
    }


def slurm_metadata() -> dict[str, Any]:
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES",
            "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE", "SLURM_CPUS_PER_TASK",
            "SLURM_GPUS_ON_NODE", "SLURM_SUBMIT_DIR", "SLURM_JOB_ACCOUNT")
    meta = {k: os.environ.get(k) for k in keys}
    meta["inside_allocation"] = bool(os.environ.get("SLURM_JOB_ID"))
    meta["captured_utc"] = _utc()
    return meta


def hardware_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "captured_utc": _utc(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                cap = torch.cuda.get_device_capability(i)
                devices.append({"index": i, "name": p.name,
                                "capability": f"{cap[0]}.{cap[1]}",
                                "total_memory_gib": round(p.total_memory / 2**30, 2),
                                "multi_processor_count": p.multi_processor_count})
            info["devices"] = devices
        info["nvidia_smi_driver"] = _sh(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "numpy", "scipy", "sklearn", "pandas", "yaml",
                 "pyarrow", "thop", "opensmile"):
        try:
            packages[name] = getattr(__import__(name), "__version__", "present")
        except Exception:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith("DSCTM_")
                     or k.startswith("NCCL_") or k in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "captured_utc": _utc(),
    }


class RunDirectory:
    """Owns one run directory and enforces the completeness contract."""

    def __init__(self, root: str | Path, task: Any, is_main: bool = True) -> None:
        self.path = Path(root) / task.task_id
        self.task = task
        self.is_main = is_main
        if is_main:
            self.path.mkdir(parents=True, exist_ok=True)
        self.waivers: dict[str, str] = {}

    # -- opening --------------------------------------------------------- #
    def open(self, resolved_config: dict[str, Any], dataset_hashes: dict[str, Any],
             split_hashes: dict[str, Any], plan_digest: str | None = None) -> None:
        if not self.is_main:
            return
        import yaml
        self._write("command.txt", " ".join([sys.executable, *sys.argv]) + "\n")
        self._write("resolved_config.yaml", yaml.safe_dump(
            {"task": self.task.to_dict(), "resolved": resolved_config,
             "plan_digest": plan_digest}, sort_keys=True))
        self._json("environment.json", environment_metadata())
        self._json("git.json", git_metadata())
        self._json("slurm.json", slurm_metadata())
        self._json("hardware.json", hardware_metadata())
        self._json("dataset_hashes.json", dataset_hashes)
        self._json("split_hashes.json", split_hashes)
        self._json("status.json", {"status": "running", "started_utc": _utc()})
        # stdout/stderr are created now so a job killed early still has the files the
        # contract requires; sbatch redirects append to them.
        for name in ("stdout.log", "stderr.log"):
            (self.path / name).touch()

    # -- results --------------------------------------------------------- #
    def write_metrics(self, metrics: dict[str, Any]) -> None:
        if self.is_main:
            self._json("metrics.json", metrics)

    def write_predictions(self, records: list[Any]) -> None:
        """Parquet if pyarrow is present; otherwise JSONL plus a recorded waiver.

        Never silently skipped — the waiver appears in status.json and the auditor sees it.
        """
        if not self.is_main:
            return
        rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]
        try:
            import pandas as pd
            pd.DataFrame(rows).to_parquet(self.path / "predictions.parquet", index=False)
        except Exception as exc:
            with (self.path / "predictions.jsonl").open("w") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=str) + "\n")
            self.waivers["predictions.parquet"] = (
                f"pyarrow unavailable ({type(exc).__name__}); wrote predictions.jsonl "
                f"instead. Install pyarrow on PARAM to satisfy the contract."
            )

    def write_checkpoint(self, payload_path: Path | None, reason: str | None = None) -> None:
        if not self.is_main:
            return
        if payload_path is None:
            self.waivers["checkpoint.pt"] = reason or "checkpoint retention disabled"

    # -- closing --------------------------------------------------------- #
    def finalize(self, status: str, **fields: Any) -> dict[str, Any]:
        """Write the receipt, audit the contract, then write the final status.

        Order matters and is not arbitrary:

        1. ``_receipt()`` first, so ``receipt.sha256`` exists when the contract is audited.
        2. ``audit()`` second, over the now-complete file set.
        3. ``status.json`` last, because it records the receipt and the verdict.

        ``status.json`` is deliberately EXCLUDED from the receipt hash. Including it would
        be circular — the receipt would have to cover a file that quotes the receipt. The
        receipt therefore binds the *evidence* (config, provenance, metrics, predictions),
        and ``status.json`` binds the *verdict* to that evidence by quoting its hash.
        """
        if not self.is_main:
            return {}
        receipt = self._receipt()
        audit = self.audit()
        effective = status
        if status == "completed" and not audit["complete"]:
            effective = "infrastructure_failed"
            fields["contract_violation"] = (
                f"missing required file(s): {audit['missing']}. A run is not complete "
                f"without them, regardless of exit code."
            )
        self._json("status.json", {
            "status": effective, "requested_status": status,
            "finished_utc": _utc(), "task_id": self.task.task_id,
            "contract": audit, "waivers": self.waivers,
            "receipt_sha256": receipt, **fields,
        })
        return {"status": effective, "receipt": receipt, "contract": audit}

    def audit(self) -> dict[str, Any]:
        present, missing = [], []
        for name in REQUIRED_FILES:
            if (self.path / name).exists():
                present.append(name)
            elif name in WAIVABLE and name in self.waivers:
                present.append(f"{name} (waived)")
            else:
                missing.append(name)
        return {"present": present, "missing": missing, "complete": not missing,
                "required_count": len(REQUIRED_FILES)}

    # -- helpers --------------------------------------------------------- #
    def _write(self, name: str, text: str) -> Path:
        target = self.path / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        return target

    def _json(self, name: str, payload: Any) -> Path:
        return self._write(name, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    #: Excluded from the receipt hash. ``receipt.sha256`` cannot hash itself, and
    #: ``status.json`` quotes the receipt, so including it would be circular.
    _RECEIPT_EXCLUDE = ("receipt.sha256", "status.json")

    def _receipt(self) -> str:
        """SHA-256 binding every evidence file in the directory. See finalize() for why
        ``status.json`` is excluded."""
        digest = hashlib.sha256()
        for p in sorted(self.path.rglob("*")):
            if p.is_file() and p.name not in self._RECEIPT_EXCLUDE and not p.name.endswith(".tmp"):
                digest.update(p.relative_to(self.path).as_posix().encode())
                digest.update(hashlib.sha256(p.read_bytes()).digest())
        value = digest.hexdigest()
        self._write("receipt.sha256", f"{value}  {self.task.task_id}\n")
        return value
