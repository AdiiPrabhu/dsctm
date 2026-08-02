"""Rank-aware artifact writing.

Two ranks writing the same run directory corrupts it: interleaved JSON, a metrics file
that is a mix of two ranks' views, a registry entry written twice. Every shared artifact
goes through rank 0 here.

Non-zero ranks are not silenced — they get their own ``rank<N>.log`` inside the run
directory, so a failure that only manifests on rank 1 is still diagnosable after the job
ends. That matters when the person reading the log is not the person who wrote the code.
"""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .runtime import DistContext, barrier


def rank_zero_only(fn: Callable) -> Callable:
    """Execute only on global rank 0. Expects a ``ctx`` kwarg or first positional ctx."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = kwargs.get("ctx")
        if ctx is None:
            ctx = next((a for a in args if isinstance(a, DistContext)), None)
        if ctx is None or ctx.is_main:
            return fn(*args, **kwargs)
        return None

    return wrapper


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Atomic JSON write: temp file then replace. A killed job never leaves half a file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


class RunLogger:
    """Owns one run directory. Rank 0 writes shared artifacts; every rank writes its own log.

    Lustre note: PARAM's filesystem is Lustre, which is poor at many small files. Per-step
    artifacts are deliberately NOT written; the per-rank log is opened once and appended,
    and shared artifacts are written once at their natural checkpoint.
    """

    def __init__(self, run_dir: str | Path, ctx: DistContext, echo: bool = True) -> None:
        self.ctx = ctx
        self.run_dir = Path(run_dir)
        self.echo = echo
        if ctx.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        barrier(ctx)
        if not self.run_dir.exists():  # non-zero rank on a shared FS that lagged
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._rank_log = self.run_dir / f"rank{ctx.rank}.log"
        self._handle = self._rank_log.open("a", encoding="utf-8")

    # -- per-rank ---------------------------------------------------------- #
    def log(self, message: str) -> None:
        """Append to this rank's own log. Safe on every rank."""
        line = f"[rank {self.ctx.rank}] {message}"
        self._handle.write(line + "\n")
        self._handle.flush()
        if self.echo and self.ctx.is_main:
            print(line, file=sys.stdout, flush=True)

    def log_all_ranks(self, message: str) -> None:
        """Append on every rank and echo on every rank. For failure paths only."""
        line = f"[rank {self.ctx.rank}] {message}"
        self._handle.write(line + "\n")
        self._handle.flush()
        print(line, file=sys.stderr, flush=True)

    # -- rank 0 only ------------------------------------------------------- #
    def write_json(self, name: str, payload: Any) -> Path | None:
        if not self.ctx.is_main:
            return None
        return write_json_atomic(self.run_dir / name, payload)

    def write_text(self, name: str, text: str) -> Path | None:
        if not self.ctx.is_main:
            return None
        destination = self.run_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    def mark_status(self, status: str, **fields: Any) -> Path | None:
        """Write ``status.json``. A run without this file is NOT complete."""
        valid = {"running", "completed", "model_failed", "infrastructure_failed",
                 "cancelled", "excluded_by_prespecified_rule"}
        if status not in valid:
            raise ValueError(f"invalid status {status!r}; expected one of {sorted(valid)}")
        return self.write_json("status.json", {"status": status, **fields})

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # pragma: no cover - teardown best effort
            pass


REQUIRED_RUN_FILES = (
    "command.txt", "resolved_config.yaml", "environment.json", "git.json",
    "slurm.json", "hardware.json", "dataset_hashes.json", "split_hashes.json",
    "stdout.log", "stderr.log", "metrics.json", "predictions.parquet",
    "checkpoint.pt", "status.json", "receipt.sha256",
)


def audit_run_directory(run_dir: str | Path,
                        required: tuple[str, ...] = REQUIRED_RUN_FILES) -> dict[str, Any]:
    """Check a run directory against the Gate 4 contract.

    A run is complete only when every required file is present, regardless of exit code.
    Used by the Gate 5 fail-closed auditor and by ``mark_status`` callers before they
    claim completion.
    """
    directory = Path(run_dir)
    present, missing = [], []
    for name in required:
        (present if (directory / name).exists() else missing).append(name)
    return {
        "run_dir": str(directory),
        "exists": directory.exists(),
        "present": present,
        "missing": missing,
        "complete": not missing,
    }
