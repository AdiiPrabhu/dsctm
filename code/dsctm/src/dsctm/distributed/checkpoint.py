"""Distributed checkpoint save / resume.

PARAM Utkarsh caps wall-clock at 72:00:00 and jobs can be preempted. A long tuning array
that cannot resume is a long tuning array that gets re-run from zero.

Requirement (Gate 2): a resumed run must continue **identically** to an uninterrupted one.
That means restoring more than weights — optimizer moments, scheduler position, AMP scale,
epoch, global step, best-dev score, patience counter, and every RNG stream. Miss the RNG
and the resumed run draws different shuffles and different dropout masks, which is a
silently different experiment.

Only rank 0 writes. Writes are atomic (`.tmp` then `os.replace`) so a job killed mid-write
leaves the previous good checkpoint intact rather than a truncated file.
"""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .runtime import DistContext, barrier, is_initialized

CHECKPOINT_VERSION = 1


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu() if hasattr(state["torch_cpu"], "cpu")
                        else state["torch_cpu"])
    cuda_states = state.get("torch_cuda_all")
    if cuda_states and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if len(cuda_states) == available:
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            # Resuming on a different GPU count: restore what we can rather than crash,
            # but say so — the run is no longer bit-identical to the original.
            for i in range(min(available, len(cuda_states))):
                torch.cuda.set_rng_state(cuda_states[i], device=i)
            print(f"WARNING: checkpoint has {len(cuda_states)} CUDA RNG states but "
                  f"{available} device(s) are visible; bit-identical resume is not "
                  f"guaranteed.", flush=True)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str | Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, scaler: Any = None,
                    epoch: int = 0, global_step: int = 0,
                    best_score: float = float("-inf"), patience: int = 0,
                    resolved_config: dict[str, Any] | None = None,
                    dataset_hash: str | None = None, split_hash: str | None = None,
                    ctx: DistContext | None = None, extra: dict[str, Any] | None = None
                    ) -> Path | None:
    """Atomically write a full-state checkpoint. Rank 0 only; other ranks return None."""
    if ctx is not None and not ctx.is_main:
        barrier(ctx)
        return None

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_score": float(best_score),
        "patience": int(patience),
        "rng": _rng_state(),
        "resolved_config": resolved_config or {},
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "world_size": ctx.world_size if ctx is not None else 1,
        "extra": extra or {},
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)

    if ctx is not None:
        barrier(ctx)
    return destination


def load_checkpoint(path: str | Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, scaler: Any = None,
                    map_location: Any = "cpu", restore_rng: bool = True,
                    expect_dataset_hash: str | None = None,
                    expect_split_hash: str | None = None) -> dict[str, Any]:
    """Restore full training state. Returns the bookkeeping fields.

    ``expect_*_hash`` are refuse-to-continue guards: resuming a checkpoint against a
    different dataset or split silently produces a result attributed to the wrong data.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)

    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint version {version} != expected {CHECKPOINT_VERSION}")
    for label, expected, actual in (
        ("dataset_hash", expect_dataset_hash, payload.get("dataset_hash")),
        ("split_hash", expect_split_hash, payload.get("split_hash")),
    ):
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(
                f"{label} mismatch: checkpoint has {actual!r}, current run has {expected!r}. "
                f"Refusing to resume onto different data."
            )

    _unwrap(model).load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng" in payload:
        _restore_rng(payload["rng"])

    return {
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "best_score": payload["best_score"],
        "patience": payload["patience"],
        "resolved_config": payload.get("resolved_config", {}),
        "dataset_hash": payload.get("dataset_hash"),
        "split_hash": payload.get("split_hash"),
        "world_size_at_save": payload.get("world_size"),
        "extra": payload.get("extra", {}),
    }


def state_digest(model: torch.nn.Module) -> str:
    """Stable SHA-256 over the model state dict. Used to prove rank agreement."""
    digest = hashlib.sha256()
    for name, tensor in sorted(_unwrap(model).state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def find_latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    """Most recent checkpoint by modification time, or None. Ignores partial `.tmp` files."""
    candidates = [p for p in Path(directory).glob(pattern) if not p.name.endswith(".tmp")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
