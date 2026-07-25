"""Synchronized failure propagation.

The failure mode this module exists to prevent: one rank raises, the other ranks keep
waiting inside a collective, and SLURM burns the whole wall-clock allocation on a job
that is already dead. On PARAM Utkarsh that is a wasted GPU-node reservation, and the
person who finds out is whoever reads the log 20 hours later.

Rule: every rank leaves a guarded region together, or every rank raises together.
"""
from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager

import torch
import torch.distributed as dist


class DistributedError(RuntimeError):
    """Base for distributed-layer failures."""


class RankFailure(DistributedError):
    """Raised on every rank when at least one rank failed inside a guarded region."""


class PreflightFailure(DistributedError):
    """Environment did not satisfy a hard requirement. Never degrade silently."""


class EvaluationCoverageError(DistributedError):
    """Gathered predictions do not cover the split exactly once."""


def _initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_ranks_ok(local_ok: bool, device: torch.device | None = None) -> bool:
    """True iff every rank reports ok. Collective; every rank must call it."""
    if not _initialized():
        return bool(local_ok)
    flag = torch.tensor([0 if local_ok else 1], dtype=torch.int64,
                        device=device if device is not None and device.type == "cuda" else "cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return int(flag.item()) == 0


@contextmanager
def fail_together(stage: str, device: torch.device | None = None):
    """Run a block so that a failure on ANY rank raises on EVERY rank.

    Usage::

        with fail_together("build_dataset", ctx.device):
            ds = build_studentlife(...)

    The local traceback is printed on the failing rank before the collective, so the
    real cause is visible in that rank's log even though every rank raises.
    """
    local_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - deliberately broad, re-raised below
        local_error = exc
        rank = dist.get_rank() if _initialized() else 0
        print(f"[rank {rank}] FAILURE in stage {stage!r}: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        ok = all_ranks_ok(local_error is None, device)
    if local_error is not None:
        raise local_error
    if not ok:
        raise RankFailure(
            f"stage {stage!r} failed on at least one other rank; "
            f"this rank is aborting so the job does not hang in a collective"
        )


def hard_abort(message: str, code: int = 42) -> None:
    """Last-resort exit that does not leave peers blocked in a collective.

    Used when the process group itself is unusable and a clean raise cannot be
    propagated. Prefer ``fail_together``; this is the backstop.
    """
    print(f"FATAL: {message}", file=sys.stderr, flush=True)
    try:
        if _initialized():
            dist.destroy_process_group()
    except Exception:  # pragma: no cover - best effort during teardown
        pass
    os._exit(code)
