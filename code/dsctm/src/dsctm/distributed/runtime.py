"""Process-group lifecycle, rank discovery, device binding, and batch semantics.

Target: PARAM Utkarsh GPU partition — 2 x NVIDIA V100 SXM2 (sm_70, 16 GB HBM2) per node,
2 x Intel Xeon Gold 6248 (40 cores), 192 GB RAM, Mellanox InfiniBand HDR, SLURM 20.11.8,
CentOS 7.9.

Design notes that are not obvious:

* ``torch.cuda.set_device`` MUST happen before ``init_process_group`` with NCCL, or NCCL
  may bind every rank on a node to device 0 and silently serialise.
* V100 is sm_70: fp16 is supported, **bf16 is not**. ``autocast_dtype()`` refuses bf16 on
  sm_70 rather than letting PyTorch fall back and produce numbers nobody can explain.
* Falling back to a single process when torchrun env vars are absent is deliberate: the
  same training code must run under ``python script.py`` for a smoke test and under
  ``torchrun`` for the real job, without a second code path.
"""
from __future__ import annotations

import datetime as _dt
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .errors import PreflightFailure

DEFAULT_TIMEOUT_MINUTES = 30
_V100_CAPABILITY = (7, 0)


@dataclass(frozen=True)
class DistContext:
    """Immutable description of this process's place in the job."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    backend: str
    launched_distributed: bool
    node_count: int = 1
    job_id: str | None = None
    node_list: str | None = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Global rank 0. The ONLY rank permitted to write shared artifacts."""
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        """Local rank 0 — use for per-node work such as a node-local cache build."""
        return self.local_rank == 0

    def describe(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "device": str(self.device),
            "backend": self.backend,
            "launched_distributed": self.launched_distributed,
            "slurm_job_id": self.job_id,
            "slurm_node_list": self.node_list,
        }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise PreflightFailure(f"environment variable {name}={raw!r} is not an integer") from exc


def _detect_node_count() -> int:
    for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        if os.environ.get(key):
            return _env_int(key, 1)
    return 1


def select_backend(device_type: str) -> str:
    """NCCL for CUDA, gloo otherwise. Gloo is the CPU test path, never the PARAM path."""
    if device_type == "cuda":
        if not (dist.is_available() and dist.is_nccl_available()):
            raise PreflightFailure(
                "CUDA is present but this PyTorch build has no NCCL. "
                "Multi-GPU training on PARAM requires a NCCL-enabled build."
            )
        return "nccl"
    return "gloo"


def init_distributed(timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
                     backend: str | None = None) -> DistContext:
    """Initialise the process group, or return a valid single-process context.

    Reads RANK / LOCAL_RANK / WORLD_SIZE (set by torchrun). When they are absent the
    process runs single-process and ``launched_distributed`` is False — no process group
    is created, and every collective helper in this package degrades to identity.
    """
    launched = all(k in os.environ for k in ("RANK", "WORLD_SIZE"))
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))
    local_world_size = _env_int("LOCAL_WORLD_SIZE", _env_int("SLURM_NTASKS_PER_NODE", 1))

    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if local_rank >= visible:
            raise PreflightFailure(
                f"LOCAL_RANK={local_rank} but only {visible} CUDA device(s) are visible. "
                f"On PARAM this usually means --gres=gpu:N was smaller than "
                f"--ntasks-per-node. Requested tasks must not exceed allocated GPUs."
            )
        # MUST precede init_process_group for NCCL.
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    resolved_backend = backend or select_backend(device.type)

    if launched and world_size > 1:
        if not dist.is_available():
            raise PreflightFailure("torch.distributed is not available in this build")
        if not dist.is_initialized():
            dist.init_process_group(
                backend=resolved_backend,
                init_method="env://",
                timeout=_dt.timedelta(minutes=timeout_minutes),
            )
    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=max(1, local_world_size),
        device=device,
        backend=resolved_backend if (launched and world_size > 1) else "none",
        launched_distributed=bool(launched and world_size > 1),
        node_count=_detect_node_count(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        node_list=os.environ.get("SLURM_JOB_NODELIST"),
    )


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier(ctx: DistContext | None = None) -> None:
    if not is_initialized():
        return
    if ctx is not None and ctx.device.type == "cuda":
        dist.barrier(device_ids=[ctx.local_rank])
    else:
        dist.barrier()


def cleanup() -> None:
    """Barrier then destroy. Safe to call when no group exists."""
    if is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, ctx: DistContext | None = None,
                    rank_aware: bool = False, deterministic: bool = False) -> int:
    """Seed Python, NumPy and Torch.

    ``rank_aware=False`` (default) gives every rank the SAME seed. That is what we want
    for model construction and for anything that must agree across ranks. DDP also
    broadcasts module state from rank 0 at construction, so identical init is guaranteed
    either way; identical seeds additionally make the two mechanisms consistent.

    ``rank_aware=True`` offsets by rank. Use it only for per-rank stochasticity that must
    NOT be correlated across ranks (e.g. a rank-local augmentation stream). Never use it
    for model init.
    """
    effective = int(seed) + (ctx.rank if (rank_aware and ctx is not None) else 0)
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch has no warn_only
            torch.use_deterministic_algorithms(True)
    return effective


# --------------------------------------------------------------------------- #
# Mixed precision (V100 / sm_70)
# --------------------------------------------------------------------------- #
def compute_capability(device: torch.device | None = None) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    index = 0 if device is None or device.type != "cuda" else (device.index or 0)
    return torch.cuda.get_device_capability(index)


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """Resolve a precision name to a dtype, refusing combinations sm_70 cannot honour.

    fp32   -> None (autocast disabled)
    fp16   -> torch.float16   (V100 tensor cores; requires GradScaler)
    bf16   -> torch.bfloat16, but REFUSED on sm_70 — V100 has no bf16 tensor cores and
              PyTorch would silently emulate, producing numbers that look fine and are
              not comparable to anything.
    """
    precision = precision.lower()
    if precision in ("fp32", "float32", "none"):
        return None
    if precision in ("fp16", "float16", "half"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        cap = compute_capability(device)
        if device.type == "cuda" and cap is not None and cap < (8, 0):
            raise PreflightFailure(
                f"bf16 requested on compute capability {cap[0]}.{cap[1]}. "
                f"PARAM Utkarsh V100s are sm_70 and have no bf16 tensor cores. "
                f"Use precision='fp16' with GradScaler."
            )
        return torch.bfloat16
    raise PreflightFailure(f"unknown precision {precision!r}; expected fp32 | fp16 | bf16")


def assert_v100_ready(device: torch.device) -> dict[str, Any]:
    """Hard preflight for the PARAM GPU partition. Raises rather than degrading."""
    if device.type != "cuda":
        raise PreflightFailure("no CUDA device bound; this is not a PARAM GPU job")
    cap = compute_capability(device)
    props = torch.cuda.get_device_properties(device.index or 0)
    info = {
        "name": props.name,
        "capability": f"{cap[0]}.{cap[1]}" if cap else None,
        "total_memory_gib": round(props.total_memory / 2**30, 2),
        "multi_processor_count": props.multi_processor_count,
        "nccl_available": dist.is_nccl_available() if dist.is_available() else False,
        "torch_cuda": torch.version.cuda,
    }
    if cap != _V100_CAPABILITY:
        info["warning"] = (
            f"expected sm_70 (V100) but found sm_{cap[0]}{cap[1]}; "
            f"precision and scaling assumptions were written for V100"
        )
    if not info["nccl_available"]:
        raise PreflightFailure("NCCL unavailable in this PyTorch build; multi-GPU is impossible")
    return info


# --------------------------------------------------------------------------- #
# Global-batch semantics  (Gate 2 requirement: never silently multiply the batch)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BatchSemantics:
    per_rank_batch_size: int
    world_size: int
    grad_accum_steps: int
    effective_global_batch: int
    scientific_global_batch: int
    matches_scientific_intent: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def resolve_batch_semantics(scientific_global_batch: int, world_size: int,
                            grad_accum_steps: int = 1,
                            allow_uneven: bool = False) -> BatchSemantics:
    """Derive the per-rank batch size that PRESERVES the scientific global batch.

    The trap this closes: taking the single-GPU ``batch_size`` from the config and passing
    it unchanged to every rank. At world_size 2 that silently doubles the optimisation
    batch, changes the effective learning rate, and makes the multi-GPU run a different
    experiment from the single-GPU run it is being compared against.

    Here ``scientific_global_batch`` is the quantity the experiment fixes; the per-rank
    size is derived from it.
    """
    if scientific_global_batch <= 0 or world_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch size, world size and accumulation steps must be positive")
    denom = world_size * grad_accum_steps
    per_rank, remainder = divmod(scientific_global_batch, denom)
    note = ""
    if per_rank == 0:
        raise PreflightFailure(
            f"global batch {scientific_global_batch} cannot be split across "
            f"{world_size} ranks x {grad_accum_steps} accumulation steps; "
            f"reduce world size or accumulation, or raise the global batch"
        )
    if remainder:
        if not allow_uneven:
            raise PreflightFailure(
                f"global batch {scientific_global_batch} is not divisible by "
                f"world_size*accum={denom} (remainder {remainder}). Pass allow_uneven=True "
                f"only if you accept an effective global batch of {per_rank * denom}."
            )
        note = (f"NOT divisible: requested {scientific_global_batch}, "
                f"effective {per_rank * denom}, dropped {remainder}")
    effective = per_rank * denom
    return BatchSemantics(
        per_rank_batch_size=per_rank,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        effective_global_batch=effective,
        scientific_global_batch=scientific_global_batch,
        matches_scientific_intent=(effective == scientific_global_batch),
        note=note,
    )


# --------------------------------------------------------------------------- #
# Small collective helpers
# --------------------------------------------------------------------------- #
def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable object from ``src`` to every rank."""
    if not is_initialized():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def all_reduce_scalar(value: float, op: str = "sum",
                      device: torch.device | None = None) -> float:
    if not is_initialized():
        return float(value)
    use = device if device is not None and device.type == "cuda" else torch.device("cpu")
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=use)
    reduce_op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX,
                 "min": dist.ReduceOp.MIN}.get(op)
    if reduce_op is None and op != "mean":
        raise ValueError(f"unsupported reduction {op!r}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op == "mean" else reduce_op)
    result = float(tensor.item())
    return result / dist.get_world_size() if op == "mean" else result


def assert_agrees_across_ranks(value: Any, label: str) -> None:
    """Every rank must hold an identical value. Used for split/data/config hashes.

    A silent disagreement here means the ranks are training on different data or
    different splits, which produces a plausible-looking and completely invalid result.
    """
    if not is_initialized():
        return
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if len(set(map(repr, gathered))) != 1:
        raise PreflightFailure(
            f"{label} differs across ranks: {gathered}. "
            f"Ranks are not running the same experiment; aborting."
        )
