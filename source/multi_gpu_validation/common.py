"""Shared, dependency-light utilities for physical multi-GPU validation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def initialize_distributed(minimum_world_size: int = 2) -> DistributedContext:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"launch with torchrun; missing environment variables: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("physical CUDA GPUs are required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < minimum_world_size:
        raise RuntimeError(f"validation requires WORLD_SIZE >= {minimum_world_size}")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds visible physical CUDA device count")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(rank, local_rank, world_size, torch.device("cuda", local_rank))


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank_offset: int = 0) -> None:
    effective_seed = seed + rank_offset
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def enable_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_all_ranks_equal_text(value: str, label: str) -> None:
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if len(set(gathered)) != 1:
        raise AssertionError(f"{label} differs across ranks: {gathered}")


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / dist.get_world_size()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware_metadata(context: DistributedContext | None = None) -> dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "pci_bus_id": getattr(props, "pci_bus_id", None),
                }
            )
    result: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": torch.cuda.device_count(),
        "devices": devices,
        "git_commit": git_commit(),
    }
    if context is not None:
        result.update(
            rank=context.rank,
            local_rank=context.local_rank,
            world_size=context.world_size,
        )
    return result


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
