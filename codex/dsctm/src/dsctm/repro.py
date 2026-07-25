"""Reproducibility: seeding, determinism modes, and environment capture.

Two execution modes (master-prompt §6):
  - "scientific": deterministic algorithms, fixed seeds, deterministic dataloading.
  - "systems":    representative high-performance execution (determinism not forced,
                  since deterministic kernels can distort timing).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys


def set_seed(seed: int, mode: str = "scientific") -> None:
    """Seed all RNGs and configure determinism for the requested mode."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if mode == "scientific":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        elif mode == "systems":
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(f"unknown mode {mode!r}")
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for deterministic multi-worker loading."""
    import numpy as np

    base = (int(os.environ.get("PYTHONHASHSEED", "0")) + worker_id) % (2**32)
    np.random.seed(base)
    random.seed(base)


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def capture_environment() -> dict:
    """Snapshot the runtime environment for a run's environment.txt / run.json."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_model"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    env["git_commit"] = _sh("git rev-parse HEAD") or None
    env["git_branch"] = _sh("git rev-parse --abbrev-ref HEAD") or None
    env["nvidia_driver"] = _sh(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"
    )
    return env


def environment_hash(env: dict) -> str:
    return hashlib.sha256(json.dumps(env, sort_keys=True, default=str).encode()).hexdigest()[:16]


def pip_freeze() -> str:
    return _sh(f"{sys.executable} -m pip freeze")
