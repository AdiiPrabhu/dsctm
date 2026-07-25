"""Record rental-machine facts without starting a distributed workload."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from common import hardware_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/multigpu/preflight.json")
    args = parser.parse_args()

    payload = hardware_metadata()
    payload["nccl_available"] = torch.distributed.is_nccl_available()
    payload["torchrun_path"] = shutil.which("torchrun")
    if shutil.which("nvidia-smi"):
        payload["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "-q"], text=True, errors="replace"
        )
        payload["topology"] = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, errors="replace"
        )
    failures = []
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    if torch.cuda.device_count() < 2:
        failures.append("fewer than two visible physical GPUs")
    if not torch.distributed.is_nccl_available():
        failures.append("PyTorch NCCL backend is unavailable")
    if payload["torchrun_path"] is None:
        failures.append("torchrun is unavailable")
    payload["status"] = "pass" if not failures else "fail"
    payload["failures"] = failures
    write_json(Path(args.output), payload)
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()

