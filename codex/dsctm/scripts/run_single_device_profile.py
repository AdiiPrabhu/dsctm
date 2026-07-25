#!/usr/bin/env python
"""EXP-6.1 single-device inference profile with raw synchronized timings."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dsctm.models import DMSTCN, DMSTCNConfig


def profile_case(name, F, T, C, n_subjects, batch_size, warmup=10, repeats=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DMSTCN(DMSTCNConfig(input_dim=F, n_classes=C, n_subjects=n_subjects)).to(device).eval()
    X = torch.randn(batch_size, T, F, device=device)
    s = torch.zeros(batch_size, dtype=torch.long, device=device)
    mask = torch.ones(batch_size, T, dtype=torch.bool, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(warmup):
            model(X, s, mask=mask)
        sync()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter_ns()
            model(X, s, mask=mask)
            sync()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
    a = np.asarray(samples)
    return {
        "case": name, "device": device, "F": F, "T": T, "batch_size": batch_size,
        "warmup": warmup, "repeats": repeats, "precision": "float32",
        "latency_ms": {"median": float(np.median(a)), "p95": float(np.quantile(a, .95)),
                       "p99": float(np.quantile(a, .99)), "mean": float(a.mean()),
                       "std": float(a.std())},
        "throughput_samples_per_s": float(batch_size / (np.median(a) / 1000)),
        "peak_allocated_mib": (float(torch.cuda.max_memory_allocated() / 2**20)
                               if device == "cuda" else None),
        "timing_samples_ms": samples,
    }


def main():
    result = {
        "experiment": "EXP-6.1", "mode": "systems_benchmark",
        "host": platform.node(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cases": [
            profile_case("studentlife", 8, 60, 3, 47, 32),
            profile_case("daicwoz", 88, 2000, 2, 108, 8),
        ],
    }
    out = Path("artifacts/resubmission/systems")
    out.mkdir(parents=True, exist_ok=True)
    (out / "single_device_profile.json").write_text(json.dumps(result, indent=2))
    for case in result["cases"]:
        print(case["case"], case["latency_ms"],
              f"throughput={case['throughput_samples_per_s']:.2f}/s",
              f"peak={case['peak_allocated_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
