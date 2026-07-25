#!/usr/bin/env python
"""Measure the per-rank batch-size ceiling on one PARAM V100. Resolves BLOCKER B-008.

The engagement brief assumed 32 GB V100s. PARAM Utkarsh has **16 GB HBM2**
(PARAM_Utkarsh_User_Manual-v3.0-1.pdf p.10). Every batch size in every config is therefore
an assumption until it is measured on the real device.

This probe binary-searches the largest per-rank batch that completes a full
forward + backward + optimizer step without OOM, for each model and dataset shape, at both
fp32 and fp16. The scientific global batch is then chosen from the measured ceiling rather
than from a guess, and recorded so the choice is auditable.

DAIC-WOZ is the hard case: T = 2000, F = 88, D = 128, three branches of four residual
blocks, activations retained for backward.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dsctm.models import DMSTCN, DMSTCNConfig  # noqa: E402
from dsctm.models.baselines import build_baseline  # noqa: E402

# (name, T, F, n_classes, n_subjects) — the two real workloads.
SHAPES = [
    ("studentlife", 60, 8, 3, 47),
    ("daicwoz", 2000, 88, 2, 108),
]
MODELS = ["dmstcn", "lstm", "temporal-cnn", "transformer", "itransformer", "timesnet"]
MAX_BATCH = 512


def _build(model_name, T, F, C, n_subjects):
    if model_name == "dmstcn":
        return DMSTCN(DMSTCNConfig(input_dim=F, n_classes=C, n_subjects=n_subjects))
    return build_baseline(model_name, F, C, seq_len=T)


def _try_batch(model_name, T, F, C, n_subjects, batch, device, precision) -> tuple[bool, float]:
    """One full training step at this batch. Returns (fits, peak_GiB)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = _build(model_name, T, F, C, n_subjects).to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        X = torch.randn(batch, T, F, device=device)
        y = torch.randint(0, C, (batch,), device=device)
        s = torch.zeros(batch, dtype=torch.long, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        amp = torch.float16 if precision == "fp16" else None
        scaler = torch.cuda.amp.GradScaler(enabled=amp is not None)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = model(X, s, mask=mask) if model_name == "dmstcn" else model(X, None, mask=mask)
            loss = nn.functional.cross_entropy(out, y)
        if amp is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        del model, opt, X, y, s, mask, out, loss
        torch.cuda.empty_cache()
        return True, peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, float("nan")
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        if "out of memory" in str(exc).lower():
            return False, float("nan")
        raise


def ceiling(model_name, T, F, C, n_subjects, device, precision) -> dict:
    """Exponential probe then binary search for the largest working batch."""
    lo, hi = 0, 1
    peak_at_hi = float("nan")
    while hi <= MAX_BATCH:
        fits, peak = _try_batch(model_name, T, F, C, n_subjects, hi, device, precision)
        if not fits:
            break
        lo, peak_at_hi = hi, peak
        hi *= 2
    if lo == 0:
        return {"max_batch": 0, "peak_gib": None,
                "note": "OOM at batch=1 — this workload does not fit on a 16 GB V100"}
    low, high = lo, min(hi, MAX_BATCH)
    best_peak = peak_at_hi
    while low + 1 < high:
        mid = (low + high) // 2
        fits, peak = _try_batch(model_name, T, F, C, n_subjects, mid, device, precision)
        if fits:
            low, best_peak = mid, peak
        else:
            high = mid
    return {"max_batch": low, "peak_gib": round(best_peak, 3) if best_peak == best_peak else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device. Submit via scripts/param/memory_probe.sbatch.")
        sys.exit(1)

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    report = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": {"name": props.name, "capability": f"{cap[0]}.{cap[1]}",
                "total_memory_gib": round(props.total_memory / 2**30, 2)},
        "note": ("Resolves BLOCKER B-008. The brief assumed 32 GB V100s; measured capacity "
                 "is recorded above and every downstream batch size derives from it."),
        "results": [],
    }
    print(f"GPU: {props.name} sm_{cap[0]}{cap[1]} "
          f"{report['gpu']['total_memory_gib']} GiB")

    for dataset, T, F, C, n_subjects in SHAPES:
        for model_name in args.models:
            for precision in args.precisions:
                res = ceiling(model_name, T, F, C, n_subjects, device, precision)
                row = {"dataset": dataset, "model": model_name, "precision": precision,
                       "T": T, "F": F, **res}
                report["results"].append(row)
                print(f"  {dataset:12s} {model_name:14s} {precision:5s} "
                      f"max_batch={row['max_batch']:4d} peak={row.get('peak_gib')} GiB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten: {out}")

    dm = [r for r in report["results"] if r["model"] == "dmstcn" and r["dataset"] == "daicwoz"]
    if dm:
        print("\nD-MSTCN on DAIC-WOZ (the binding constraint):")
        for r in dm:
            print(f"  {r['precision']}: per-rank batch <= {r['max_batch']}")
        print("  Set the scientific global batch to a value divisible by world_size that "
              "keeps per-rank at or below this.")


if __name__ == "__main__":
    main()
