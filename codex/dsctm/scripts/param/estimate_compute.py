#!/usr/bin/env python
"""Resolve BLOCKER B-006: turn the 294-task plan into a GPU-hour number for CDAC.

    python scripts/param/estimate_compute.py                       # a-priori bracket
    python scripts/param/estimate_compute.py --probe <memory_probe.json>   # measured

Two modes:

  a-priori   uses conservative per-step timings for a V100 and states them as assumptions.
             Good enough to request an allocation. Bracketed low/high, never a single
             number pretending to be precise.
  measured   reads the memory-probe output and replaces the assumption with the real
             per-step cost on the actual hardware.

Nothing here is a promise. It is an estimate with its inputs written down, which is what an
allocation request needs.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dsctm.campaign import FAMILIES, build_plan   # noqa: E402

# Per-optimizer-step wall time on one V100, fp16. Conservative brackets.
# StudentLife: T=60, F=8   -> tiny; dominated by kernel launch overhead.
# DAIC-WOZ:    T=2000, F=88 -> activation-heavy; the real cost driver.
STEP_MS = {
    "studentlife": {"low": 8.0,  "high": 25.0},
    "daicwoz":     {"low": 60.0, "high": 220.0},
}
# samples, batch, and the epoch budget actually reachable under early stopping
SHAPE = {
    "studentlife": dict(n=2160, batch=64, max_epochs=100, typical_epochs=35, folds=5),
    "daicwoz":     dict(n=163,  batch=32, max_epochs=40,  typical_epochs=18, folds=1),
}
OVERHEAD = 1.25   # data loading, evaluation each epoch, checkpoint I/O on Lustre


def task_hours(dataset: str, bound: str, epochs_key: str = "typical_epochs") -> float:
    s = SHAPE[dataset]
    steps_per_epoch = max(1, s["n"] // s["batch"])
    steps = steps_per_epoch * s[epochs_key] * s["folds"]
    return steps * STEP_MS[dataset][bound] / 1000.0 / 3600.0 * OVERHEAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=None, help="memory_probe JSON to calibrate from")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    note = "a-priori estimate; per-step timings are ASSUMPTIONS listed below"
    if args.probe and Path(args.probe).exists():
        blob = json.loads(Path(args.probe).read_text())
        note = f"calibrated from {args.probe} on {blob.get('gpu', {}).get('name')}"
        # The probe records max batch, not step time; if a future probe adds timing, wire
        # it here. Until then, say plainly that it did not calibrate the timing.
        note += " (batch ceiling only — step timing still assumed)"

    rows, totals = [], {"low": 0.0, "high": 0.0}
    for fam in sorted(FAMILIES):
        tasks = build_plan(fam)
        ds = "studentlife" if "studentlife" in fam else (
            "daicwoz" if "daicwoz" in fam else "studentlife")
        lo = len(tasks) * task_hours(ds, "low")
        hi = len(tasks) * task_hours(ds, "high")
        totals["low"] += lo
        totals["high"] += hi
        rows.append({"family": fam, "tasks": len(tasks), "dataset": ds,
                     "gpu_hours_low": round(lo, 1), "gpu_hours_high": round(hi, 1)})

    # Systems experiments: short jobs, but multi-GPU, so node-hours matter more.
    systems = [
        {"job": "2gpu_ddp_smoke", "gpus": 2, "hours": 0.5},
        {"job": "memory_probe", "gpus": 1, "hours": 1.0},
        {"job": "scaling 1/2/4 GPU x strong+weak x 5 rep", "gpus": 4, "hours": 3.0},
        {"job": "sap_tcp_sweep (8-cell default) x 4 modes", "gpus": 4, "hours": 4.0},
        {"job": "extract_features (CPU partition)", "gpus": 0, "hours": 6.0},
    ]
    sys_gpu_hours = sum(s["gpus"] * s["hours"] for s in systems)

    print(f"COMPUTE ESTIMATE — {note}\n")
    print(f"{'family':24s} {'tasks':>6s} {'dataset':>12s} {'GPU-h low':>10s} {'GPU-h high':>11s}")
    print("-" * 68)
    for r in rows:
        print(f"{r['family']:24s} {r['tasks']:6d} {r['dataset']:>12s} "
              f"{r['gpu_hours_low']:10.1f} {r['gpu_hours_high']:11.1f}")
    print("-" * 68)
    print(f"{'SCIENCE TOTAL':24s} {sum(r['tasks'] for r in rows):6d} {'':>12s} "
          f"{totals['low']:10.1f} {totals['high']:11.1f}")
    print(f"\n{'systems experiments':24s} {'':>6s} {'':>12s} {sys_gpu_hours:10.1f} "
          f"{sys_gpu_hours * 2:11.1f}")
    grand_lo = totals["low"] + sys_gpu_hours
    grand_hi = totals["high"] + sys_gpu_hours * 2
    print(f"{'GRAND TOTAL (GPU-hours)':24s} {'':>6s} {'':>12s} {grand_lo:10.1f} "
          f"{grand_hi:11.1f}")

    print("\nWhat to request from CDAC:")
    print(f"  GPU-hours       : {int(grand_hi * 1.3)}  (high bracket + 30% contingency)")
    print(f"  Peak concurrency: 4 GPUs (arrays throttled %4) + one 2-node reservation")
    print(f"  Longest job     : 12 h (within the 72 h limit)")
    print(f"  CPU-partition   : ~6 h x 48 cores for eGeMAPS extraction")
    print(f"  Storage         : ~250 GB Lustre scratch")

    print("\nAssumptions (change these and the number changes):")
    for ds, b in STEP_MS.items():
        s = SHAPE[ds]
        print(f"  {ds:12s} {b['low']}-{b['high']} ms/step, batch {s['batch']}, "
              f"{s['typical_epochs']} epochs under early stop, {s['folds']} fold(s)")
    print(f"  overhead factor {OVERHEAD}x for loading, per-epoch eval and Lustre I/O")
    print("\nThe DAIC-WOZ step time is the dominant uncertainty. memory_probe.sbatch")
    print("measures the batch ceiling; re-run this with --probe once it exists.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"note": note, "families": rows, "systems": systems,
             "science_gpu_hours": totals, "systems_gpu_hours": sys_gpu_hours,
             "grand_total_low": grand_lo, "grand_total_high": grand_hi,
             "request_with_contingency": int(grand_hi * 1.3),
             "assumptions": {"step_ms": STEP_MS, "shape": SHAPE, "overhead": OVERHEAD}},
            indent=2) + "\n")
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
