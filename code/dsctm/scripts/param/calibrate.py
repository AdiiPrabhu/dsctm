#!/usr/bin/env python
"""Run a SMALL sample of real tasks, measure them, and extrapolate to the full 294.

    python scripts/param/calibrate.py --sample 1        # one task per family (5 tasks)
    python scripts/param/calibrate.py --sample 1 --families tuning-daicwoz
    python scripts/param/calibrate.py --extrapolate calibration.json

Purpose: never submit a 294-task array on an estimate. Run a handful of REAL tasks on the
real hardware, measure what they actually cost, then decide.

What is measured per task: wall seconds, peak GPU memory, epochs actually executed (early
stopping means the a-priori epoch guess is the largest unknown), and the resulting run
directory's contract status. Extrapolation multiplies the measured per-family mean by the
family size — no hidden fudge factor, and the sample size is printed alongside so a
one-task extrapolation is visibly a one-task extrapolation.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dsctm.campaign import FAMILIES, build_plan   # noqa: E402


def run_one(family: str, index: int, extra_env: dict) -> dict:
    env = {**os.environ, **extra_env}
    cmd = [sys.executable, str(REPO / "scripts/param/run_task.py"),
           "--family", family, "--index", str(index)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    peak_gib = None
    try:
        import torch
        if torch.cuda.is_available():
            peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception:
        pass

    results_root = Path(os.environ.get("DSCTM_RESULTS_ROOT", "results")) / family
    task_id = build_plan(family)[index].task_id
    run_dir = results_root / task_id
    status, epochs, contract = None, None, None
    if (run_dir / "status.json").exists():
        blob = json.loads((run_dir / "status.json").read_text())
        status = blob.get("status")
        contract = blob.get("contract", {}).get("complete")
    if (run_dir / "metrics.json").exists():
        m = json.loads((run_dir / "metrics.json").read_text())
        epochs = (m.get("best_epoch") if isinstance(m.get("best_epoch"), int)
                  else len(m.get("curve", []) or []) or None)

    return {"family": family, "index": index, "task_id": task_id,
            "wall_seconds": round(wall, 2), "returncode": proc.returncode,
            "status": status, "contract_complete": contract,
            "epochs_executed": epochs, "peak_gib": peak_gib,
            "stderr_tail": proc.stderr.strip().splitlines()[-5:] if proc.stderr else []}


def extrapolate(cal: dict) -> dict:
    per_family: dict[str, list[float]] = {}
    for r in cal["runs"]:
        if r["status"] == "completed":
            per_family.setdefault(r["family"], []).append(r["wall_seconds"])

    rows, total_lo, total_hi = [], 0.0, 0.0
    for fam in sorted(FAMILIES):
        n = len(build_plan(fam))
        samples = per_family.get(fam, [])
        if not samples:
            rows.append({"family": fam, "tasks": n, "sampled": 0,
                         "mean_seconds": None, "gpu_hours": None,
                         "note": "no completed sample — not extrapolated"})
            continue
        mean = sum(samples) / len(samples)
        # A single sample cannot bound variance. Widen deliberately rather than pretend.
        spread = 1.5 if len(samples) < 3 else 1.2
        lo = n * mean / 3600.0
        hi = lo * spread
        total_lo += lo
        total_hi += hi
        rows.append({"family": fam, "tasks": n, "sampled": len(samples),
                     "mean_seconds": round(mean, 1),
                     "gpu_hours": round(lo, 2), "gpu_hours_upper": round(hi, 2),
                     "spread_factor": spread})
    return {"rows": rows, "science_gpu_hours_low": round(total_lo, 2),
            "science_gpu_hours_high": round(total_hi, 2),
            "sample_note": ("Extrapolated from a SMALL sample. With <3 samples per family "
                            "the upper bound is the mean x1.5, which is a guess about "
                            "variance, not a measurement of it.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1, help="tasks per family")
    ap.add_argument("--families", nargs="*", default=sorted(FAMILIES))
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="cap epochs for a fast probe (extrapolation then UNDERSTATES cost)")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--extrapolate", default=None)
    args = ap.parse_args()

    if args.extrapolate:
        cal = json.loads(Path(args.extrapolate).read_text())
        est = extrapolate(cal)
        print(f"{'family':24s} {'tasks':>6s} {'sampled':>8s} {'mean s':>9s} {'GPU-h':>8s} {'upper':>8s}")
        print("-" * 68)
        for r in est["rows"]:
            if r.get("mean_seconds") is None:
                print(f"{r['family']:24s} {r['tasks']:6d} {0:8d} {'--':>9s} {'--':>8s} {'--':>8s}")
            else:
                print(f"{r['family']:24s} {r['tasks']:6d} {r['sampled']:8d} "
                      f"{r['mean_seconds']:9.1f} {r['gpu_hours']:8.2f} {r['gpu_hours_upper']:8.2f}")
        print("-" * 68)
        print(f"{'SCIENCE TOTAL':24s} {'':>6s} {'':>8s} {'':>9s} "
              f"{est['science_gpu_hours_low']:8.2f} {est['science_gpu_hours_high']:8.2f}")
        print(f"\n{est['sample_note']}")
        print("\nAdd the systems experiments (~30-60 GPU-h) for the full request.")
        Path(args.extrapolate).with_suffix(".extrapolated.json").write_text(
            json.dumps(est, indent=2) + "\n")
        return 0

    extra = {}
    if args.max_epochs:
        extra["DSCTM_MAX_EPOCHS"] = str(args.max_epochs)
        print(f"NOTE: capping epochs at {args.max_epochs}. The extrapolation will "
              f"UNDERSTATE real cost — use it to prove the pipeline, not to size the ask.\n")

    runs = []
    for fam in args.families:
        n = len(build_plan(fam))
        # Spread the sample across the family rather than always taking index 0, so a
        # cheap first model does not stand in for an expensive later one.
        picks = sorted({int(i * (n - 1) / max(1, args.sample - 1)) if args.sample > 1 else 0
                        for i in range(args.sample)})
        for idx in picks:
            print(f"--- {fam}[{idx}] ---", flush=True)
            r = run_one(fam, idx, extra)
            runs.append(r)
            print(f"    {r['status']}  {r['wall_seconds']}s  "
                  f"epochs={r['epochs_executed']}  peak={r['peak_gib']} GiB", flush=True)
            if r["status"] != "completed":
                for line in r["stderr_tail"]:
                    print(f"    ! {line}", flush=True)

    cal = {"generated_utc": time.strftime("%FT%TZ", time.gmtime()),
           "sample_per_family": args.sample,
           "epoch_cap": args.max_epochs, "runs": runs}
    Path(args.out).write_text(json.dumps(cal, indent=2) + "\n")
    print(f"\nwritten: {args.out}")
    ok = sum(1 for r in runs if r["status"] == "completed")
    print(f"{ok}/{len(runs)} task(s) completed")
    print(f"\nNext: python scripts/param/calibrate.py --extrapolate {args.out}")
    return 0 if ok == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
