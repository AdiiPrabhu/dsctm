#!/usr/bin/env python
"""Gate 7 — strong and weak scaling for full-model DDP. The CONTROL for SAP/TCP.

    torchrun ... scripts/param/scaling_benchmark.py --mode strong --repeats 5

Strong and weak scaling are measured separately and NEVER mixed in one claim:

  strong  effective GLOBAL batch fixed; per-rank batch shrinks as ranks are added.
          Answers "does adding GPUs make this run faster?"
  weak    per-rank batch fixed; global batch grows with ranks.
          Answers "can I process more data by adding GPUs?"

Efficiency is only meaningful against a 1-GPU baseline measured on the SAME hardware, so
the single-rank reference is recorded in every report and eta is left null when absent.

Real datasets here are tiny (StudentLife 2,160 windows; DAIC-WOZ 275 sessions), so a
synthetic workload is used for the systems measurement and is LABELLED as such. Replicated
or synthetic samples are never counted as additional scientific subjects.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, statistics, sys, time
from pathlib import Path

import torch, torch.distributed as dist, torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dsctm.distributed import (autocast_dtype, build_grad_scaler, cleanup,   # noqa: E402
                               init_distributed, resolve_batch_semantics, seed_everything,
                               wrap_ddp)
from dsctm.models import DMSTCN, DMSTCNConfig                                 # noqa: E402

WORKLOADS = {
    "studentlife_like": dict(T=60,   F=8,  C=3, n_subjects=48,  global_batch=256),
    "daicwoz_like":     dict(T=2000, F=88, C=2, n_subjects=108, global_batch=32),
    "synthetic_large":  dict(T=4000, F=128, C=3, n_subjects=512, global_batch=64),
}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(ctx, spec, mode, repeats, warmup, precision):
    ws = ctx.world_size
    if mode == "strong":
        sem = resolve_batch_semantics(spec["global_batch"], ws, allow_uneven=True)
        per_rank, global_batch = sem.per_rank_batch_size, sem.effective_global_batch
    else:  # weak: per-rank fixed at the single-rank global batch
        per_rank = spec["global_batch"]
        global_batch = per_rank * ws

    seed_everything(0, ctx)
    model = DMSTCN(DMSTCNConfig(input_dim=spec["F"], n_classes=spec["C"],
                                n_subjects=spec["n_subjects"]))
    model = wrap_ddp(model, ctx)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    amp = autocast_dtype(precision, ctx.device)
    scaler = build_grad_scaler(precision, ctx.device)

    X = torch.randn(per_rank, spec["T"], spec["F"], device=ctx.device)
    y = torch.randint(0, spec["C"], (per_rank,), device=ctx.device)
    s = torch.zeros(per_rank, dtype=torch.long, device=ctx.device)
    mask = torch.ones(per_rank, spec["T"], dtype=torch.bool, device=ctx.device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=ctx.device.type, dtype=amp, enabled=amp is not None):
            loss = nn.functional.cross_entropy(model(X, s, mask=mask), y)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

    for _ in range(warmup):
        step()
    _sync(ctx.device)
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(ctx.device)
    if dist.is_initialized():
        dist.barrier()

    samples = []
    for _ in range(repeats):
        _sync(ctx.device); t0 = time.perf_counter_ns()
        step()
        _sync(ctx.device); samples.append((time.perf_counter_ns() - t0) / 1e6)

    # Slowest rank defines the iteration; that is what a synchronous step actually costs.
    local = torch.tensor(samples, dtype=torch.float64, device=ctx.device)
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
    ms = local.tolist()
    median = statistics.median(ms)
    return {
        "mode": mode, "world_size": ws, "node_count": ctx.node_count,
        "per_rank_batch": per_rank, "global_batch": global_batch,
        "precision": precision, "repeats": repeats, "warmup": warmup,
        "iteration_ms": {"median": median, "mean": statistics.mean(ms),
                         "std": statistics.pstdev(ms) if len(ms) > 1 else 0.0,
                         "p95": sorted(ms)[max(0, int(0.95 * len(ms)) - 1)],
                         "p99": sorted(ms)[max(0, int(0.99 * len(ms)) - 1)],
                         "min": min(ms), "max": max(ms)},
        "throughput_samples_per_s": global_batch / (median / 1000.0),
        "peak_memory_gib": (torch.cuda.max_memory_allocated(ctx.device) / 2**30
                            if ctx.device.type == "cuda" else None),
        "raw_iteration_ms": ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strong", "weak", "both"], default="strong")
    ap.add_argument("--workloads", nargs="*", default=list(WORKLOADS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", default=os.environ.get("DSCTM_PRECISION", "fp16"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None,
                    help="path to the world_size=1 report, for efficiency")
    args = ap.parse_args()

    ctx = init_distributed()
    modes = ["strong", "weak"] if args.mode == "both" else [args.mode]
    report = {
        "experiment": "EXP-6.2 DDP scaling", "generated_utc": time.strftime("%FT%TZ", time.gmtime()),
        "host": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "world_size": ctx.world_size,
        "node_count": ctx.node_count, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "workload_note": ("Synthetic/replicated shapes are used for the SYSTEMS measurement "
                          "only. They are never counted as scientific subjects."),
        "results": [],
    }
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None

    for name in args.workloads:
        for mode in modes:
            res = bench(ctx, WORKLOADS[name], mode, args.repeats, args.warmup, args.precision)
            res["workload"] = name
            res["synthetic"] = True
            if baseline:
                ref = next((r for r in baseline["results"]
                            if r["workload"] == name and r["mode"] == mode), None)
                if ref and ref["world_size"] == 1:
                    speedup = ref["iteration_ms"]["median"] / res["iteration_ms"]["median"]
                    res["baseline_median_ms"] = ref["iteration_ms"]["median"]
                    res["speedup_vs_1gpu"] = speedup
                    res["parallel_efficiency"] = speedup / ctx.world_size
            else:
                res["speedup_vs_1gpu"] = None
                res["parallel_efficiency"] = None
                res["efficiency_note"] = ("no 1-GPU baseline supplied; efficiency is left "
                                          "null rather than assumed")
            report["results"].append(res)
            if ctx.is_main:
                eff = res.get("parallel_efficiency")
                print(f"{name:18s} {mode:6s} ws={res['world_size']:2d} "
                      f"median={res['iteration_ms']['median']:8.2f} ms  "
                      f"p95={res['iteration_ms']['p95']:8.2f}  "
                      f"thr={res['throughput_samples_per_s']:9.1f}/s  "
                      f"eta={'n/a' if eff is None else f'{eff:.3f}'}", flush=True)

    if ctx.is_main and args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n")
        print(f"written: {p}")
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
