#!/usr/bin/env python
"""Gate 10 — SAP/TCP systems sweep across the four execution modes.

    torchrun --nproc_per_node=2 --nnodes=2 scripts/param/sap_tcp_sweep.py --out ...

Compares:  full-model DDP (control) | synchronous SAP | async SAP without TCP |
           async SAP with TCP

Sweeps delta_max, T_sync, injected branch delay, straggler frequency, and network
impairment. EVERY impairment is applied deliberately, recorded exactly, and labelled a
controlled experiment. No network condition is fabricated: injected delays are recorded in
milliseconds as configured, and measured separately from them.

Requires >= 4 ranks for the SAP modes (3 branches + aggregator). On PARAM that is
--nodes=2 --gres=gpu:2, because a node has only 2 V100s.
"""
from __future__ import annotations

import argparse, itertools, json, os, socket, statistics, sys, time
from pathlib import Path

import torch, torch.distributed as dist, torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dsctm.distributed import (CommStats, ExecutionMode, MODE_DESCRIPTIONS,   # noqa: E402
                               SAPModel, TemporalConsistencyProtocol, cleanup,
                               init_distributed, plan_placement, seed_everything, wrap_ddp)
from dsctm.distributed.sap import sap_step                                     # noqa: E402
from dsctm.models import DMSTCN, DMSTCNConfig                                  # noqa: E402

DELTA_MAX_GRID = [5, 10, 20, 50, 100, 200]
T_SYNC_GRID = [10, 50, 100, 200]
BRANCH_DELAY_MS = [0.0, 1.0, 5.0, 20.0]      # injected, recorded, never fabricated
STRAGGLER_RATE = [0.0, 0.1, 0.25]


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _inject(delay_ms: float, rank: int, straggler_rate: float, step: int) -> float:
    """Apply a CONTROLLED delay. Returns the delay actually slept, in ms."""
    total = delay_ms
    if straggler_rate > 0 and ((step * 2654435761 + rank * 40503) % 1000) / 1000.0 < straggler_rate:
        total += delay_ms * 4 if delay_ms > 0 else 10.0
    if total > 0:
        time.sleep(total / 1000.0)
    return total


def run_mode(ctx, mode: ExecutionMode, cfg: dict, steps: int, warmup: int) -> dict:
    T, F, C, Bsz = cfg["T"], cfg["F"], cfg["C"], cfg["batch"]
    seed_everything(7, ctx)
    stats = CommStats()
    base = DMSTCN(DMSTCNConfig(input_dim=F, n_classes=C, D=cfg["D"],
                               head_hidden=cfg["D"], n_subjects=8))
    X = torch.randn(Bsz, T, F, device=ctx.device)
    y = torch.randint(0, C, (Bsz,), device=ctx.device)
    s = torch.zeros(Bsz, dtype=torch.long, device=ctx.device)
    mask = torch.ones(Bsz, T, dtype=torch.bool, device=ctx.device)
    lossf = nn.functional.cross_entropy

    tcp = None
    if mode == ExecutionMode.DDP_SYNC:
        model = wrap_ddp(base, ctx)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        def step_fn(i):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(X, s, mask=mask), y)
            loss.backward(); opt.step()
            return float(loss.detach())
    else:
        placement = plan_placement(("ssb", "msb", "lsb"), ctx.world_size)
        model = SAPModel(base, placement, ctx, stats)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        if mode == ExecutionMode.SAP_ASYNC_TCP:
            tcp = TemporalConsistencyProtocol(placement, ctx, cfg["delta_max"],
                                              cfg["t_sync"], stats=stats)
        def step_fn(i):
            _inject(cfg["branch_delay_ms"], ctx.rank, cfg["straggler_rate"], i)
            loss = sap_step(model, X, y, s, mask, lossf, opt)
            if tcp is not None:
                decision = tcp.decide()
                if decision["sync"]:
                    tcp.synchronize(model, opt)
            elif mode == ExecutionMode.SAP_SYNC and dist.is_initialized():
                dist.barrier()
            return loss

    for i in range(warmup):
        step_fn(i)
    _sync(ctx.device)
    if dist.is_initialized():
        dist.barrier()

    times, losses = [], []
    for i in range(steps):
        _sync(ctx.device); t0 = time.perf_counter_ns()
        loss = step_fn(warmup + i)
        _sync(ctx.device); times.append((time.perf_counter_ns() - t0) / 1e6)
        if loss is not None:
            losses.append(loss)

    local = torch.tensor(times, dtype=torch.float64, device=ctx.device)
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
    ms = local.tolist()
    out = {
        "mode": mode.value, "description": MODE_DESCRIPTIONS[mode],
        "world_size": ctx.world_size, "steps": steps, "warmup": warmup,
        "iteration_ms": {"median": statistics.median(ms), "mean": statistics.mean(ms),
                         "p95": sorted(ms)[max(0, int(.95 * len(ms)) - 1)],
                         "std": statistics.pstdev(ms) if len(ms) > 1 else 0.0},
        "throughput_samples_per_s": (cfg["batch"] / (statistics.median(ms) / 1000.0)),
        "final_loss": losses[-1] if losses else None,
        "communication": stats.to_dict(),
        "injected_branch_delay_ms": cfg["branch_delay_ms"],
        "injected_straggler_rate": cfg["straggler_rate"],
        "controlled_experiment": True,
    }
    if tcp is not None:
        out["tcp"] = {"delta_max": cfg["delta_max"], "t_sync": cfg["t_sync"],
                      "invariants": tcp.check_invariants(),
                      "state": tcp.state.to_dict()}
        out["tcp"]["state"]["sync_log"] = out["tcp"]["state"]["sync_log"][-20:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--F", type=int, default=32)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--full-grid", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctx = init_distributed()
    if ctx.world_size < 4:
        print(f"NOTE: world_size={ctx.world_size} < 4; SAP modes are skipped. "
              f"SAP needs 3 branches + aggregator = 4 ranks (PARAM: --nodes=2 --gres=gpu:2).",
              flush=True)

    grid = (list(itertools.product(DELTA_MAX_GRID, T_SYNC_GRID, BRANCH_DELAY_MS, STRAGGLER_RATE))
            if args.full_grid else
            [(10, 50, 0.0, 0.0), (10, 50, 5.0, 0.0), (10, 50, 5.0, 0.25),
             (5, 50, 5.0, 0.0), (50, 50, 5.0, 0.0), (200, 50, 5.0, 0.0),
             (10, 10, 5.0, 0.0), (10, 200, 5.0, 0.0)])

    report = {"experiment": "EXP-6.4/6.5 SAP+TCP sweep",
              "generated_utc": time.strftime("%FT%TZ", time.gmtime()),
              "host": socket.gethostname(), "world_size": ctx.world_size,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "impairment_note": ("All delays and straggler events are INJECTED by this "
                                  "script and recorded exactly as configured. Nothing is "
                                  "inferred, extrapolated or fabricated."),
              "grid_size": len(grid), "results": []}

    modes = [ExecutionMode.DDP_SYNC]
    if ctx.world_size >= 4:
        modes += [ExecutionMode.SAP_SYNC, ExecutionMode.SAP_ASYNC_NO_TCP,
                  ExecutionMode.SAP_ASYNC_TCP]

    for delta_max, t_sync, delay, strag in grid:
        cfg = dict(T=args.T, F=args.F, C=3, D=args.D, batch=args.batch,
                   delta_max=delta_max, t_sync=t_sync,
                   branch_delay_ms=delay, straggler_rate=strag)
        for mode in modes:
            if mode != ExecutionMode.SAP_ASYNC_TCP and (delta_max, t_sync) != grid[0][:2]:
                continue   # delta/T_sync only vary the TCP mode
            try:
                res = run_mode(ctx, mode, cfg, args.steps, args.warmup)
            except Exception as exc:
                res = {"mode": mode.value, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}", **cfg}
            res["config"] = cfg
            report["results"].append(res)
            if ctx.is_main:
                print(f"{mode.value:36s} dmax={delta_max:4d} tsync={t_sync:4d} "
                      f"delay={delay:5.1f}ms strag={strag:.2f}  "
                      f"median={res.get('iteration_ms', {}).get('median', float('nan')):8.2f} ms",
                      flush=True)

    if ctx.is_main and args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"written: {p}")
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
