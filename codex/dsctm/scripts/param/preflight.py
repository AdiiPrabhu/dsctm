#!/usr/bin/env python
"""PARAM Utkarsh preflight. Fails loudly rather than degrading silently.

Run it twice:

    python scripts/param/preflight.py                # login node: CPU-only checks
    srun ... python scripts/param/preflight.py --gpu # inside an allocation: GPU + NCCL

Why this exists. The person who wrote the pipeline cannot see the cluster, and the person
running it should not have to read a traceback to learn that `thop` is missing or that the
GPU is not sm_70. Every check below either passes, or prints exactly what is wrong and what
to do about it, and the process exits non-zero.

Nothing here is allowed to "warn and continue" on a hard requirement. A run that starts on
a misconfigured node produces numbers that look fine and are not comparable to anything.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REQUIRED_CAPABILITY = (7, 0)   # V100
CHECKS: list[dict] = []


def record(name, ok, detail, hard=True, fix=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail,
                   "hard": bool(hard), "fix": fix})
    mark = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{mark}] {name}: {detail}")
    if not ok and fix:
        print(f"       fix: {fix}")
    return ok


def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def check_python():
    v = sys.version_info
    record("python.version", v >= (3, 9), f"{platform.python_version()}",
           fix="source scripts/param/env.sh  (creates a Python 3.10 conda env)")
    record("python.glibc", True, f"libc={platform.libc_ver()}", hard=False)


def check_packages():
    required = {"torch": True, "numpy": True, "scipy": True, "sklearn": True,
                "pandas": True, "yaml": True,
                # pyarrow is hard: the Gate 4 run contract requires predictions.parquet.
                # thop (FLOPs, E4-07) and opensmile (eGeMAPS) are soft: neither blocks the
                # GPU validation path, and on CentOS 7 some have no C89-safe build. They
                # are reported as warnings so the gap is visible without stopping work.
                "pyarrow": True,
                "thop": False, "opensmile": False, "soundfile": False}
    for name, hard in required.items():
        try:
            mod = __import__(name)
            record(f"package.{name}", True, getattr(mod, "__version__", "present"), hard=hard)
        except Exception as exc:
            record(f"package.{name}", False, f"MISSING ({type(exc).__name__})", hard=hard,
                   fix=f"pip install {name}")


def check_repo():
    try:
        import dsctm
        record("repo.import", True, f"dsctm from {Path(dsctm.__file__).parent}")
    except Exception as exc:
        record("repo.import", False, f"cannot import dsctm: {exc}",
               fix="source scripts/param/env.sh  (sets PYTHONPATH and pip install -e)")
        return
    try:
        from dsctm.distributed import UnpaddedDistributedSampler, audit_sampler_partition
        audit = audit_sampler_partition(47, 2)
        record("repo.distributed", audit["covers_exactly_once"],
               f"eval partition 47/ws2 covers exactly once "
               f"(padded sampler would emit {audit['padded_sampler_would_emit']})")
    except Exception as exc:
        record("repo.distributed", False, f"{type(exc).__name__}: {exc}")


def check_slurm():
    inside = bool(os.environ.get("SLURM_JOB_ID"))
    record("slurm.available", shutil.which("sinfo") is not None,
           "sinfo present" if shutil.which("sinfo") else "sinfo NOT on PATH", hard=False)
    record("slurm.inside_allocation", inside,
           f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}" if inside
           else "not inside an allocation (expected on a login node)", hard=False)
    if shutil.which("sinfo"):
        # Record what the cluster actually reports. CDAC's own documents disagree about
        # the GPU node count (Access Guide says 10, architecture diagram says 30), so we
        # trust sinfo and write it down rather than trusting either PDF.
        out = _sh("sinfo -h -o '%P %D %t' 2>/dev/null | head -40")
        record("slurm.partitions", bool(out), (out or "unreadable").replace("\n", " | "),
               hard=False)
        gpu = _sh("sinfo -h -p gpu -o '%D %t' 2>/dev/null")
        record("slurm.gpu_partition", bool(gpu), gpu or "gpu partition not visible",
               hard=False)


def check_paths():
    for var in ("DSCTM_SCRATCH", "DSCTM_DATA_ROOT", "DSCTM_RESULTS_ROOT"):
        value = os.environ.get(var)
        if not value:
            record(f"path.{var}", False, "unset",
                   fix="source scripts/param/env.sh")
            continue
        p = Path(value)
        ok = p.exists() and os.access(p, os.W_OK)
        usage = shutil.disk_usage(p) if p.exists() else None
        detail = str(p)
        if usage:
            detail += f"  free={usage.free / 2**30:.1f} GiB"
        record(f"path.{var}", ok, detail,
               fix=f"mkdir -p {value}" if not p.exists() else "check write permission")
    # DAIC-WOZ audio alone is ~86 GB; extraction needs comparable headroom again.
    scratch = os.environ.get("DSCTM_SCRATCH")
    if scratch and Path(scratch).exists():
        free_gib = shutil.disk_usage(scratch).free / 2**30
        record("path.free_space", free_gib >= 250,
               f"{free_gib:.1f} GiB free on scratch (need >= 250 GiB for DAIC-WOZ + E-DAIC "
               f"+ extracted features)", hard=False,
               fix="point DSCTM_SCRATCH at a Lustre scratch filesystem, not $HOME")


def check_datasets():
    from_env = {
        "DAIC-WOZ": os.environ.get("DSCTM_DAICWOZ_ROOT"),
        "E-DAIC": os.environ.get("DSCTM_EDAIC_ROOT"),
        "StudentLife": os.environ.get("DSCTM_STUDENTLIFE_ROOT"),
    }
    for name, root in from_env.items():
        if not root:
            record(f"dataset.{name}", False, "root unset", hard=False,
                   fix="source scripts/param/env.sh")
            continue
        p = Path(root)
        n = len(list(p.glob("*"))) if p.exists() else 0
        record(f"dataset.{name}", p.exists() and n > 0,
               f"{p} ({n} entries)" if p.exists() else f"{p} MISSING", hard=False,
               fix="bash scripts/param/stage_datasets.sh --all   (login node)")


def check_gpu(strict: bool):
    try:
        import torch
    except Exception:
        record("gpu.torch", False, "torch not importable")
        return
    available = torch.cuda.is_available()
    if not available:
        record("gpu.available", not strict,
               "no CUDA device (expected on a login node)" if not strict
               else "CUDA unavailable inside a GPU allocation",
               hard=strict,
               fix="submit with --partition=gpu --gres=gpu:N")
        return
    count = torch.cuda.device_count()
    record("gpu.count", count >= 1, f"{count} visible device(s)")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        mem = props.total_memory / 2**30
        record(f"gpu[{i}].device", True,
               f"{props.name} sm_{cap[0]}{cap[1]} {mem:.1f} GiB "
               f"({props.multi_processor_count} SMs)")
        record(f"gpu[{i}].capability", cap == REQUIRED_CAPABILITY,
               f"sm_{cap[0]}{cap[1]}" + ("" if cap == REQUIRED_CAPABILITY
                                         else " (expected sm_70 / V100)"),
               hard=False,
               fix="precision and scaling assumptions were written for V100 sm_70")
        # The brief assumed 32 GB; the manual says 16 GB HBM2. Record the truth.
        record(f"gpu[{i}].memory", True, f"{mem:.1f} GiB "
               f"({'16 GB class as documented' if mem < 20 else 'larger than the documented 16 GB'})",
               hard=False)
    import torch.distributed as dist
    record("gpu.nccl", dist.is_available() and dist.is_nccl_available(),
           "NCCL available" if (dist.is_available() and dist.is_nccl_available())
           else "NCCL MISSING — multi-GPU impossible",
           fix="install a CUDA build of PyTorch (not the CPU wheel)")
    record("gpu.torch_cuda", torch.version.cuda is not None,
           f"torch {torch.__version__} built for CUDA {torch.version.cuda}")
    # fp16 smoke: V100 supports it, bf16 does not.
    try:
        dev = torch.device("cuda", 0)
        x = torch.randn(64, 64, device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            y = (x @ x).sum()
        record("gpu.fp16_autocast", bool(torch.isfinite(y).item()),
               f"fp16 matmul finite (loss={float(y):.3e})")
    except Exception as exc:
        record("gpu.fp16_autocast", False, f"{type(exc).__name__}: {exc}")


def check_nccl_allreduce():
    """Only meaningful under torchrun with world_size > 1."""
    import torch
    import torch.distributed as dist
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        record("nccl.allreduce", True,
               "skipped (not launched with torchrun world_size>=2)", hard=False)
        return
    from dsctm.distributed import cleanup, init_distributed
    ctx = init_distributed(timeout_minutes=5)
    t = torch.ones(1024, device=ctx.device) * (ctx.rank + 1)
    dist.all_reduce(t)
    expected = sum(range(1, ctx.world_size + 1))
    ok = bool(torch.allclose(t, torch.full_like(t, float(expected))))
    record("nccl.allreduce", ok,
           f"world_size={ctx.world_size} backend={ctx.backend} "
           f"sum={float(t[0]):.0f} expected={expected}")
    record("nccl.topology", True,
           f"rank={ctx.rank} local_rank={ctx.local_rank} nodes={ctx.node_count} "
           f"nodelist={ctx.node_list}", hard=False)
    cleanup()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true",
                    help="require CUDA (run inside a GPU allocation)")
    ap.add_argument("--nccl", action="store_true",
                    help="also run a live all-reduce (launch under torchrun)")
    ap.add_argument("--json", type=str, default=None, help="write the report here")
    args = ap.parse_args()

    print("=" * 72)
    print(f"PARAM Utkarsh preflight — host={socket.gethostname()} "
          f"job={os.environ.get('SLURM_JOB_ID', 'none')}")
    print("=" * 72)

    check_python()
    check_packages()
    check_repo()
    check_slurm()
    check_paths()
    check_datasets()
    check_gpu(strict=args.gpu)
    if args.nccl:
        check_nccl_allreduce()

    hard_failures = [c for c in CHECKS if not c["ok"] and c["hard"]]
    soft = [c for c in CHECKS if not c["ok"] and not c["hard"]]

    print("=" * 72)
    print(f"{len(CHECKS)} checks · {len(hard_failures)} hard failure(s) · "
          f"{len(soft)} warning(s)")
    if hard_failures:
        print("\nBLOCKING:")
        for c in hard_failures:
            print(f"  - {c['check']}: {c['detail']}")
            if c["fix"]:
                print(f"      {c['fix']}")

    report = {
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checks": CHECKS,
        "hard_failures": len(hard_failures),
        "warnings": len(soft),
        "verdict": "PASS" if not hard_failures else "FAIL",
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nreport written: {args.json}")

    sys.exit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()
