#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-1gpu-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# SINGLE-GPU smoke test. Use when 2 GPUs cannot be scheduled.
#
# Observed on PARAM 2026-07-26: gpu[001-009] are unavailable (drained/reserved), leaving
# effectively one usable GPU node. --gres=gpu:2 therefore cannot be satisfied at all, and
# --gres=gpu:1 is the only way to touch a V100.
#
# This validates everything that does not require a second device:
#   CUDA present, sm_70, fp16 autocast + GradScaler, the model forward/backward on GPU,
#   the full CPU/gloo distributed suite, and a 2-rank gloo run ON ONE GPU.
#
# It does NOT validate NCCL across devices. That still needs 2 GPUs and remains the
# outstanding half of Gate 3.
source scripts/param/_header.inc
mkdir -p logs "$DSCTM_RESULTS_ROOT/preflight"

echo "=== preflight (GPU) ==="
python scripts/param/preflight.py --gpu \
  --json "$DSCTM_RESULTS_ROOT/preflight/preflight_1gpu_${SLURM_JOB_ID}.json"

echo
echo "=== single-GPU model exercise: fp32 and fp16 ==="
python - <<'PYEOF'
import torch, torch.nn as nn, time, json, os, sys
sys.path.insert(0, "src")
from dsctm.models import DMSTCN, DMSTCNConfig
from dsctm.distributed import autocast_dtype, build_grad_scaler

dev = torch.device("cuda", 0)
p = torch.cuda.get_device_properties(0); cap = torch.cuda.get_device_capability(0)
print(f"device: {p.name} sm_{cap[0]}{cap[1]} {p.total_memory/2**30:.1f} GiB")

for precision in ("fp32", "fp16"):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    model = DMSTCN(DMSTCNConfig(input_dim=8, n_classes=3, n_subjects=48)).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    amp = autocast_dtype(precision, dev)
    scaler = build_grad_scaler(precision, dev)
    X = torch.randn(32, 60, 8, device=dev)
    y = torch.randint(0, 3, (32,), device=dev)
    s = torch.zeros(32, dtype=torch.long, device=dev)
    m = torch.ones(32, 60, dtype=torch.bool, device=dev)
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss = nn.functional.cross_entropy(model(X, s, mask=m), y)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        losses.append(float(loss))
    torch.cuda.synchronize()
    finite_g = all(torch.isfinite(q.grad).all().item()
                   for q in model.parameters() if q.grad is not None)
    print(f"  {precision}: losses {[round(l,4) for l in losses]}  "
          f"grads_finite={finite_g}  peak={torch.cuda.max_memory_allocated(dev)/2**30:.3f} GiB")
    assert all(l == l for l in losses), f"{precision}: NaN loss"
    assert finite_g, f"{precision}: non-finite gradients"
print("single-GPU fp32 + fp16 OK")
PYEOF

echo
echo "=== distributed suite (CPU/gloo; NCCL still needs 2 GPUs) ==="
python -m pytest -q tests/test_distributed_units.py tests/test_distributed_gloo.py \
  tests/test_sap_tcp.py tests/test_theorem_invariant.py \
  --junitxml="$DSCTM_RESULTS_ROOT/preflight/gpu_node_tests_${SLURM_JOB_ID}.xml"

echo
echo "NOTE: NCCL across two devices is NOT validated by this job."
echo "      Gate 3 stays open until 2gpu_ddp_smoke.sbatch runs."
echo "done=$(date -u +%FT%TZ)"
