#!/bin/bash
#SBATCH --job-name=dsctm-ddp-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# FIRST JOB TO RUN ON PARAM. Validates the environment, both V100s, NCCL, fp16 and the
# distributed layer before any science job consumes a queue slot.
source scripts/param/_header.inc
mkdir -p logs "$DSCTM_RESULTS_ROOT/preflight"

echo "--- preflight: GPU ---"
python scripts/param/preflight.py --gpu \
  --json "$DSCTM_RESULTS_ROOT/preflight/preflight_gpu_${SLURM_JOB_ID}.json"

echo "--- preflight: live NCCL all-reduce across 2 ranks ---"
DSCTM_GPUS_PER_NODE=2 bash scripts/param/launch_torchrun.sh \
  scripts/param/preflight.py --gpu --nccl \
  --json "$DSCTM_RESULTS_ROOT/preflight/preflight_nccl_${SLURM_JOB_ID}.json"

echo "--- distributed test suite on real GPUs ---"
python -m pytest -q tests/test_distributed_units.py tests/test_distributed_gloo.py \
  --junitxml="$DSCTM_RESULTS_ROOT/preflight/gpu_tests_${SLURM_JOB_ID}.xml"

echo "done=$(date -u +%FT%TZ)"
