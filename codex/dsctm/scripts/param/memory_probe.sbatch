#!/bin/bash
#SBATCH --job-name=dsctm-memprobe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# BLOCKER B-008. The engagement brief assumed 32 GB V100s; PARAM has 16 GB HBM2.
# Every batch size in every config is therefore an assumption until this job runs.
# Sweeps per-rank batch on ONE V100 until OOM and records the ceiling.
source scripts/param/_header.inc
mkdir -p logs
python scripts/param/memory_probe.py \
  --out "$DSCTM_RESULTS_ROOT/systems/memory_probe_${SLURM_JOB_ID}.json"
