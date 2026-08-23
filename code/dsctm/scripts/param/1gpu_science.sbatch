#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# Single-V100 scientific job. Most D-MSTCN experiments belong HERE, not on multi-GPU:
# StudentLife is 2,160 windows and DAIC-WOZ is 275 sessions. Spending 2 GPUs on a job
# that fits comfortably on 1 halves the number of jobs the queue will run for you.
# Multi-GPU is for the systems experiments (Gate 7+), not for the science.
source scripts/param/_header.inc
mkdir -p logs
srun python "${DSCTM_SCRIPT:?set DSCTM_SCRIPT=scripts/run_xxx.py}" ${DSCTM_ARGS:-}
