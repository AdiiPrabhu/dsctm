#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-scaling
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# Gate 7 strong/weak scaling. Committed matrix is 1, 2 and 4 GPUs; 8 is best-effort.
# 16 is NOT in the matrix: it is 80% of the cluster's 20 V100s (BLOCKERS B-009), and its
# absence is stated in the paper rather than papered over.
# Strong and weak scaling are measured separately and never mixed in one claim.
source scripts/param/_header.inc
mkdir -p logs
DSCTM_GPUS_PER_NODE=2 bash scripts/param/launch_torchrun.sh \
  scripts/param/scaling_benchmark.py --mode "${DSCTM_SCALING_MODE:-strong}" --repeats 5
