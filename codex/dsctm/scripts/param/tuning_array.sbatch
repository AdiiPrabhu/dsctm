#!/bin/bash
#SBATCH --job-name=dsctm-tune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00
#SBATCH --array=0-47%4
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
#
# Equal-budget tuning: 6 models x 8 dev trials = 48 tasks, one V100 each.
# %4 throttles to 4 concurrent tasks = 4 of the cluster's 20 V100s. Raise only if the
# queue is empty; this is a shared facility.
source scripts/param/_header.inc
mkdir -p logs
srun python scripts/run_exp22_fair_tuning.py --trial-index "$SLURM_ARRAY_TASK_ID"
