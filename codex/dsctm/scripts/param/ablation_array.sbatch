#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --array=0-77%4          # ablation: 78 tasks (was WRONG at 0-27 -> 50 silently dropped).
#                                #   python scripts/param/plan.py --sbatch-array ablation
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
#
# Gate 6 ablation families: branch combinations, dilation schedules, fusion variants
# (incl. linear_csag vs nonlinear_csag), personalization controls, preprocessing.
source scripts/param/_header.inc
mkdir -p logs
srun python scripts/param/run_task.py --family ablation --index "$SLURM_ARRAY_TASK_ID"
