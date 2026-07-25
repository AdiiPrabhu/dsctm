#!/bin/bash
#SBATCH --job-name=dsctm-ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --array=0-27%4
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
#
# Gate 6 ablation families: branch combinations, dilation schedules, fusion variants
# (incl. linear_csag vs nonlinear_csag), personalization controls, preprocessing.
source scripts/param/_header.inc
mkdir -p logs
srun python scripts/run_phase5_ablation.py --variant-index "$SLURM_ARRAY_TASK_ID"
