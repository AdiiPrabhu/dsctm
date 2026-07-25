#!/bin/bash
#SBATCH --job-name=dsctm-seeds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --array=0-59%4          # confirm-daicwoz: 6 models x 10 seeds. VERIFY with:
#                                #   python scripts/param/plan.py --sbatch-array confirm-daicwoz
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
#
# Confirmation seeds: 6 models x 10 seeds, frozen configs, one V100 each.
source scripts/param/_header.inc
mkdir -p logs
srun python scripts/param/run_task.py --family confirm-daicwoz --index "$SLURM_ARRAY_TASK_ID"
