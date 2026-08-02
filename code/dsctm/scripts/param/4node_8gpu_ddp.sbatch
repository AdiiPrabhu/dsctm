#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-4n8g
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# 8 ranks = 40% of every GPU in the cluster. Expect a long queue wait. Submit this only
# after the 2-node configuration has produced a clean result. See BLOCKERS B-009.
source scripts/param/_header.inc
mkdir -p logs
DSCTM_GPUS_PER_NODE=2 bash scripts/param/launch_torchrun.sh \
  "${DSCTM_SCRIPT:?set DSCTM_SCRIPT}" ${DSCTM_ARGS:-}
