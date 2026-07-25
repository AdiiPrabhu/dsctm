#!/bin/bash
#SBATCH --job-name=dsctm-2n4g
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# 4 ranks over 2 nodes -> 20% of the cluster's 20 V100s. This is also the minimum
# topology that can host the manuscript's SAP layout (3 branches + aggregator), because a
# PARAM node has only 2 GPUs. See DECISIONS.md D-007.
source scripts/param/_header.inc
mkdir -p logs
DSCTM_GPUS_PER_NODE=2 bash scripts/param/launch_torchrun.sh \
  "${DSCTM_SCRIPT:?set DSCTM_SCRIPT}" ${DSCTM_ARGS:-}
