#!/bin/bash
# torchrun launcher for PARAM Utkarsh. Sourced/called from inside an sbatch script.
#
#   bash scripts/param/launch_torchrun.sh <python_script> [args...]
#
# Derives the rendezvous endpoint from the SLURM allocation. Nothing is hardcoded:
# hostnames, node counts and GPU counts all come from SLURM_* at runtime, so the same
# launcher serves 1 GPU and 8 nodes.
set -euo pipefail

[[ -z "${SLURM_JOB_ID:-}" ]] && { echo "FATAL: not inside a SLURM allocation. Use sbatch."; exit 1; }

# Master = first node of the allocation. `scontrol show hostnames` expands the compact
# nodelist form (e.g. "gpu[01-04]") that SLURM_JOB_NODELIST uses.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# Port derived from the job id so two concurrent jobs on one node cannot collide.
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

NNODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE="${DSCTM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$GPUS_PER_NODE" -lt 1 ]] && GPUS_PER_NODE=1

export MASTER_ADDR MASTER_PORT
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

echo "=== torchrun launch ==="
echo "job          : $SLURM_JOB_ID"
echo "nodes        : $NNODES  ($SLURM_JOB_NODELIST)"
echo "gpus/node    : $GPUS_PER_NODE"
echo "world size   : $WORLD_SIZE"
echo "rendezvous   : $MASTER_ADDR:$MASTER_PORT"
echo "script       : $*"
echo "======================="

# --rdzv_backend=c10d is the supported path on torch>=1.9 and works across nodes over the
# InfiniBand-backed hostname. srun launches one torchrun per node; torchrun then forks one
# process per GPU on that node.
srun --ntasks="$NNODES" --ntasks-per-node=1 --export=ALL \
  torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$@"
