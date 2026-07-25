#!/bin/bash
# PARAM Utkarsh environment bootstrap.
#
#   source scripts/param/env.sh          # activate (creates on first run)
#   scripts/param/env.sh --rebuild       # force a clean rebuild
#
# Facts this encodes (PARAM_Utkarsh_User_Manual-v3.0-1.pdf, PARAM Utkarsh Access Guide.pdf):
#   OS            CentOS 7.9  -> glibc 2.17
#   Scheduler     SLURM 20.11.8
#   Partitions    standard | cpu | gpu | hm
#   GPU nodes     10 nodes x 2 NVIDIA V100 SXM2, 16 GB HBM2 each, sm_70
#   GPU node CPU  2 x Intel Xeon G-6248, 40 cores, 192 GB RAM, 480 GB local SSD
#   Interconnect  Mellanox InfiniBand HDR 100 Gbps
#   Storage       Lustre, 1.3 PB
#   Max walltime  72:00:00
#
# glibc 2.17 is the reason this script does not simply `pip install torch`. Recent PyTorch
# wheels target manylinux_2_28 (glibc >= 2.28) and will either refuse to install or install
# and then fail at import inside a queued job — which costs a queue slot to discover.
# Strategy: prefer a site conda env, verify hard, and report exactly what was resolved.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${DSCTM_ENV_NAME:-dsctm}"
ENV_PREFIX="${DSCTM_ENV_PREFIX:-$HOME/.conda/envs/$ENV_NAME}"
PY_VERSION="${DSCTM_PY_VERSION:-3.10}"
REBUILD=0
[[ "${1:-}" == "--rebuild" ]] && REBUILD=1

echo "=== D-MSTCN PARAM environment ==="
echo "repo:   $REPO_ROOT"
echo "env:    $ENV_PREFIX  (python $PY_VERSION)"
echo "host:   $(hostname)"

# --------------------------------------------------------------------------- #
# 1. Site modules. Names vary between PARAM images, so try a list and report.
# --------------------------------------------------------------------------- #
if command -v module >/dev/null 2>&1; then
  set +u
  module purge 2>/dev/null || true
  for m in ${DSCTM_MODULES:-anaconda3/anaconda3 cuda/11.8 cuda/12.0}; do
    if module load "$m" 2>/dev/null; then
      echo "module loaded: $m"
    else
      echo "module NOT available (continuing): $m"
    fi
  done
  set -u
else
  echo "WARNING: no 'module' command; assuming conda is already on PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "FATAL: conda not found after module load."
  echo "  Run 'module avail' and re-run with, e.g.:"
  echo "    DSCTM_MODULES='anaconda3/anaconda3' source scripts/param/env.sh"
  return 1 2>/dev/null || exit 1
fi
eval "$(conda shell.bash hook)"

# --------------------------------------------------------------------------- #
# 2. Create or reuse the environment
# --------------------------------------------------------------------------- #
if [[ $REBUILD -eq 1 && -d "$ENV_PREFIX" ]]; then
  echo "removing existing env for rebuild: $ENV_PREFIX"
  conda env remove -p "$ENV_PREFIX" -y || rm -rf "$ENV_PREFIX"
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "creating conda env at $ENV_PREFIX ..."
  conda create -p "$ENV_PREFIX" -y "python=$PY_VERSION"
  conda activate "$ENV_PREFIX"

  # PyTorch first, from the CUDA index. cu118 is chosen deliberately: it is the newest
  # CUDA build still published as manylinux_2_17 wheels, which is what glibc 2.17 needs.
  # Override with DSCTM_TORCH_SPEC / DSCTM_TORCH_INDEX after checking `nvidia-smi`.
  TORCH_SPEC="${DSCTM_TORCH_SPEC:-torch==2.1.2 torchvision==0.16.2}"
  TORCH_INDEX="${DSCTM_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
  echo "installing: $TORCH_SPEC  (index $TORCH_INDEX)"
  pip install --no-cache-dir $TORCH_SPEC --index-url "$TORCH_INDEX"

  pip install --no-cache-dir \
    "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
    "pyyaml>=6.0" "pytest>=7.4" "pyarrow>=14.0" "thop>=0.1.1" \
    "opensmile>=2.5.0" "soundfile>=0.12" "matplotlib>=3.7"

  pip install -e "$REPO_ROOT"
else
  conda activate "$ENV_PREFIX"
  echo "reusing existing env"
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export DSCTM_REPO_ROOT="$REPO_ROOT"

# --------------------------------------------------------------------------- #
# 3. Data roots. Home is small and Lustre scratch is where 90 GB belongs.
# --------------------------------------------------------------------------- #
export DSCTM_SCRATCH="${DSCTM_SCRATCH:-$HOME/scratch/dsctm}"
export DSCTM_DATA_ROOT="${DSCTM_DATA_ROOT:-$DSCTM_SCRATCH/datasets}"
export DSCTM_DAICWOZ_ROOT="${DSCTM_DAICWOZ_ROOT:-$DSCTM_DATA_ROOT/DAIC-WOZ}"
export DSCTM_EDAIC_ROOT="${DSCTM_EDAIC_ROOT:-$DSCTM_DATA_ROOT/E-DAIC}"
export DSCTM_STUDENTLIFE_ROOT="${DSCTM_STUDENTLIFE_ROOT:-$DSCTM_DATA_ROOT/StudentLife/dataset}"
export DSCTM_RESULTS_ROOT="${DSCTM_RESULTS_ROOT:-$REPO_ROOT/../../results/param_utkarsh_authoritative}"
mkdir -p "$DSCTM_SCRATCH" "$DSCTM_DATA_ROOT" "$DSCTM_RESULTS_ROOT"

# --------------------------------------------------------------------------- #
# 4. NCCL over Mellanox InfiniBand HDR
# --------------------------------------------------------------------------- #
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"          # INFO only when diagnosing; it is noisy
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"   # 1 forces TCP fallback (slow; last resort)
export NCCL_ASYNC_ERROR_HANDLING=1               # a dead peer aborts instead of hanging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1         # newer torch spelling
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"   # 40 cores / 2 ranks, leave room for I/O

echo
echo "python:  $(which python)  $(python -V 2>&1)"
echo "scratch: $DSCTM_SCRATCH"
echo "results: $DSCTM_RESULTS_ROOT"
echo
echo "Next:  python scripts/param/preflight.py          # login node, CPU-only checks"
echo "       sbatch  scripts/param/2gpu_ddp_smoke.sbatch # first real GPU validation"
