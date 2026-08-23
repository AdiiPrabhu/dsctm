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
PY_VERSION="${DSCTM_PY_VERSION:-3.8}"   # PARAM ships anaconda3 with 3.8.5
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
  for m in ${DSCTM_MODULES:-anaconda3/anaconda3 cuda/11.8}; do
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

if [[ -n "${DSCTM_USE_SITE_TORCH:-}" ]]; then
  # Use the cluster's own PyTorch module instead of building an env. Fastest path on an
  # air-gapped node, but you inherit whatever Python and torch version the site ships.
  echo "using SITE torch module (DSCTM_USE_SITE_TORCH set); skipping conda env creation"
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
  export DSCTM_REPO_ROOT="$REPO_ROOT"
  python -c "import torch,sys;print('site python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda)" \
    || { echo "FATAL: site torch not importable. module load anaconda3/pytorch first."; return 1 2>/dev/null || exit 1; }
  SKIP_ENV=1
fi

if [[ -z "${SKIP_ENV:-}" ]] && ! command -v conda >/dev/null 2>&1; then
  echo "FATAL: conda not found after module load."
  echo "  Run 'module avail' and re-run with, e.g.:"
  echo "    DSCTM_MODULES='anaconda3/anaconda3' source scripts/param/env.sh"
  return 1 2>/dev/null || exit 1
fi
[[ -z "${SKIP_ENV:-}" ]] && eval "$(conda shell.bash hook)"

# --------------------------------------------------------------------------- #
# 2. Create or reuse the environment
# --------------------------------------------------------------------------- #
if [[ $REBUILD -eq 1 && -d "$ENV_PREFIX" ]]; then
  echo "removing existing env for rebuild: $ENV_PREFIX"
  conda env remove -p "$ENV_PREFIX" -y || rm -rf "$ENV_PREFIX"
fi

if [[ -n "${SKIP_ENV:-}" ]]; then
  :
elif [[ ! -d "$ENV_PREFIX" ]]; then
  echo "creating conda env at $ENV_PREFIX ..."
  conda create -p "$ENV_PREFIX" -y "python=$PY_VERSION"
  conda activate "$ENV_PREFIX"

  # PyTorch from the CUDA index. cu118 is chosen deliberately: it is the newest CUDA build
  # still published as manylinux_2_17 wheels, which is what CentOS 7.9's glibc 2.17 needs,
  # and `cuda/11.8` is available as a module on this cluster.
  #
  # PARAM also ships a `glibc/2.28` module. If you ever need a newer torch, that is the
  # route -- but loading an alternate glibc at runtime is fragile (it manipulates the
  # dynamic linker path and can break unrelated libraries), so it is NOT the default.
  # Verified working combination on PARAM Utkarsh:
  #     module load anaconda3/anaconda3 cuda/11.8   +   torch 2.1.2+cu118
  # Override with DSCTM_TORCH_SPEC / DSCTM_TORCH_INDEX after checking `nvidia-smi`.
  TORCH_SPEC="${DSCTM_TORCH_SPEC:-torch==2.1.2}"  # torchvision is unused by this project
  # download.pytorch.org does NOT resolve from PARAM, but pypi.org does.
  # PyPI's torch==2.1.2 is the cu121 build: manylinux (glibc 2.17 OK) and
  # CUDA 12.1 supports sm_70, so it runs on the V100s.
  TORCH_INDEX="${DSCTM_TORCH_INDEX:-https://pypi.org/simple}"

  if [[ -n "${DSCTM_OFFLINE_WHEELS:-}" ]]; then
    # ---- OFFLINE: install from a pre-staged wheel directory --------------- #
    # PARAM login and compute nodes have NO internet egress (DNS resolution
    # fails outright), so pip cannot reach PyPI or download.pytorch.org.
    # Build the bundle on a machine that does have egress:
    #     bash scripts/param/offline_bundle.sh      # on your laptop
    # then rsync it here and point DSCTM_OFFLINE_WHEELS at it.
    if [[ ! -d "$DSCTM_OFFLINE_WHEELS" ]]; then
      echo "FATAL: DSCTM_OFFLINE_WHEELS=$DSCTM_OFFLINE_WHEELS does not exist"
      return 1 2>/dev/null || exit 1
    fi
    echo "installing OFFLINE from $DSCTM_OFFLINE_WHEELS"
    pip install --no-cache-dir --no-index --find-links "$DSCTM_OFFLINE_WHEELS" \
      torch torchvision numpy scipy scikit-learn pandas pyyaml pytest \
      pyarrow thop opensmile soundfile matplotlib \
      || { echo "FATAL: offline install failed. Is the bundle complete for cp${PY_VERSION/./}?"; \
           return 1 2>/dev/null || exit 1; }
    pip install --no-cache-dir --no-index --no-build-isolation -e "$REPO_ROOT" \
      || pip install --no-cache-dir --no-deps -e "$REPO_ROOT"
  else
    # ---- ONLINE ------------------------------------------------------------ #
    echo "installing: $TORCH_SPEC  (index $TORCH_INDEX)"
    if ! pip install --no-cache-dir $TORCH_SPEC --index-url "$TORCH_INDEX"; then
      echo
      echo "FATAL: pip could not reach $TORCH_INDEX."
      echo "  PARAM nodes may have no internet egress. Check first:"
      echo "     env | grep -i proxy ; getent hosts pypi.org"
      echo "  If there is no egress, use the offline route:"
      echo "     1. on a machine WITH internet:  bash scripts/param/offline_bundle.sh"
      echo "     2. rsync dsctm_wheels.tar.gz to PARAM and untar it"
      echo "     3. DSCTM_OFFLINE_WHEELS=~/dsctm_wheels source scripts/param/env.sh"
      echo "  Or try the site PyTorch instead:"
      echo "     DSCTM_USE_SITE_TORCH=1 source scripts/param/env.sh"
      return 1 2>/dev/null || exit 1
    fi
    pip install --no-cache-dir --only-binary=:all: \
      "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
      "pyyaml>=6.0" "pytest>=7.4" "pyarrow>=14.0"
      for pkg in "thop>=0.1.1" "soundfile>=0.12" "opensmile>=2.5.0" "matplotlib>=3.7"; do
        pip install --no-cache-dir --only-binary=:all: "$pkg" >/dev/null 2>&1 \
          && echo "  ok: $pkg" || echo "  SKIPPED (no compatible wheel): $pkg"
      done
    pip install -e "$REPO_ROOT"
  fi
else
  conda activate "$ENV_PREFIX"
  echo "reusing existing env  ($(python -V 2>&1))"
  # An env directory can exist while the install that was meant to fill it FAILED --
  # exactly what happens when `conda create` succeeds and the following `pip install torch`
  # dies on a DNS error. Reusing it blindly leaves a Python with no torch and a very
  # confusing preflight. Verify, and finish the job if it is incomplete.
  if ! python -c "import torch" >/dev/null 2>&1; then
    echo "  env is INCOMPLETE (torch not importable) - completing the install"
    TORCH_SPEC="${DSCTM_TORCH_SPEC:-torch==2.1.2}"
    TORCH_INDEX="${DSCTM_TORCH_INDEX:-https://pypi.org/simple}"
    pip install --no-cache-dir $TORCH_SPEC --index-url "$TORCH_INDEX" \
      || { echo "FATAL: torch install failed. If pypi.org is unreachable, use:"; \
           echo "  bash scripts/param/bootstrap_offline.sh ~/dsctm_wheels"; \
           return 1 2>/dev/null || exit 1; }
    pip install --no-cache-dir --only-binary=:all: \
      "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
      "pyyaml>=6.0" "pytest>=7.4" "pyarrow>=14.0"
      for pkg in "thop>=0.1.1" "soundfile>=0.12" "opensmile>=2.5.0" "matplotlib>=3.7"; do
        pip install --no-cache-dir --only-binary=:all: "$pkg" >/dev/null 2>&1 \
          && echo "  ok: $pkg" || echo "  SKIPPED (no compatible wheel): $pkg"
      done
    pip install -e "$REPO_ROOT" || true
  fi
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
