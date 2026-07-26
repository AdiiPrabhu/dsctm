#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-install
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# Build the Python environment AS A BATCH JOB.
#
# Installing torch is a ~2.5 GB download plus a large unpack. On a login node that is
# indistinguishable from running a job, and PARAM's banner says users are disabled
# automatically for that. It also gets killed by the login-node CPU/memory limit part way
# through, leaving exactly the half-built env that produced "package.torch: MISSING".
#
# Running it here fixes both: no login-node limit, and an SSH disconnect cannot kill it.
set -euo pipefail
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -u +%FT%TZ)"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge
module load anaconda3/anaconda3 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true
eval "$(conda shell.bash hook)"

ENV_PREFIX="${DSCTM_ENV_PREFIX:-$HOME/.conda/envs/dsctm}"
PY_VERSION="${DSCTM_PY_VERSION:-3.10}"
REPO_ROOT="$(pwd)"

if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "=== creating conda env (python $PY_VERSION) ==="
  conda create -p "$ENV_PREFIX" -y "python=$PY_VERSION"
fi
conda activate "$ENV_PREFIX"
echo "python: $(which python)  $(python -V 2>&1)"

echo
echo "=== torch (largest download; ~2.5 GB) ==="
# download.pytorch.org does not resolve from PARAM; pypi.org does. PyPI's torch==2.1.2 is
# the cu121 build: manylinux (glibc 2.17 OK) and CUDA 12.1 supports sm_70, so V100s work.
pip install --no-cache-dir \
  "${DSCTM_TORCH_SPEC:-torch==2.1.2}" "${DSCTM_TV_SPEC:-torchvision==0.16.2}" \
  --index-url "${DSCTM_TORCH_INDEX:-https://pypi.org/simple}"

echo
echo "=== scientific stack ==="
pip install --no-cache-dir \
  "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
  "pyyaml>=6.0" "pytest>=7.4" "pyarrow>=14.0" "thop>=0.1.1" \
  "opensmile>=2.5.0" "soundfile>=0.12" "matplotlib>=3.7"

echo
echo "=== project (editable) ==="
pip install --no-cache-dir -e "$REPO_ROOT" || echo "editable install skipped; PYTHONPATH covers it"

echo
echo "=== verification ==="
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
python - <<'PYEOF'
import sys
print("python:", sys.version.split()[0])
import torch, torch.distributed as d
print("torch :", torch.__version__, "cuda-build", torch.version.cuda)
print("nccl  :", d.is_nccl_available())
for m in ("numpy","scipy","sklearn","pandas","yaml","pyarrow","thop"):
    mod = __import__(m); print(f"{m:8s}:", getattr(mod, "__version__", "present"))
PYEOF

echo
echo "=== preflight ==="
python scripts/param/preflight.py || true

echo "done=$(date -u +%FT%TZ)"
