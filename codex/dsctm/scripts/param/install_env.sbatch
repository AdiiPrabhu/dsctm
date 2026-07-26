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
# torchvision is NOT imported anywhere in this project. It was only ever pulled in as a
# habit, and it drags in pillow, which has no wheel that builds under CentOS 7's gcc 4.8.
# Dropping it removes the entire failure mode.
pip install --no-cache-dir --only-binary=:all: \
  "${DSCTM_TORCH_SPEC:-torch==2.1.2}" \
  --index-url "${DSCTM_TORCH_INDEX:-https://pypi.org/simple}"

echo
echo "=== scientific stack ==="
# CORE: must succeed. --only-binary=:all: is essential on CentOS 7 -- its gcc is 4.8.5
# and defaults to C89, so any package that falls back to a source build fails with
# "'for' loop initial declarations are only allowed in C99 mode". Refusing sdists makes
# pip pick a wheel or fail loudly, instead of trying to compile and dying halfway.
pip install --no-cache-dir --only-binary=:all: \
  "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
  "pyyaml>=6.0" "pytest>=7.4" "pyarrow>=14.0"

# OPTIONAL: never blocks the environment.
#   thop      -> FLOPs accounting (tracker E4-07)
#   opensmile -> eGeMAPS extraction (needed before DAIC experiments, not before GPU tests)
#   matplotlib-> one plotting script; pulls pillow, which has no C89-safe sdist path
echo
echo "=== optional packages (failures are non-fatal) ==="
for pkg in "thop>=0.1.1" "soundfile>=0.12" "opensmile>=2.5.0" "matplotlib>=3.7"; do
  pip install --no-cache-dir --only-binary=:all: "$pkg" \
    && echo "  ok: $pkg" || echo "  SKIPPED (no compatible wheel): $pkg"
done

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
for m in ("numpy","scipy","sklearn","pandas","yaml","pyarrow"):
    mod = __import__(m); print(f"{m:8s}:", getattr(mod, "__version__", "present"))
PYEOF

echo
echo "=== preflight ==="
python scripts/param/preflight.py || true

echo "done=$(date -u +%FT%TZ)"
