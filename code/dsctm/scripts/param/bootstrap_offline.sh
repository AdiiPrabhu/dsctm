#!/bin/bash
# Offline environment bootstrap for an air-gapped PARAM node.
#
#   bash scripts/param/bootstrap_offline.sh ~/dsctm_wheels
#
# Why this exists separately from env.sh: `conda create -p ... python=3.10` DOWNLOADS the
# interpreter from the conda repos, which needs internet PARAM does not have. So the
# environment must be built from a Python that already exists on the machine, using
# `python -m venv`, which is entirely offline.
#
# This script:
#   1. surveys every Python it can find on the cluster (module-provided and system)
#   2. picks the newest >= 3.9
#   3. checks the wheel bundle's cp tag MATCHES that interpreter, and stops with the exact
#      rebuild command if it does not — a cp310 wheel silently will not install on 3.9
#   4. creates a venv and installs from the bundle with --no-index
#   5. verifies torch actually imports
set -uo pipefail

BUNDLE="${1:-$HOME/dsctm_wheels}"
VENV="${DSCTM_VENV:-$HOME/dsctm-venv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== offline bootstrap ==="
echo "bundle: $BUNDLE"
echo "venv  : $VENV"
echo "repo  : $REPO_ROOT"
echo

# --------------------------------------------------------------------------- #
# 1. What Python interpreters exist here?
# --------------------------------------------------------------------------- #
echo "--- available interpreters ---"
declare -a FOUND=()
probe() {
  local py="$1" label="$2"
  [[ -x "$py" ]] || command -v "$py" >/dev/null 2>&1 || return 0
  local v
  v="$("$py" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" || return 0
  [[ -z "$v" ]] && return 0
  printf "  %-42s %s   %s\n" "$py" "$v" "$label"
  FOUND+=("$v|$py")
}

if command -v module >/dev/null 2>&1; then
  set +u
  for m in anaconda3/anaconda3 anaconda3/pytorch anaconda3/tensorflow \
           python/conda-python/3.7 IntelPy/3.7 rapid/conda-rapid/21.08; do
    module purge >/dev/null 2>&1
    if module load "$m" >/dev/null 2>&1; then
      probe "$(command -v python3 || command -v python)" "(module $m)"
    fi
  done
  module purge >/dev/null 2>&1
  set -u
fi
for p in /usr/bin/python3 /usr/local/bin/python3 python3.11 python3.10 python3.9 python3; do
  probe "$p" "(system)"
done

if [[ ${#FOUND[@]} -eq 0 ]]; then
  echo "FATAL: no usable Python found."
  exit 1
fi

# newest version wins
BEST="$(printf '%s\n' "${FOUND[@]}" | sort -t'|' -k1,1 -V | tail -n1)"
PYVER="${BEST%%|*}"
PYBIN="${BEST##*|}"
PYTAG="cp${PYVER/./}"
echo
echo "selected: $PYBIN  (Python $PYVER, wheel tag $PYTAG)"

if [[ "$(printf '%s\n3.9\n' "$PYVER" | sort -V | head -n1)" != "3.9" ]]; then
  echo
  echo "FATAL: Python $PYVER is too old. This project needs >= 3.9."
  echo "  Nothing newer is available on this cluster without internet."
  echo "  Options: ask CDAC for a newer python module, or use singularity/3.4.1"
  echo "  with a container built off-cluster."
  exit 1
fi

# --------------------------------------------------------------------------- #
# 2. Does the bundle match this interpreter?
# --------------------------------------------------------------------------- #
if [[ ! -d "$BUNDLE" ]]; then
  echo "FATAL: bundle directory not found: $BUNDLE"
  echo "  Did you untar it?   tar xzf ~/dsctm_wheels.tar.gz -C ~"
  exit 1
fi

TORCH_WHL="$(ls "$BUNDLE"/torch-*.whl 2>/dev/null | head -n1)"
if [[ -z "$TORCH_WHL" ]]; then
  echo "FATAL: no torch wheel in $BUNDLE"
  exit 1
fi
echo "bundle torch: $(basename "$TORCH_WHL")"

if [[ "$(basename "$TORCH_WHL")" != *"$PYTAG"* ]]; then
  BUNDLE_TAG="$(basename "$TORCH_WHL" | grep -o 'cp3[0-9]*' | head -n1)"
  cat <<EOF

FATAL: wheel/interpreter mismatch.
  bundle was built for : $BUNDLE_TAG
  this cluster provides: $PYTAG  (Python $PYVER)

  A $BUNDLE_TAG wheel will not install on Python $PYVER.

  REBUILD ON YOUR LAPTOP with the matching tag:

    cd <repo>/code/dsctm
    rm -rf dsctm_wheels dsctm_wheels.tar.gz
    DSCTM_PY_VERSION_TAG=${PYVER/./} bash scripts/param/offline_bundle.sh
    rsync -avP -e 'ssh -p 4422' dsctm_wheels.tar.gz \\
      \$USER@paramutkarsh.cdac.in:~/

  then re-run this script.
EOF
  exit 2
fi
echo "wheel tag matches interpreter."

# --------------------------------------------------------------------------- #
# 3. Build the venv (fully offline)
# --------------------------------------------------------------------------- #
echo
echo "--- creating venv ---"
rm -rf "$VENV"
"$PYBIN" -m venv "$VENV" || { echo "FATAL: venv creation failed"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

PIP_WHL="$(ls "$BUNDLE"/pip-*.whl 2>/dev/null | head -n1)"
if [[ -n "$PIP_WHL" ]]; then
  python -m pip install -q --no-index "$PIP_WHL" || true
fi
echo "pip: $(python -m pip --version)"

echo
echo "--- installing from bundle (no network) ---"
python -m pip install --no-cache-dir --no-index --find-links "$BUNDLE" \
  torch torchvision numpy scipy scikit-learn pandas pyyaml pytest \
  pyarrow matplotlib thop soundfile opensmile \
  || {
    echo
    echo "Some packages failed. Retrying the essential set only..."
    python -m pip install --no-cache-dir --no-index --find-links "$BUNDLE" \
      torch numpy scipy scikit-learn pandas pyyaml pytest pyarrow \
      || { echo "FATAL: essential install failed"; exit 1; }
  }

python -m pip install --no-cache-dir --no-index --no-build-isolation -e "$REPO_ROOT" \
  2>/dev/null || python -m pip install --no-cache-dir --no-deps -e "$REPO_ROOT" 2>/dev/null \
  || echo "NOTE: editable install skipped; PYTHONPATH will be used instead"

# --------------------------------------------------------------------------- #
# 4. Verify
# --------------------------------------------------------------------------- #
echo
echo "--- verification ---"
python - <<'PYEOF'
import sys
print("python :", sys.version.split()[0])
try:
    import torch
    print("torch  :", torch.__version__, "cuda-build", torch.version.cuda)
    print("cuda   :", torch.cuda.is_available(), "(False on a login node is EXPECTED)")
    import torch.distributed as d
    print("nccl   :", d.is_nccl_available())
except Exception as e:
    print("TORCH IMPORT FAILED:", type(e).__name__, e); sys.exit(1)
for m in ("numpy", "scipy", "sklearn", "pandas", "yaml", "pyarrow"):
    try:
        mod = __import__(m)
        print(f"{m:7s}: {getattr(mod,'__version__','present')}")
    except Exception as e:
        print(f"{m:7s}: MISSING ({type(e).__name__})")
PYEOF

cat <<EOF

=== done ===
Activate it in every future shell with:

  source $VENV/bin/activate
  export PYTHONPATH=$REPO_ROOT/src:$REPO_ROOT
  export DSCTM_ACCOUNT=nsmexternal
  export DSCTM_SCRATCH=/scratch/\$USER/dsctm
  export DSCTM_DATA_ROOT=\$DSCTM_SCRATCH/datasets
  export DSCTM_RESULTS_ROOT=$REPO_ROOT/../../results/param_utkarsh_authoritative

Add those to ~/.bashrc, then:

  python scripts/param/preflight.py
EOF
