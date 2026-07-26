#!/bin/bash
# Build a wheel bundle on a machine WITH internet, for the air-gapped PARAM nodes.
#
# RUN THIS ON YOUR LAPTOP, not on PARAM. PARAM cannot resolve external hostnames at all
# (BLOCKERS.md B-026), so every wheel has to arrive by rsync.
#
#   bash scripts/param/offline_bundle.sh
#   rsync -avP -e 'ssh -p 4422' dsctm_wheels.tar.gz \
#         basavarajh@paramutkarsh.cdac.in:~/
#
# On PARAM:
#   tar xzf ~/dsctm_wheels.tar.gz -C ~
#   DSCTM_OFFLINE_WHEELS=~/dsctm_wheels source scripts/param/env.sh
#
# TWO NON-OBVIOUS DETAILS, both found the hard way:
#
#  1. macOS system pip (21.x) cannot do this. Its `--platform` handling against the PyTorch
#     index is broken and it reports "No matching distribution found" with no hint why.
#     This script bootstraps a modern pip into a throwaway venv.
#
#  2. PyTorch CUDA wheels are tagged `linux_x86_64`, NOT `manylinux2014_x86_64`. Passing the
#     manylinux tag silently matches nothing. Everything else on PyPI does use manylinux,
#     so the two groups need different --platform values.
set -euo pipefail

OUT="${1:-dsctm_wheels}"
PYVER="${DSCTM_PY_VERSION_TAG:-310}"
TORCH_VER="${DSCTM_TORCH_VER:-2.1.2}"
TV_VER="${DSCTM_TV_VER:-0.16.2}"
CUDA_TAG="${DSCTM_CUDA_TAG:-cu118}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== offline bundle: $OUT  (linux_x86_64, cp${PYVER}, ${CUDA_TAG}) ==="

# --- 1. modern pip in a throwaway venv ------------------------------------- #
TMPROOT="$(mktemp -d)"
trap 'rm -rf "$TMPROOT"' EXIT
python3 -m venv "$TMPROOT/venv"
"$TMPROOT/venv/bin/pip" -q install --upgrade pip
PIP="$TMPROOT/venv/bin/pip"
echo "using $($PIP --version)"

mkdir -p "$OUT"

# --- 2. torch + torchvision from the CUDA index (linux_x86_64 tag) --------- #
echo
echo "--- torch ${TORCH_VER}+${CUDA_TAG} / torchvision ${TV_VER}+${CUDA_TAG}  (~2.5 GB) ---"
"$PIP" download \
  "torch==${TORCH_VER}+${CUDA_TAG}" "torchvision==${TV_VER}+${CUDA_TAG}" \
  --index-url "$TORCH_INDEX" \
  --dest "$OUT" \
  --platform linux_x86_64 \
  --python-version "$PYVER" \
  --only-binary=:all: \
  --no-deps

# --- 3. torch's own runtime deps, from PyPI (manylinux tag) ----------------- #
echo
echo "--- torch runtime dependencies ---"
"$PIP" download \
  filelock typing_extensions sympy networkx jinja2 fsspec mpmath MarkupSafe \
  --dest "$OUT" --platform manylinux2014_x86_64 --python-version "$PYVER" \
  --only-binary=:all: --no-deps

# --- 4. project dependencies ------------------------------------------------ #
echo
echo "--- project dependencies ---"
"$PIP" download \
  "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
  "pyyaml>=6.0" "pytest>=7.4" "pyarrow>=14.0" "matplotlib>=3.7" "soundfile>=0.12" \
  joblib threadpoolctl python-dateutil pytz tzdata six cffi pycparser \
  packaging pluggy iniconfig exceptiongroup tomli \
  contourpy cycler fonttools kiwisolver pillow pyparsing \
  --dest "$OUT" --platform manylinux2014_x86_64 --python-version "$PYVER" \
  --only-binary=:all: --no-deps

# --- 5. pure-python packages with no platform wheel ------------------------- #
# thop and the opensmile/audeer stack ship sdists or py3-none-any wheels; --platform
# rejects sdists, so these are fetched without it. They are pure python and need no
# compiler on PARAM.
echo
echo "--- pure-python packages (no platform constraint) ---"
"$PIP" download \
  thop opensmile audeer audobject audformat audinterface audiofile oyaml \
  --dest "$OUT" --no-deps || echo "  (some optional packages unavailable; continuing)"

# --- 6. archive ------------------------------------------------------------- #
COUNT=$(find "$OUT" -type f | wc -l | tr -d ' ')
SIZE=$(du -sh "$OUT" | cut -f1)
echo
echo "bundled $COUNT files, $SIZE"
tar czf "${OUT}.tar.gz" "$OUT"
ARCHIVE="${OUT}.tar.gz"
echo "archive: $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
shasum -a 256 "$ARCHIVE" | tee "${ARCHIVE}.sha256"

cat <<EOF

Next, from this laptop:
  rsync -avP -e 'ssh -p 4422' $ARCHIVE ${ARCHIVE}.sha256 \\
    \${PARAM_USER:-basavarajh}@paramutkarsh.cdac.in:~/

Then on PARAM:
  cd ~ && shasum -a 256 -c $(basename "${ARCHIVE}").sha256
  tar xzf ~/$(basename "$ARCHIVE") -C ~
  cd ~/dsctm/codex/dsctm
  DSCTM_OFFLINE_WHEELS=~/$(basename "$OUT") source scripts/param/env.sh
  python scripts/param/preflight.py
EOF
