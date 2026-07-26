#!/bin/bash
# Build a wheel bundle on a machine WITH internet, for an air-gapped PARAM node.
#
# RUN THIS ON YOUR LAPTOP, not on PARAM.
#
#   bash scripts/param/offline_bundle.sh                  # ~2.5 GB
#   rsync -avP -e 'ssh -p 4422' dsctm_wheels.tar.gz \
#         basavarajh@paramutkarsh.cdac.in:~/
#
# Then on PARAM:
#   tar xzf ~/dsctm_wheels.tar.gz -C ~
#   DSCTM_OFFLINE_WHEELS=~/dsctm_wheels source scripts/param/env.sh
#
# Cross-platform download: wheels are fetched for linux x86_64 / cp310 regardless of the
# host OS, so a Mac can build a bundle for CentOS.
set -euo pipefail

OUT="${1:-dsctm_wheels}"
PYVER="${DSCTM_PY_VERSION:-310}"
TORCH_SPEC="${DSCTM_TORCH_SPEC:-torch==2.1.2 torchvision==0.16.2}"
TORCH_INDEX="${DSCTM_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"

mkdir -p "$OUT"
echo "=== building offline bundle in $OUT (linux_x86_64, cp$PYVER) ==="

# torch/torchvision come from the CUDA index and are the bulk of the size.
python3 -m pip download $TORCH_SPEC \
  --index-url "$TORCH_INDEX" \
  --dest "$OUT" \
  --platform manylinux2014_x86_64 \
  --python-version "$PYVER" \
  --only-binary=:all: \
  --no-deps

# Their runtime dependencies, from PyPI.
python3 -m pip download \
  filelock typing_extensions sympy networkx jinja2 fsspec mpmath MarkupSafe \
  "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
  "pyyaml>=6.0" "pytest>=7.4" "pyarrow>=14.0" "thop>=0.1.1" \
  "opensmile>=2.5.0" "soundfile>=0.12" "matplotlib>=3.7" \
  joblib threadpoolctl python-dateutil pytz tzdata six cffi pycparser \
  packaging pluggy iniconfig exceptiongroup tomli \
  contourpy cycler fonttools kiwisolver pillow pyparsing \
  audeer audobject audformat audinterface audiofile oyaml \
  --dest "$OUT" \
  --platform manylinux2014_x86_64 \
  --python-version "$PYVER" \
  --only-binary=:all: || {
    echo "NOTE: some pure-python deps have no manylinux wheel; retrying those without --platform"
    python3 -m pip download audeer audobject audformat audinterface audiofile oyaml \
      --dest "$OUT" --no-deps || true
  }

COUNT=$(ls -1 "$OUT" | wc -l | tr -d ' ')
SIZE=$(du -sh "$OUT" | cut -f1)
echo
echo "bundled $COUNT files, $SIZE"
tar czf "${OUT}.tar.gz" "$OUT"
echo "archive: ${OUT}.tar.gz  ($(du -h "${OUT}.tar.gz" | cut -f1))"
echo
echo "Transfer:"
echo "  rsync -avP -e 'ssh -p 4422' ${OUT}.tar.gz \\"
echo "    \${USER}@paramutkarsh.cdac.in:~/"
echo
echo "On PARAM:"
echo "  tar xzf ~/${OUT}.tar.gz -C ~"
echo "  DSCTM_OFFLINE_WHEELS=~/${OUT} source scripts/param/env.sh"
