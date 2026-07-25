#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 GPU_COUNT [GPU_COUNT ...]" >&2
  exit 2
fi

config_path="${DMSTCN_MULTIGPU_CONFIG:-multi_gpu_validation/config.yaml}"

python multi_gpu_validation/preflight.py --output artifacts/multigpu/preflight.json

# Same-host one-GPU reference required to calculate scaling efficiency.
torchrun \
  --standalone \
  --nproc-per-node=1 \
  multi_gpu_validation/validate.py \
  scaling \
  --config "$config_path"

for gpu_count in "$@"; do
  if ! [[ "$gpu_count" =~ ^[2-9][0-9]*$ ]]; then
    echo "GPU_COUNT must be an integer of at least 2: $gpu_count" >&2
    exit 2
  fi
  for validation in correctness scaling checkpoint; do
    torchrun \
      --standalone \
      --nproc-per-node="$gpu_count" \
      multi_gpu_validation/validate.py \
      "$validation" \
      --config "$config_path"
  done
done

python multi_gpu_validation/summarize.py \
  --input-dir artifacts/multigpu \
  --output artifacts/multigpu/summary.json
