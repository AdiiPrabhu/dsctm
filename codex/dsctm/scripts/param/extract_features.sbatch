#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-extract
#SBATCH --partition=standard   # NOT 'cpu' - no such partition here
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# eGeMAPS extraction is CPU-bound and must NOT run on a login node (User Manual p.9:
# login-node CPU/memory limits kill offending processes, which would leave a half-built
# cache that looks complete). It also must not waste a GPU node. `cpu` partition, 48 cores.
set -euo pipefail
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -u +%FT%TZ)"
cd "${SLURM_SUBMIT_DIR}"
source scripts/param/env.sh
mkdir -p logs
python scripts/build_daicwoz_egemaps88.py
python scripts/build_daic_egemaps88.py
python scripts/param/verify_datasets.py --json "$DSCTM_DATA_ROOT/_manifests/dataset_hashes.json"
