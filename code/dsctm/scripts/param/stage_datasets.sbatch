#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-stage
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# Dataset staging AS A BATCH JOB, not on the login node.
#
# The login MOTD is explicit: "Please Don't run any jobs in Login Nodes. If you run, user
# will be disabled automatically." An 86 GB multi-hour wget is indistinguishable from a job.
# The cost of being wrong is a disabled account, so staging runs here instead.
#
# This job first checks egress. If the compute node cannot reach the internet, it says so
# and exits non-zero rather than silently producing an empty dataset tree.
set -euo pipefail
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -u +%FT%TZ)"
cd "${SLURM_SUBMIT_DIR}"
source scripts/param/env.sh
mkdir -p logs

echo "=== egress check on this compute node ==="
if ! bash scripts/param/check_egress.sh | tee /dev/stderr | grep -q "has egress"; then
  echo
  echo "ABORTING: this compute node cannot reach the dataset hosts."
  echo "Options, in order of preference:"
  echo "  1. Ask CDAC support for a data-transfer node or a proxy for these hosts."
  echo "  2. Stage from the login node in SHORT resumable chunks (wget -c), watching"
  echo "     your CPU time. Never leave a multi-hour download unattended there."
  echo "  3. Transfer from a machine you control:"
  echo "     rsync -avP --partial <local>/datasets/ \\"
  echo "       ${USER}@paramutkarsh.cdac.in:${DSCTM_DATA_ROOT}/   # via -e 'ssh -p 4422'"
  exit 3
fi

echo; echo "=== staging ==="
bash scripts/param/stage_datasets.sh --all
echo; echo "=== verify ==="
python scripts/param/verify_datasets.py --json "$DSCTM_DATA_ROOT/_manifests/dataset_hashes.json"
echo "done=$(date -u +%FT%TZ)"
