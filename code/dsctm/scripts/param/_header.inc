# Shared preamble for every D-MSTCN sbatch script.
# Partitions on PARAM Utkarsh: standard | cpu | gpu | hm      (Access Guide p.5)
# Max walltime: 72:00:00                                       (Access Guide p.6)
# GPU nodes: 10 x (2 V100 SXM2 16 GB, 40 cores, 192 GB RAM)    (User Manual p.10)
set -euo pipefail
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -u +%FT%TZ)"
cd "${SLURM_SUBMIT_DIR}"
source scripts/param/env.sh
nvidia-smi || true
