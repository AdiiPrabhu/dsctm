#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-monitor
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# Long-lived monitoring poller. Runs on `standard` (1 core, negligible load) so it never
# consumes a GPU node and never trips the login-node CPU limit.
#
# View the dashboard:
#   scp -P 4422 <user>@paramutkarsh.cdac.in:<repo>/artifacts/monitoring/dashboard.html .
# then open it locally. It is self-contained: no CDN, no fonts, no JS libraries.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/param/env.sh
mkdir -p logs ../../artifacts/monitoring
python scripts/param/monitor.py --watch --interval "${DSCTM_MONITOR_INTERVAL:-120}" \
  --dashboard ../../artifacts/monitoring/dashboard.html
