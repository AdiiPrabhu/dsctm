#!/bin/bash
# Does this node reach the internet? Run it BEFORE planning where staging happens.
#
#   bash scripts/param/check_egress.sh              # safe on a login node: 3 tiny HEADs
#   bash scripts/param/submit.sh stage_datasets.sbatch   # runs the same check on a compute node
#
# Three single HEAD requests, a few hundred bytes total. This is not "running a job";
# it is the check that tells you whether you are ALLOWED to avoid running one.
set -uo pipefail
echo "host: $(hostname)   slurm_job: ${SLURM_JOB_ID:-none}"
ok=0
for url in https://dcapswoz.ict.usc.edu/wwwdaicwoz/ \
           https://dcapswoz.ict.usc.edu/wwwedaic/ \
           https://www.kaggle.com/; do
  if curl -sS -I --max-time 15 "$url" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "^(2|3)"; then
    echo "  REACHABLE  $url"; ok=$((ok+1))
  else
    echo "  BLOCKED    $url"
  fi
done
echo
if [[ $ok -eq 3 ]]; then
  echo "VERDICT: this node has egress. Staging can run here."
else
  echo "VERDICT: egress is restricted ($ok/3 reachable)."
  echo "  If a COMPUTE node is blocked but the LOGIN node is not, you must stage from the"
  echo "  login node - carefully, in small resumable chunks - or ask CDAC for a data-transfer"
  echo "  route. Do NOT run a multi-hour download on the login node: the MOTD states users"
  echo "  are disabled automatically for running jobs there."
fi
