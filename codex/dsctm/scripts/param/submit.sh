#!/bin/bash
# Submit wrapper. PARAM requires "#SBATCH -A <account>" on every job (see login MOTD).
# The sbatch files carry a placeholder; this substitutes your real account at submit time
# so the account never has to be committed to the repository.
#
#   export DSCTM_ACCOUNT=<your project account>
#   bash scripts/param/submit.sh 2gpu_ddp_smoke.sbatch
#   bash scripts/param/submit.sh --debug 2gpu_ddp_smoke.sbatch   # 1 h debug partition
set -euo pipefail

if [[ -z "${DSCTM_ACCOUNT:-}" ]]; then
  echo "FATAL: DSCTM_ACCOUNT is unset."
  echo "  PARAM requires an account on every job. Find yours with:"
  echo "     sacctmgr show associations user=\$USER format=Account,Partition,QOS"
  echo "  then:  export DSCTM_ACCOUNT=<account>"
  exit 2
fi

USE_DEBUG=0
[[ "${1:-}" == "--debug" ]] && { USE_DEBUG=1; shift; }
SCRIPT="${1:?usage: submit.sh [--debug] <script.sbatch> [sbatch args...]}"
shift || true
[[ -f "$SCRIPT" ]] || SCRIPT="$(dirname "$0")/$SCRIPT"

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
sed "s|DSCTM_ACCOUNT_PLACEHOLDER|${DSCTM_ACCOUNT}|" "$SCRIPT" > "$TMP"

if [[ $USE_DEBUG -eq 1 ]]; then
  # debug: cn001-005, gpu001, hm001 - max 01:00:00. Ideal for a first smoke test:
  # it is a separate pool, so it does not queue behind the production gpu partition.
  sed -i.bak -E 's|^#SBATCH --partition=.*|#SBATCH --partition=debug|; s|^#SBATCH --time=.*|#SBATCH --time=01:00:00|' "$TMP"
  rm -f "$TMP.bak"
  echo "submitting to DEBUG partition (1 h cap, gpu001)"
fi

echo "account   : $DSCTM_ACCOUNT"
echo "script    : $SCRIPT"
grep -E "^#SBATCH (--partition|--time|--gres|--nodes|--array)" "$TMP" | sed 's/^/  /'
sbatch "$@" "$TMP"
