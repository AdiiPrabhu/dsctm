#!/bin/bash
# Submit wrapper. PARAM requires "#SBATCH -A <account>" on every job (see login MOTD).
# The sbatch files carry a placeholder; this substitutes your real account at submit time
# so the account never has to be committed to the repository.
#
#   export DSCTM_ACCOUNT=<your project account>
#   bash scripts/param/submit.sh 2gpu_ddp_smoke.sbatch
#   bash scripts/param/submit.sh --debug 2gpu_ddp_smoke.sbatch   # 1 h debug partition
set -euo pipefail

# Resolve the account. sacctmgr's default column width is 10 characters and it TRUNCATES
# with a trailing '+', so "nsmextern+" is not a usable account name. -P (parsable) and an
# explicit width both avoid that; we use -P because it is exact.
if [[ -z "${DSCTM_ACCOUNT:-}" ]]; then
  DSCTM_ACCOUNT="$(sacctmgr -n -P show associations user="$USER" format=Account 2>/dev/null \
                   | grep -v '^$' | sort -u | head -n1)"
  [[ -n "$DSCTM_ACCOUNT" ]] && echo "auto-detected account: $DSCTM_ACCOUNT"
fi

if [[ -z "${DSCTM_ACCOUNT:-}" ]]; then
  echo "FATAL: could not determine your SLURM account."
  echo "  PARAM requires one on every job. Get the UNTRUNCATED name with either:"
  echo "     sacctmgr -P show associations user=\$USER format=Account,Partition,QOS"
  echo "     sacctmgr show associations user=\$USER format=Account%30,Partition,QOS"
  echo "  (plain 'format=Account' truncates at 10 chars and appends '+')"
  echo "  then:  export DSCTM_ACCOUNT=<account>"
  exit 2
fi

if [[ "$DSCTM_ACCOUNT" == *+ ]]; then
  echo "FATAL: DSCTM_ACCOUNT='$DSCTM_ACCOUNT' ends in '+' - that is sacctmgr truncation,"
  echo "  not part of the name. Get the full value with:"
  echo "     sacctmgr -P show associations user=\$USER format=Account"
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
