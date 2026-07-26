#!/bin/bash
# Submit wrapper. PARAM requires "#SBATCH -A <account>" on every job (see login MOTD).
# The sbatch files carry a placeholder; this substitutes your real account at submit time
# so the account never has to be committed to the repository.
#
#   export DSCTM_ACCOUNT=<your project account>
#   bash scripts/param/submit.sh 2gpu_ddp_smoke.sbatch
#   bash scripts/param/submit.sh -p standard 2gpu_ddp_smoke.sbatch   # gpu nodes via standard
#   bash scripts/param/submit.sh -p standard -t 00:20:00 memory_probe.sbatch
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
OVERRIDE_PARTITION=""
OVERRIDE_TIME=""
while true; do
  case "${1:-}" in
    --debug)      USE_DEBUG=1; shift ;;
    -p|--partition) OVERRIDE_PARTITION="${2:?--partition needs a value}"; shift 2 ;;
    -t|--time)      OVERRIDE_TIME="${2:?--time needs a value}"; shift 2 ;;
    *) break ;;
  esac
done
SCRIPT="${1:?usage: submit.sh [--debug] <script.sbatch> [sbatch args...]}"
shift || true
[[ -f "$SCRIPT" ]] || SCRIPT="$(dirname "$0")/$SCRIPT"

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
sed "s|DSCTM_ACCOUNT_PLACEHOLDER|${DSCTM_ACCOUNT}|" "$SCRIPT" > "$TMP"

# Which partitions can this account actually submit to? The login banner advertises
# `debug`, but sinfo may not expose it to every association -- and sbatch only tells you
# after you have already tried. Check first.
AVAILABLE="$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u | tr '\n' ' ')"
[[ -n "$AVAILABLE" ]] && echo "partitions available: $AVAILABLE"

if [[ $USE_DEBUG -eq 1 ]]; then
  if [[ " $AVAILABLE " == *" debug "* ]]; then
    sed -i.bak -E 's|^#SBATCH --partition=.*|#SBATCH --partition=debug|; s|^#SBATCH --time=.*|#SBATCH --time=01:00:00|' "$TMP"
    rm -f "$TMP.bak"
    echo "submitting to DEBUG partition (1 h cap)"
  else
    echo "NOTE: --debug requested but no 'debug' partition is available to this account."
    echo "      Falling back to the partition declared in the script."
  fi
fi

# Explicit override. On PARAM the `gpu` partition is heavily reserved (2-day queues are
# normal), while `standard` ALSO contains gpu[001-010] per the login banner and often
# schedules far sooner. `-p standard` with --gres=gpu:N is a legitimate route to the same
# hardware through a different queue.
if [[ -n "$OVERRIDE_PARTITION" ]]; then
  sed -i.bak -E "s|^#SBATCH --partition=.*|#SBATCH --partition=${OVERRIDE_PARTITION}|" "$TMP"
  rm -f "$TMP.bak"
  echo "partition overridden -> $OVERRIDE_PARTITION"
fi
if [[ -n "$OVERRIDE_TIME" ]]; then
  sed -i.bak -E "s|^#SBATCH --time=.*|#SBATCH --time=${OVERRIDE_TIME}|" "$TMP"
  rm -f "$TMP.bak"
  echo "walltime overridden -> $OVERRIDE_TIME"
fi

# Validate the partition the script actually asks for, so a typo or a stale assumption
# fails here with a useful message rather than inside sbatch.
WANT="$(grep -m1 -E '^#SBATCH --partition=' "$TMP" | sed -E 's/.*--partition=([^ ]*).*/\1/')"
if [[ -n "$AVAILABLE" && -n "$WANT" && " $AVAILABLE " != *" $WANT "* ]]; then
  echo "FATAL: partition '$WANT' is not available to you."
  echo "       available: $AVAILABLE"
  exit 2
fi

echo "account   : $DSCTM_ACCOUNT"
echo "script    : $SCRIPT"
grep -E "^#SBATCH (--partition|--time|--gres|--nodes|--array)" "$TMP" | sed 's/^/  /'
sbatch "$@" "$TMP"
