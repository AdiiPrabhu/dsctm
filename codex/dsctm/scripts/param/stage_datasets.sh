#!/bin/bash
# Dataset staging for PARAM Utkarsh.
#
#   bash scripts/param/stage_datasets.sh --daicwoz      # ~86 GB
#   bash scripts/param/stage_datasets.sh --edaic
#   bash scripts/param/stage_datasets.sh --studentlife  # needs a Kaggle API token
#   bash scripts/param/stage_datasets.sh --all
#   bash scripts/param/stage_datasets.sh --verify       # hash + count only, no download
#
# WHERE TO RUN THIS
#   Downloads go on the LOGIN NODE. They are I/O bound and low CPU, which fits inside the
#   login-node limits (User Manual p.9: CPU-time and memory limits, exceed them and the
#   process is killed). Feature EXTRACTION is CPU-heavy and must NOT run here — submit
#   scripts/param/extract_features.sbatch to the `cpu` partition instead.
#
#   If compute nodes turn out to have no internet egress (common, and unconfirmed for
#   PARAM), this login-node split is not an optimisation but a requirement.
#
# RESUMABILITY
#   wget -c everywhere. A killed download resumes; it does not restart. Every archive is
#   size-checked against the server's Content-Length before extraction, because a truncated
#   zip that extracts partially is worse than one that fails loudly — the prior campaign
#   lost a session to exactly this (DAIC-WOZ 440, corrupt at source).
set -euo pipefail

DAICWOZ_URL="https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
EDAIC_URL="https://dcapswoz.ict.usc.edu/wwwedaic/"
STUDENTLIFE_KAGGLE="dartweichen/student-life"

DATA_ROOT="${DSCTM_DATA_ROOT:-$HOME/scratch/dsctm/datasets}"
DAICWOZ_ROOT="${DSCTM_DAICWOZ_ROOT:-$DATA_ROOT/DAIC-WOZ}"
EDAIC_ROOT="${DSCTM_EDAIC_ROOT:-$DATA_ROOT/E-DAIC}"
SL_ROOT="${DSCTM_STUDENTLIFE_ROOT:-$DATA_ROOT/StudentLife/dataset}"
MANIFEST_DIR="${DATA_ROOT}/_manifests"
mkdir -p "$DATA_ROOT" "$MANIFEST_DIR"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

require_wget() {
  command -v wget >/dev/null 2>&1 || { echo "FATAL: wget not found"; exit 1; }
}

# --------------------------------------------------------------------------- #
verify_zip() {
  local f="$1"
  if ! unzip -t "$f" >/dev/null 2>&1; then
    log "CORRUPT: $f  (quarantining as ${f}.CORRUPT)"
    mv "$f" "${f}.CORRUPT"
    echo "$(basename "$f")" >> "$MANIFEST_DIR/corrupt_archives.txt"
    return 1
  fi
  return 0
}

# --------------------------------------------------------------------------- #
stage_daicwoz() {
  require_wget
  log "DAIC-WOZ (AVEC2017) -> $DAICWOZ_ROOT"
  log "  source: $DAICWOZ_URL   (~86 GB, 189 sessions)"
  mkdir -p "$DAICWOZ_ROOT"
  cd "$DAICWOZ_ROOT"

  # Official AVEC2017 split + label files. These define 107 train / 35 dev / 47 test.
  for f in train_split_Depression_AVEC2017.csv \
           dev_split_Depression_AVEC2017.csv \
           full_test_split.csv \
           test_split_Depression_AVEC2017.csv; do
    wget -c -q --show-progress "${DAICWOZ_URL}${f}" || log "  (optional file absent: $f)"
  done

  # Session archives 300_P.zip .. 492_P.zip. Not every id in the range exists; a 404 is
  # expected and is skipped rather than treated as a failure.
  local ok=0 miss=0 bad=0
  for pid in $(seq 300 492); do
    local z="${pid}_P.zip"
    [[ -d "${pid}_P" ]] && { ok=$((ok+1)); continue; }
    if wget -c -q "${DAICWOZ_URL}${z}"; then
      if verify_zip "$z"; then
        unzip -q -o "$z" -d "${pid}_P" && rm -f "$z" && ok=$((ok+1))
      else
        bad=$((bad+1))
      fi
    else
      miss=$((miss+1))
    fi
  done
  log "DAIC-WOZ: $ok extracted, $miss absent, $bad corrupt"
  {
    echo "source=$DAICWOZ_URL"
    echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "sessions_ok=$ok sessions_absent=$miss sessions_corrupt=$bad"
  } > "$MANIFEST_DIR/daicwoz.manifest"
}

# --------------------------------------------------------------------------- #
stage_edaic() {
  require_wget
  log "E-DAIC (AVEC2019) -> $EDAIC_ROOT"
  log "  source: $EDAIC_URL   (275 sessions, official 163/56/56)"
  mkdir -p "$EDAIC_ROOT/labels"
  cd "$EDAIC_ROOT"
  for f in train_split.csv dev_split.csv test_split.csv \
           Detailed_PHQ8_Labels.csv metadata_mapped.csv; do
    wget -c -q --show-progress "${EDAIC_URL}labels/${f}" -O "labels/${f}" \
      || wget -c -q "${EDAIC_URL}${f}" -O "labels/${f}" \
      || log "  (could not fetch $f — check the directory listing at $EDAIC_URL)"
  done
  # The repository already ships these split CSVs under reviewer-package/data/; the
  # downloaded copies are cross-checked against them by scripts/param/verify_datasets.py.
  log "E-DAIC: label files staged. Feature archives are listed at $EDAIC_URL —"
  log "        mirror the ones your EULA covers into $EDAIC_ROOT/data/"
  {
    echo "source=$EDAIC_URL"
    echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$MANIFEST_DIR/edaic.manifest"
}

# --------------------------------------------------------------------------- #
stage_studentlife() {
  log "StudentLife -> $SL_ROOT"
  log "  source: https://www.kaggle.com/datasets/${STUDENTLIFE_KAGGLE}"
  if ! command -v kaggle >/dev/null 2>&1; then
    log "FATAL: kaggle CLI not found."
    log "  pip install kaggle"
    log "  then place your API token at ~/.kaggle/kaggle.json (chmod 600)"
    log "  token: kaggle.com -> Account -> Create New API Token"
    return 1
  fi
  if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    log "FATAL: ~/.kaggle/kaggle.json missing. Kaggle downloads need an API token."
    return 1
  fi
  mkdir -p "$(dirname "$SL_ROOT")"
  kaggle datasets download -d "$STUDENTLIFE_KAGGLE" -p "$(dirname "$SL_ROOT")" --unzip
  log "StudentLife staged. Expect sensing/ and EMA/ subtrees."
  {
    echo "source=kaggle:${STUDENTLIFE_KAGGLE}"
    echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$MANIFEST_DIR/studentlife.manifest"
}

# --------------------------------------------------------------------------- #
verify() {
  log "verifying staged datasets"
  python "${DSCTM_REPO_ROOT:-.}/scripts/param/verify_datasets.py" \
    --json "$MANIFEST_DIR/dataset_hashes.json"
}

# --------------------------------------------------------------------------- #
case "${1:---help}" in
  --daicwoz)     stage_daicwoz ;;
  --edaic)       stage_edaic ;;
  --studentlife) stage_studentlife ;;
  --all)         stage_daicwoz; stage_edaic; stage_studentlife; verify ;;
  --verify)      verify ;;
  *)
    sed -n '2,30p' "$0"
    exit 0 ;;
esac

log "done. Next: python scripts/param/preflight.py"
