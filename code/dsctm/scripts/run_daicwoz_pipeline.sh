#!/usr/bin/env bash
# DAIC-WOZ 88-dim EXP-4.2 pipeline: extract -> train -> summarize -> test.
# Sequential, abort-on-failure, clear markers. Logs to artifacts/daicwoz_pipeline.log.
set -o pipefail
cd /mnt/adissd/phd/dsctm-resubmission/cold/dsctm || exit 91
export PYTHONPATH="$PWD/src:$PWD"
VP=../../venv/bin/python
LOG=artifacts/daicwoz_pipeline.log
mkdir -p artifacts
ts(){ date '+%H:%M:%S'; }

echo "[$(ts)] PIPELINE_START" | tee -a "$LOG"

echo "[$(ts)] STEP1 extract 88-dim eGeMAPS" | tee -a "$LOG"
$VP -u scripts/build_daicwoz_egemaps88.py >>"$LOG" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[$(ts)] STEP1_FAIL rc=$rc" | tee -a "$LOG"; echo PIPELINE_FAILED; exit 1; }
echo "[$(ts)] STEP1_OK" | tee -a "$LOG"

echo "[$(ts)] STEP2 run EXP-4.2 (5 seeds + bootstrap)" | tee -a "$LOG"
$VP -u scripts/run_exp42_daicwoz.py >>"$LOG" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[$(ts)] STEP2_FAIL rc=$rc" | tee -a "$LOG"; echo PIPELINE_FAILED; exit 2; }
echo "[$(ts)] STEP2_OK" | tee -a "$LOG"

echo "[$(ts)] STEP3 summarize" | tee -a "$LOG"
$VP scripts/summarize_phase4.py >>"$LOG" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[$(ts)] STEP3_FAIL rc=$rc" | tee -a "$LOG"; echo PIPELINE_FAILED; exit 3; }
echo "[$(ts)] STEP3_OK" | tee -a "$LOG"

echo "[$(ts)] STEP4 pytest" | tee -a "$LOG"
$VP -m pytest -q >>"$LOG" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[$(ts)] STEP4_FAIL rc=$rc (tests failed; results still produced)" | tee -a "$LOG"; echo PIPELINE_DONE_TESTS_FAILED; exit 0; }
echo "[$(ts)] STEP4_OK" | tee -a "$LOG"

echo "[$(ts)] PIPELINE_DONE" | tee -a "$LOG"
echo PIPELINE_DONE
