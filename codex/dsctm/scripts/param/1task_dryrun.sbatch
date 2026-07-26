#!/bin/bash
#SBATCH --account=DSCTM_ACCOUNT_PLACEHOLDER
#SBATCH --job-name=dsctm-dryrun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%J.out
#SBATCH --error=logs/%x.%J.err
#
# DRY RUN: one real task per family (5 tasks), fully instrumented, before committing to
# the 294-task campaign.
#
# Answers, with measurements rather than assumptions:
#   * does a task complete end-to-end on PARAM?
#   * does it produce all 15 contract files?
#   * what does one task actually cost in wall-clock and GPU memory?
#   * how many epochs does early stopping really use?
#
# Then extrapolate and decide. Never submit 294 tasks on an estimate.
source scripts/param/_header.inc
mkdir -p logs

echo "=== plan ==="
python scripts/param/plan.py

echo; echo "=== dry run: 1 task per family ==="
python scripts/param/calibrate.py --sample 1 \
  --out "$DSCTM_RESULTS_ROOT/calibration_${SLURM_JOB_ID}.json"

echo; echo "=== extrapolation to the full campaign ==="
python scripts/param/calibrate.py \
  --extrapolate "$DSCTM_RESULTS_ROOT/calibration_${SLURM_JOB_ID}.json"

echo; echo "=== contract audit of what the dry run produced ==="
python scripts/param/audit_campaign.py --all --allow-partial \
  --out "$DSCTM_RESULTS_ROOT/dryrun_audit_${SLURM_JOB_ID}.json" || true

echo; echo "done=$(date -u +%FT%TZ)"
