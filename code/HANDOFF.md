# Codex D-MSTCN Resubmission Handoff

Last updated: 2026-07-19 21:15 IST

## Active objective

Produce a complete, scientifically honest D-MSTCN implementation and execute the
reviewer-complete experiment campaign using the datasets at
`/mnt/adissd/phd/dsctm-resubmission/dataset`. Maintain separate status, metrics, and
code-aligned mathematics records throughout.

## Current checkpoint

- The original `code/` directory is an input-only resubmission package and is not a
  Git repository.
- A full candidate Git checkout was found in sibling path `cold/dsctm`, branch
  `experimentation1`, local commit `03cc9ec` (three commits ahead of its remote).
- That checkout has one modified `HANDOFF.md`; no edits will be made there.
- Next: clone the candidate locally into `code/dsctm`, create/use an isolated Codex
  branch, rerun tests and Gate 0/1, then audit all result JSON before scheduling new
  experiments.

## Important current facts

- Hardware: one RTX 4060 Ti 16 GB. Genuine 2–8 GPU systems experiments are blocked.
- Python packages are absent from system Python, but a populated shared virtual
  environment exists at `/mnt/adissd/phd/dsctm-resubmission/venv`.
- Classic DAIC-WOZ is present for 188/189 sessions. Session 440 is quarantined as a
  source-truncated dev archive; test membership is unaffected.
- Existing matched-budget results do not reproduce the manuscript's headline accuracy
  advantage. Preserve and report this negative result.

## Resume command

Work from `/mnt/adissd/phd/dsctm-resubmission/code/dsctm` after the isolated checkout
is created. Read this file plus `artifacts/resubmission/STATUS.md`,
`METRICS_LOG.md`, and `MATHEMATICAL_FORMULATION.md` before continuing.
