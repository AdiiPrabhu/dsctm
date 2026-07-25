# results/param_utkarsh_authoritative/ — THE ONLY EVIDENCE ROOT

This is the sole source for manuscript tables, figures and reviewer-response claims.

## Admission policy

A run directory may exist here only if it was produced on PARAM Utkarsh AND contains every file
required by the Gate 4 run-directory contract:

    command.txt            resolved_config.yaml   environment.json
    git.json               slurm.json             hardware.json
    dataset_hashes.json    split_hashes.json      stdout.log
    stderr.log             metrics.json           predictions.parquet
    checkpoint.pt          status.json            receipt.sha256

A run missing any required file is NOT complete, regardless of exit code.

An experiment family may be cited only after the fail-closed auditor
(`scripts/audit_*.py`) admits it and emits a SHA-256 receipt under `receipts/`.

## Rules

1. Never hand-edit a metric here.
2. Never copy anything in from `results/local_non_authoritative/`.
3. Never mark a run complete without its artifacts.
4. Every displayed number must resolve to: run ID, git SHA, config hash, dataset hash,
   split hash, seed, prediction file, metric computation, receipt.

## Status at Gate 0 (2026-07-26)

Empty. No PARAM Utkarsh run has been executed. See `BLOCKERS.md` B-001.
