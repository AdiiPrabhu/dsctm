# Continuation instructions for Claude

Continue the D-MSTCN reviewer experiment campaign in this repository until every locally
feasible, approved experiment and audit is complete. Work from the repository and live
process state as authoritative. Preserve negative, null, failed, cancelled, and quarantined
results. Never invent, improve, omit, or reinterpret a metric to support the manuscript.

## Repository and branch

- Repository: `/mnt/adissd/phd/dsctm-resubmission/codex/dsctm`
- Branch: `experimentation2`
- Push target: `origin experimentation2`
- Current code contains post-launch durability and documentation improvements.
- The live experiment itself was launched from commit `10b6c48`; never attribute later
  implementation changes to its numerical result.
- Before changing anything, run `git status --short` and preserve unrelated/user changes.
- Use `apply_patch` for source and documentation edits.

## Live corrected StudentLife experiment

The active command is:

```bash
PYTHONPATH=src /mnt/adissd/phd/dsctm-resubmission/venv/bin/python -u \
  scripts/run_exp41_corrected.py
```

Live process facts at handoff:

- PID: `59422`
- Started: `2026-07-19 22:19:20 IST`
- Launch revision: `10b6c48`
- Data cache: `artifacts/cache/studentlife_causal_ffill_v2.npz`
- Data hash: `a9cbaa3a22c2bf4e`
- Split hash: `6208d08f0b8db52b`
- Dataset audit: N=2160, T=60, F=8, 46 participants, class counts
  578/973/609, all finite, zero train/validation participant overlap.
- Runner scope: six models, three seeds per model, five grouped folds per seed.
- Model order: `dmstcn`, `lstm`, `temporal-cnn`, `transformer`, `timesnet`,
  `itransformer`.

Do not kill, suspend, renice, restart, or launch a competing GPU workload while PID 59422
is healthy. Check health with:

```bash
ps -p 59422 -o pid,lstart,etime,time,%cpu,%mem,stat,cmd
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv,noheader
```

The original interactive stdout session may not be available to Claude. This is not a
reason to restart. The runner writes a crash-resilience summary after each complete model:

```text
artifacts/resubmission/phase4/studentlife_headline_corrected_partial.json
```

and the final result to:

```text
artifacts/resubmission/phase4/studentlife_headline_corrected.json
```

Poll for those files and inspect modification time/JSON. A local ignored observation log
is at `artifacts/resubmission/phase4/studentlife_headline_corrected_live.log`.

Already emitted D-MSTCN seed metrics, each after all five folds:

- seed 0 pooled macro-F1: `0.3376`
- seed 1 pooled macro-F1: `0.3524`
- seed 2 is currently running at handoff.

These are provisional optimization-repeat metrics. Do not rank, average, or interpret them
as the D-MSTCN result until seed 2 completes and the model partial exists. When a new
complete seed/model metric becomes observable, append it to the ignored live log and add a
clearly provisional row to `METRICS.md`. Update `STATUS.md` and `HANDOFF.md`, commit, and
push. Do not report inferred fold values or metrics that the runner did not emit.

## Final EXP-4.1 admission gate

After the final JSON exists, run:

```bash
PYTHONPATH=src /mnt/adissd/phd/dsctm-resubmission/venv/bin/python \
  scripts/audit_exp41_corrected.py
```

The audit must pass before reporting a final ranking. Inspect the generated receipt:

```text
artifacts/resubmission/phase4/studentlife_headline_corrected_audit.json
```

Independently verify:

- exact models and seeds;
- five fold values per model;
- finite and in-range values;
- recomputed means and valid intervals;
- complete paired-comparison family;
- expected split/data hashes;
- source SHA-256;
- that the historical leakage-affected StudentLife artifact remains quarantined.

The summary-level audit cannot by itself prove all 90 fits ran because the launch revision
predates immutable per-fold headline writes. State this limitation. Preserve the launch
stdout/partial/final evidence and do not claim newer registry behavior for this run.

## Required living documents

Keep these synchronized with actual evidence:

- `STATUS.md` — phase dashboard, active work, blockers, next actions.
- `HANDOFF.md` — exact continuation commands, provenance, decisions, failures.
- `METRICS.md` — append-only human-readable metric ledger; raw JSON remains authoritative.
- `docs/DMSTCN_ALGORITHM.md` — code-faithful mathematical formulation.

The mathematical record was comprehensively audited and pushed. Do not weaken its evidence
boundaries. In particular:

- held-out participants map to trained FiLM unknown row 0; there is no individualized
  test-time subject adaptation;
- TCP is a tested bookkeeping simulation, not an implemented distributed optimizer;
- no real AllReduce, FedAvg training, or multi-GPU speedup is implemented or measured;
- historical simplified-TimesNet comparisons are non-confirmatory;
- StudentLife hash `62de62987570bc40` is invalid due to leading-prefix backward-fill
  leakage and must remain quarantined.

## Remaining approved experiment queue

After corrected EXP-4.1 finishes and its final evidence is committed, run the remaining
approved, locally feasible work sequentially on the single GPU. Before each launch, confirm
the exact script/config/output and preserve failures rather than silently reallocating them.

1. `scripts/run_exp13_preprocessing.py` — EXP-1.3 four leakage-safe preprocessing
   conditions on identical folds.
2. `scripts/run_exp33_delay.py` — EXP-3.3 prespecified controlled delay/XOR task with
   success and falsification criteria fixed before results.
3. `scripts/run_exp22_fair_tuning.py` — EXP-2.2/2.3 equal eight-trial model-specific
   dev search for all six models, followed by frozen five-seed confirmation. Test must
   remain inaccessible during search. Use the complete DAIC-WOZ cache.
4. `scripts/build_daicwoz_participant_egemaps88.py` — participant-only representation;
   then run the corresponding fair all-model refinement only after verifying cache and
   split provenance. Do not tune only D-MSTCN.

Review `STATUS.md` and `HANDOFF.md` before launching because the queue may be refined as
audits complete. Run one GPU experiment at a time.

## Phase 5 approval boundary

The complete implemented Phase-5 family has 14 conditions x 3 seeds x 5 folds = 210 fits.
The earlier approval covered 105 fits. Do not launch the expanded 210-fit Phase 5 run
without explicit revised compute approval. Implementation completeness is not experiment
completion. If revised approval is obtained, use `scripts/run_phase5_ablation.py`, preserve
every fold via the immutable registry, and apply the prespecified 13-comparison Holm/BH
family exactly as documented.

## Verification and git discipline

For code changes, run focused tests and then the full CPU suite when appropriate:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/adissd/phd/dsctm-resubmission/venv/bin/python -m pytest -q
git diff --check
```

At this handoff the full suite passes 31/31. Push reviewed commits with:

```bash
git push origin experimentation2
```

Do not use destructive git commands. Do not commit raw datasets, participant identifiers,
or caches. Artifacts under `artifacts/` are intentionally ignored; report their paths and
hashes in the tracked ledgers.

## Completion standard

Do not declare completion merely because scripts exist or a subset of runs finished.
Completion requires a requirement-by-requirement audit showing all approved locally feasible
experiments finished, all outputs passed their admission checks, failures/negative results
remain visible, living documents match raw evidence, the full test suite passes, the
worktree is clean, and reviewed commits are pushed. Hardware-only 2--8 GPU scaling and
missing manuscript/decision-letter work must remain explicit external blockers rather than
fabricated deliverables.
