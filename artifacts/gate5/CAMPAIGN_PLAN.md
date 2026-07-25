# Gate 5 — Fresh PARAM Scientific Campaign

Generated: 2026-07-26 · Branch `param-main`
Evidence: `gate5_tests.xml` (181 passed), `campaign_plan.json`

**Status: PLAN BUILT AND TESTED. NOT EXECUTED.** No task has run on PARAM.
Gate 5 passes when the fail-closed auditor admits a complete experiment family from
`results/param_utkarsh_authoritative/`.

---

## 1. The matrix — 294 tasks, every one uniquely addressable

| Family | Tasks | `--array` | Plan digest |
|---|---:|---|---|
| `tuning-studentlife` | 48 | `0-47%4` | `99e8187f2cd00615` |
| `tuning-daicwoz` | 48 | `0-47%4` | `236956a0f6b99059` |
| `confirm-studentlife` | 60 | `0-59%4` | `6bfcb4986429ff6c` |
| `confirm-daicwoz` | 60 | `0-59%4` | `9f44701755325d6b` |
| `ablation` | 78 | `0-77%4` | `7876bb29c494c34b` |
| **TOTAL** | **294** | | `dcb6e197431c369d` |

By experiment: EXP-2.2 96 · EXP-4.1 60 · EXP-4.2 60 · EXP-5.1 21 · EXP-5.2 18 ·
EXP-5.3 15 · EXP-1.3 12 · EXP-5.5 12.

**294 unique task ids / 294 tasks** — asserted, not assumed.

### 1.1 A bug this module caught immediately

`ablation_array.sbatch` was written at Gate 4 with `--array=0-27%4`. The real ablation
family has **78** tasks. Fifty would have been silently dropped, every submitted task would
have succeeded, and the campaign would have reported completion.

This is exactly the failure the plan module exists to prevent, and it surfaced on the first
run of `scripts/param/plan.py`. The bound is now derived
(`plan.py --sbatch-array <family>`) and a test asserts every sbatch bound matches its
family length.

---

## 2. Design decisions

### 2.1 One entry point, not one script per family

`scripts/param/run_task.py --family F --index N` handles every task. The run-directory
contract, provenance capture, cross-rank hash agreement and failure semantics are therefore
identical everywhere. Per-family scripts drift; a single entry point does not.

### 2.2 The plan is data, not control flow

Tasks are built from prespecified constants by nested loops over sorted keys. Consequences:

* deterministic — index N means the same task in every submission
* append-only — adding a model or seed extends its family without renumbering earlier tasks
* hashable — `plan_digest` is recorded in every run directory, so plan drift is detectable
  after the fact rather than being invisible

`get_task` raises `IndexError` on an out-of-range index. It never returns `None` and never
silently skips.

### 2.3 Equal tuning budget is enforced, not documented

`TUNING_TRIALS_PER_MODEL = 8`. Every model's search space is validated to yield exactly
eight trials, and `tuning_tasks` **raises** otherwise. Search *spaces* differ because the
architectures differ; the *budget* is identical, which is what tracker E4-01 (R2, R6)
actually requires.

A test monkeypatches a 2-trial space and asserts the plan refuses to build.

### 2.4 Tuning tasks cannot reach test

Tuning protocol is `official_train_dev_search`. `run_task.py` calls `train_model`
(train + dev only) for the tuning family and `train_select_evaluate` only for
confirmation. A test asserts no tuning task declares a protocol containing "test".

### 2.5 Ablation families (Gate 6 content, planned here)

| Family | Variants | × seeds | Tasks |
|---|---:|---:|---:|
| Branch combinations (all 7) | 7 | 3 | 21 |
| Dilation schedules | 5 | 3 | 15 |
| Fusion (incl. `linear_csag` **and** `nonlinear_csag`) | 6 | 3 | 18 |
| Personalization (incl. parameter-matched global) | 4 | 3 | 12 |
| Preprocessing | 4 | 3 | 12 |

Dilation schedules are named (`original`, `compressed`, `expanded`, `uniform`,
`duration_aligned`); their **receptive fields are derived from `Branch.theoretical_rf_two_conv()`
at run time, never typed**. A test recomputes all 15 and asserts the formula.

---

## 3. Run-directory contract

`campaign/rundir.py` writes all 15 required files and `finalize()` **refuses to record
`completed` when any is missing** — it downgrades to `infrastructure_failed` and writes
`contract_violation` naming the missing files.

Ordering inside `finalize()` is deliberate:

1. `_receipt()` — so `receipt.sha256` exists when the contract is audited
2. `audit()` — over the now-complete file set
3. `status.json` — records the receipt and the verdict

`status.json` is **excluded from the receipt hash**. Including it would be circular: the
receipt would have to cover a file that quotes the receipt. So the receipt binds the
*evidence* and `status.json` binds the *verdict* to that evidence by quoting its hash.

This ordering was wrong on the first implementation — `audit()` ran before the receipt was
written, so `receipt.sha256` always registered as missing and every run downgraded to
`infrastructure_failed`. Caught by `test_complete_run_is_accepted`.

Waivable files (`checkpoint.pt`, `predictions.parquet`) are accepted **only** with a
written reason recorded in `status.json`. Missing pyarrow writes `predictions.jsonl` plus a
waiver rather than silently producing nothing.

`run.open()` is called **before** training, so a job killed mid-run still leaves full
provenance rather than an empty directory.

---

## 4. Distributed safety in the task runner

Before any training starts, `run_task.py` asserts across ranks:

| Quantity | Failure meaning |
|---|---|
| `task_id` | ranks are running different experiments |
| `plan_digest` | ranks built different plans |
| `dataset_hash` | ranks loaded different data |
| `split_hash` | ranks are using different splits |

Any disagreement raises `PreflightFailure` on every rank. All four produce
plausible-looking, completely invalid results if left undetected.

Dataset loading and training are wrapped in `fail_together`, so a failure on one rank
raises on all — no rank is left blocked in a collective holding a GPU reservation until the
72-hour limit expires.

---

## 5. Test coverage (48 added, 181 total)

| Area | Tests | Notable |
|---|---:|---|
| Plan determinism | 5 | same ids across invocations; stable digests |
| Array bounds | 12 | `--array` matches plan length; out-of-range is a hard error |
| Tuning fairness | 4 | equal budget enforced; a wrong budget is refused |
| Test isolation | 1 | no tuning task declares a test protocol |
| Confirmation coverage | 1 | 10 seeds × 6 models |
| Ablation coverage | 4 | every prespecified variant present; RF derived not typed |
| Run contract | 8 | incomplete run cannot be `completed`; receipt binds content |
| Rank discipline | 1 | non-main rank writes nothing |

---

## 6. Not yet done in Gate 5

| Item | Status |
|---|---|
| Execute any task on PARAM | ⛔ requires Phases 0–4 of `RUNBOOK.md` |
| Freeze tuned configs into the confirmation family | ⬜ needs real tuning output; currently `config_source: frozen_from_tuning` |
| `scripts/param/audit_campaign.py` — family-level fail-closed admission | ⬜ next |
| Aggregate statistics with Holm/BH across the admitted family | ⬜ next (`eval/statistics.py` already has the primitives) |
| Transfer families (E-DAIC, SEED) | ⬜ blocked on the corpus decision and SEED scope |

The confirmation family currently carries a placeholder for the frozen configuration. It is
resolved after tuning completes and before the seeds array is submitted — which is why the
runbook orders them tuning → seeds and says so.
