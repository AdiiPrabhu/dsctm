# EXPERIMENT LEDGER

Append-only index of every experiment attempted in the PARAM campaign. Raw artifacts under
`results/param_utkarsh_authoritative/` are authoritative; this file is a human-readable index and
must point at them. Failed, cancelled and negative runs stay visible permanently.

Last updated: 2026-07-26 (Gate 0)

---

## Verification states

| State | Meaning |
|---|---|
| `AUTHORITATIVE` | Executed on PARAM Utkarsh, complete run directory, admitted by the fail-closed auditor, receipt issued |
| `PARAM-PENDING-AUDIT` | Executed on PARAM, artifacts complete, auditor not yet run |
| `PARAM-INCOMPLETE` | Executed on PARAM, run directory missing required files — **not citable** |
| `LOGIC-VERIFIED (CPU/gloo)` | Logic validated locally without GPU. Never evidence for a hardware claim |
| `LOCAL-DEBUG` | Non-PARAM output under `results/local_non_authoritative/`. Never citable |
| `QUARANTINED` | Historical RTX 4060 Ti claim with no backing artifact. Never citable |
| `BLOCKED` | Cannot execute; see `BLOCKERS.md` |

---

## PARAM Utkarsh runs

**None. No experiment has been executed on PARAM Utkarsh.**

`results/param_utkarsh_authoritative/` contains only its policy `README.md`.

| Run ID | Experiment | Dataset | World size | Seeds | Status | Receipt |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

---

## Local verification runs

| ID | What | Command | Result | State | Evidence |
|---|---|---|---|---|---|
| L-0001 | Codex foundation baseline suite | `cd codex/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q` | **31 passed**, 1 benign warning | `LOGIC-VERIFIED (CPU)` | `artifacts/gate0/codex_baseline_tests.{xml,log}` |
| L-0002 | Claude archived suite (evidence only) | `cd claude/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q` | 11 passed | `LOGIC-VERIFIED (CPU)` | `artifacts/gate0/claude_archived_tests.{xml,log}` |
| L-0003 | `source/` DDP harness import | `cd source && PYTHONPATH=src:. python3 -m pytest -q` | **1 collection error** — `TypeError` on PEP-604 annotation, Python 3.9 | `BLOCKED` (B-004) | `artifacts/gate0/source_harness_tests.{xml,log}` |
| L-0004 | Repository file inventory | Gate 0 inventory script | 212 files hashed; 119 Python files, 10,457 LOC | `LOGIC-VERIFIED (CPU)` | `artifacts/gate0/FILE_INVENTORY.csv` |
| L-0005 | Environment capture | Gate 0 environment script | Python 3.9.6, torch 2.8.0, CUDA absent, NCCL absent, gloo present | — | `artifacts/gate0/environment.json` |

---

## Planned experiment matrix

Nothing below has been launched. Sizes are estimates for the compute request (B-006), not results.

### Gate 5 — scientific campaign (single V100 per job, SLURM arrays)

| ID | Experiment | Dataset | Models | Trials / seeds | Fits | Status |
|---|---|---|---|---|---:|---|
| P5-TUNE-SL | Equal-budget dev tuning | StudentLife | 6 | 8 dev trials each | 48 | `BLOCKED` B-001/B-002/B-006 |
| P5-TUNE-DW | Equal-budget dev tuning | DAIC-WOZ | 6 | 8 dev trials each | 48 | `BLOCKED` |
| P5-CONF-SL | Frozen-config confirmation | StudentLife | 6 | 10 seeds × 5 folds | 300 | `BLOCKED` |
| P5-CONF-DW | Frozen-config confirmation | DAIC-WOZ | 6 | 10 seeds | 60 | `BLOCKED` |
| P5-XFER-EDAIC | Transfer evaluation | E-DAIC | 6 | 10 seeds | 60 | `BLOCKED` (+ corpus decision) |
| P5-XFER-SEED | Transfer evaluation | SEED | 3 | 10 seeds | 30 | `BLOCKED` (+ scope decision) |

### Gate 6 — ablations (single V100 per job)

| ID | Family | Variants | Protocol | Fits | Status |
|---|---|---:|---|---:|---|
| P6-BRANCH | All branch combinations | 7 | 5 folds × 3 seeds | 105 | `BLOCKED` |
| P6-DILATION | Dilation schedules (original / compressed / expanded / uniform / duration-aligned) | 5 | 5 folds × 3 seeds | 75 | `BLOCKED` |
| P6-FUSION | mean / static / `linear_csag` / `nonlinear_csag` / temp ×0.5 / temp ×2 | 6 | 5 folds × 3 seeds | 90 | `BLOCKED` |
| P6-PERSON | no-FiLM / subject / global / matched-global / unknown-subject / cold-start | 6 | 5 folds × 3 seeds | 90 | `BLOCKED` |
| P6-PREPROC | causal ffill / zero / train-mean / mask-channel | 4 | 5 folds × 3 seeds | 60 | `BLOCKED` |

### Gate 7 — DDP systems baseline (multi-GPU)

| ID | Config | Nodes × GPUs | Mode | Repeats | Status |
|---|---|---|---|---:|---|
| P7-S1 | Single GPU | 1 × 1 | reference | 5 | `BLOCKED` B-001 |
| P7-S2 | One-node DDP | 1 × 2 | strong + weak | 5 each | `BLOCKED` |
| P7-S4 | Two-node DDP | 2 × 4 | strong + weak | 5 each | `BLOCKED` |
| P7-S8 | Four-node DDP | 4 × 8 | strong + weak | 5 each | `BLOCKED` |
| P7-S16 | Eight-node DDP | 8 × 16 | strong + weak | 5 each | `BLOCKED` (+ allocation) |

### Gates 8–10 — SAP / TCP

| ID | Experiment | Min ranks | Status |
|---|---|---:|---|
| P8-PARITY | SAP forward/backward equivalence vs monolithic | 4 | `BLOCKED` B-001 |
| P9-MODES | 4 modes: sync DDP / sync SAP / async SAP no-TCP / async SAP + TCP | 4 | `BLOCKED` |
| P10-SWEEP | `delta_max`, `T_sync`, branch delay, straggler rate, latency, jitter, bandwidth, imbalance | 4 | `BLOCKED` |

**Note on Gate 7/8 topology.** A PARAM node has two V100s. The manuscript's one-branch-per-GPU
SAP (3 branches + aggregator) needs ≥ 4 ranks = 2 nodes. See `DECISIONS.md` D-007.

---

## Quarantined historical claims

27 numeric claims from the RTX 4060 Ti campaign are registered in
`artifacts/gate0/quarantined_claims.csv`, all with **no backing artifact in this repository**.
None may be cited. Details: `artifacts/gate0/OLD_RESULT_QUARANTINE.md`.

Three prior findings survive because they are properties of code or arithmetic and were
re-verified at Gate 0:

| Finding | Value | Re-verified by |
|---|---|---|
| Receptive fields SSB / MSB / LSB | 61 / 481 / 1921 (manuscript's 47 / 383 / 1535 wrong) | `test_measured_rf_matches_two_conv_formula` (L-0001) |
| Per-subject stored adapter cost | `d_s` = 8, not 2D = 256 | structural; explicit assertion added in Gate 1 |
| Two-sided exact Wilcoxon, n = 5 | min p = 0.0625 → p < 0.05 unreachable | `test_wilcoxon_n5_two_sided_cannot_reach_significance` (L-0001) |

---

## Required fields for every future PARAM entry

Run ID · experiment/condition · dataset and feature version · dataset hash · split hash · config
hash · fold/repeat/seed · world size and rank count · node count · start/end time · status and
failure class · selection metric · per-class and aggregate metrics · calibration metrics · timing
where relevant · prediction file path · CI/effect-size method · receipt SHA-256.
