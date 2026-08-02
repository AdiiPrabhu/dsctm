# D-MSTCN PARAM Utkarsh Campaign — Status

Last updated: 2026-07-26
Branch: `param-main` · Baseline tag: `baseline-flattened` (`52ad6b1`)
Foundation: `code/dsctm/` — see `DECISIONS.md` D-001

---

## Gate dashboard

| Gate | Title | Status | Evidence |
|---|---|---|---|
| **0** | Repository discovery and baseline freeze | ✅ **PASS** | `artifacts/gate0/` |
| **1** | Single-process correctness | ✅ **PASS** | `artifacts/gate1/` |
| **2** | Full-model DDP | ✅ LOGIC-VERIFIED (CPU/gloo) | `artifacts/gate2/` — parity 1.49e-08 |
| **3** | Distributed correctness tests | ✅ LOGIC-VERIFIED · 🔒 hardware half open | `artifacts/gate3/` |
| **4** | PARAM/SLURM infrastructure | ✅ BUILT · 🔒 not executed | `artifacts/gate4/`, `scripts/param/` |
| **5** | Fresh PARAM scientific campaign | ✅ PLANNED (294 tasks) · 🔒 not executed | `artifacts/gate5/` |
| **6** | Fresh ablation campaign | ✅ PLANNED (78 tasks) · 🔒 not executed | `artifacts/gate6/` |
| **7** | DDP systems baseline | ✅ IMPLEMENTED · 🔒 not executed | `artifacts/gate7/` |
| **8** | Scale-Aware Partitioner | ✅ EQUIVALENCE-VERIFIED (gloo ws4) · 🔒 no NCCL | `artifacts/gate8/` |
| **9** | Real TCP | ✅ IMPLEMENTED + VERIFIED · 🔒 no NCCL | `artifacts/gate9/` |
| **10** | SAP/TCP systems experiments | ✅ IMPLEMENTED · 🔒 needs 2 nodes | `artifacts/gate10/` |
| **11** | Theorem and formal claims | ✅ **RESOLVED — Outcome B** | `artifacts/gate11/` |
| **12** | Final evidence generation | ✅ IMPLEMENTED · admits nothing (correct) | `artifacts/gate12/`, `artifacts/final/` |
| — | PARAM monitoring | ✅ BUILT | `scripts/param/monitor.py`, `artifacts/monitoring/` |

**Test suite: 316 passed, 0 failures, 0 errors** (Gate 0 baseline was 31).

**PARAM-ready: NO.** Every gate from 3 onward requires execution on PARAM. The gating job is
`sbatch scripts/param/2gpu_ddp_smoke.sbatch` — see `RUNBOOK.md`.

## Gate 0 — PASS

**Files changed:** none under `cold/`, `code/`, `source/`, `reviewer-package/`. Gate 0 is
read-only with respect to all four implementation trees.

**Created:**

```
artifacts/gate0/REPOSITORY_DISCOVERY.md      required
artifacts/gate0/BASELINE_TEST_REPORT.md      required
artifacts/gate0/FILE_INVENTORY.csv           required — 212 files, SHA-256 each
artifacts/gate0/OLD_RESULT_QUARANTINE.md     required
artifacts/gate0/environment.json             supporting
artifacts/gate0/quarantined_claims.csv        supporting — 27 registered claims
artifacts/gate0/codex_baseline_tests.{xml,log}
artifacts/gate0/claude_archived_tests.{xml,log}
artifacts/gate0/source_harness_tests.{xml,log}
results/local_non_authoritative/README.md
results/param_utkarsh_authoritative/README.md
STATUS.md  DECISIONS.md  BLOCKERS.md  EXPERIMENT_LEDGER.md
```

**Tests executed and results**

| Suite | Command | Result |
|---|---|---|
| Codex foundation | `cd code/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q` | **31 passed** |
| Claude archived | `cd cold/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q` | 11 passed |
| `source/` harness | `cd source && PYTHONPATH=src:. python3 -m pytest -q` | **1 collection error** (Python ≥ 3.10 required — B-004) |

**Environment:** macOS 26.5.2 arm64 · Python 3.9.6 · PyTorch 2.8.0 · CUDA **unavailable** ·
NCCL **unavailable** · gloo available · `thop`/`opensmile`/`pyarrow` missing.

### Gate 0 findings

1. **D0-1 — `03cc9ec` does not exist here.** Single squashed root commit `0993ed0`. The requested
   `baseline-03cc9ec` tag was not fabricated; four path-scoped tags were created instead
   (D-003).
2. **D0-2 — three implementations, not two.** `code/dsctm` (5,403 LOC, foundation),
   `cold/dsctm` (3,352 LOC, archived), `source/` (851 LOC, contains a real DDP harness),
   plus `reviewer-package/code/` which is a byte-identical copy of `source/` except `README.md`.
3. **D0-3 — a working DDP harness already exists** in `source/multi_gpu_validation/` (889 LOC),
   never executed, targeting a *different* minimal model with synthetic batches only. Adopted as
   a design donor (D-002).
4. **D0-4 — `source/` cannot import on Python 3.9.** The package shipped to reviewers has a
   PEP-604 annotation defect (B-004).
5. **D0-5 — zero raw results exist.** Every artifact cited by every ledger is absent. Prior
   numbers are *unverifiable*, not merely non-authoritative (D-008, B-007).

### Gate 0 pass conditions

| Condition | Result |
|---|---|
| Codex implementation identified unambiguously | ✅ eight independent source markers |
| Baseline test suite passes | ✅ 31/31 |
| Old results cannot be confused with PARAM results | ✅ separate roots + guards; and there are no old raw results |
| No Claude correctness defect merged into the working tree | ✅ `cold/` untouched, zero cross-imports |

### Known limitations at Gate 0

- Test results validate **single-process CPU logic only**. Nothing about CUDA kernels, fp16
  numerics, NCCL, multi-rank behaviour or throughput is asserted.
- Torch 2.8.0 CPU/arm64 diverges from the declared `torch==2.6.0+cu124`. Numerical results from
  this machine are not comparable to PARAM.
- No dataset is present; all cohort statistics must be re-derived on PARAM (B-002).
- `thop` missing → FLOPs silently degrade to `"unavailable"` in `exp_0_2_params_flops`.
- One benign warning in `exp_0_4_causality` (missing `no_grad` around the determinism probe),
  queued as Gate 1 item **G1-W1**.

---

## Next gate — Gate 1: single-process correctness

Gate 1 is fully executable locally: it is CPU logic and needs no GPU and no dataset.

**Planned work**

| Item | Type |
|---|---|
| `test_lstm_baseline_uses_packed_sequences` | new test |
| `test_transformer_baseline_applies_padding_mask` | new test |
| `test_lengths_validated_and_propagated_end_to_end` | new test |
| `test_per_subject_adapter_cost_is_d_s_not_2D` | new test |
| `test_official_daicwoz_split_is_used_and_disjoint` | new test (uses `reviewer-package/data/*.csv`) |
| `test_no_duplicate_or_overlapping_window_content_across_splits` | new test |
| `test_class_weights_use_training_labels_only` | new test |
| `test_receptive_fields_are_61_481_1921_from_implementation` | strengthen (assert exact values, derive not hardcode) |
| Add `nonlinear_csag` mode; rename existing to explicit `linear_csag` alias | implementation (D-006) |
| Fix G1-W1 `no_grad` warning | implementation |
| `artifacts/gate1/SINGLE_PROCESS_CORRECTNESS.md` | artifact |
| `artifacts/gate1/DATA_PIPELINE_SEMANTICS.md` | artifact |
| `artifacts/gate1/MATHEMATICAL_CORRECTIONS.md` | artifact |

**Exact commands to reproduce Gate 0**

```bash
cd /Users/adii/Documents/phd/DSTCM_Resubmission/resubmit/dsctm
git checkout param-main

cd code/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q -rA
cd ../../cold/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q -rA
cd ../../source && PYTHONPATH=src:. python3 -m pytest -q -rA   # expected: collection error, B-004
```

---

## Open decisions requiring the author

These are not engineering choices and cannot be resolved from code:

1. **Which corpus produced the manuscript's headline** — classic DAIC-WOZ (189, 107/82) or
   E-DAIC (274, official splits)? Determines what Gate 5 runs.
2. **Is SEED in scope?** It appears in the manuscript (Experiment E6, Table 4) but is present in
   neither implementation and in no dataset directory.
3. **Is test-label access on E-DAIC authorized** for reporting?
4. **Compute budget** — node-hours, wall-clock ceiling, queue limits, max concurrent array tasks
   (B-006).
5. **Scope decision** (tracker D1-01 / D1-05 / W5-01) — whether the paper defends the accuracy
   headline or repositions around causality, efficiency and the methodology corrections. This
   should be settled before Gate 5 consumes the allocation, not after.
