# Gate 0 — Baseline Test Report

Captured: 2026-07-25T19:28:03Z
Git SHA: `52ad6b1714c89d0e543c389004fe41c8c3ef1fc6` · Branch: `param-main`
Dirty at capture: **yes** — Gate 0 artifacts were being written; no source file under
`claude/`, `codex/` or `source/` was modified before these runs.
Machine-readable copy: `artifacts/gate0/environment.json`

---

## 1. Audit environment

| Property | Value |
|---|---|
| Platform | macOS 26.5.2, arm64 (Apple Silicon) |
| Python | **3.9.6** (Clang 21.0.0) |
| PyTorch | **2.8.0** |
| PyTorch CUDA build | `None` (CPU/MPS wheel) |
| `torch.cuda.is_available()` | **False** |
| CUDA device count | **0** |
| MPS available | True (not used — not a PARAM target) |
| `torch.distributed` available | True |
| NCCL available | **False** |
| Gloo available | **True** |
| MPI available | False |
| numpy / scipy / scikit-learn / pandas / PyYAML | 2.0.2 / 1.13.1 / 1.6.1 / 2.3.3 / 6.0.3 |
| pytest | 8.4.2 |
| thop / opensmile / pyarrow | **MISSING** (all three) |

### 1.1 Divergence from the declared project environment

Both `codex/dsctm/requirements.txt` and `claude/dsctm/requirements.txt` declare `torch>=2.6`
built for CUDA 12.4; the repo-root `requirements.txt` pins `torch==2.6.0+cu124` and
`torchvision==0.21.0+cu124`. **The audit machine runs torch 2.8.0 CPU on arm64.** Baseline test
results below are therefore valid as *logic* regression evidence and **not** as numerical or
performance evidence. This is expected and does not block Gate 0; it does bound what Gates 1–3
can assert locally.

Missing optional packages and their effect:

- `thop` — `exp_0_2_params_flops` silently degrades to `flops = "unavailable (...)"` via its
  `except Exception` branch. Parameter counts are unaffected; FLOPs are simply not produced.
  Required for tracker item **E4-07**. Tracked as **B-005**.
- `opensmile` — blocks all eGeMAPS feature extraction (`build_daic_egemaps88.py`,
  `build_daicwoz_*`). Required before Gate 5. Tracked as **B-002**.
- `pyarrow` — required for the `predictions.parquet` artifact mandated by Gate 4. Tracked as
  **B-005**.

---

## 2. Test commands

All suites were run with the working directory set to the package root and with test isolation
from CUDA:

```bash
# Codex foundation
cd codex/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q -rA \
  --junitxml=../../artifacts/gate0/codex_baseline_tests.xml

# Claude archived evidence
cd claude/dsctm && PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q -rA \
  --junitxml=../../artifacts/gate0/claude_archived_tests.xml

# source/ DDP harness
cd source && PYTHONPATH=src:. python3 -m pytest -q -rA \
  --junitxml=../artifacts/gate0/source_harness_tests.xml
```

`PYTHONPATH=src:.` is mandatory. Bare `pytest` fails collection in `codex/` and `claude/`
because neither package is pip-installed in this environment; the `:.` component is additionally
required for the cross-script `from scripts...` imports.

---

## 3. Results

| Suite | Files | Tests | Passed | Failed | Errors | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **`codex/dsctm` (foundation)** | 15 | **31** | **31** | 0 | 0 | ✅ **PASS** |
| `claude/dsctm` (archived) | 4 | 11 | 11 | 0 | 0 | ✅ pass (evidence only) |
| `source/` (DDP harness) | 1 | — | — | — | **1 collection error** | ❌ **FAIL** |

Raw evidence: `codex_baseline_tests.{xml,log}`, `claude_archived_tests.{xml,log}`,
`source_harness_tests.{xml,log}` in this directory.

### 3.1 Codex — 31 passed (the PARAM foundation)

```
tests/test_ablation_modes.py::test_prespecified_phase5_control_family_is_complete
tests/test_ablation_modes.py::test_static_csag_and_global_film_modes_have_expected_semantics
tests/test_ablation_statistics.py::test_ablation_multiplicity_helpers_preserve_shape_and_bounds
tests/test_causality.py::test_causality_and_reproducibility
tests/test_causality.py::test_tcp_invariants
tests/test_causality.py::test_padding_mask_makes_head_invariant_to_masked_tail
tests/test_delay_task.py::test_delay_task_is_balanced_grouped_and_has_exact_signal_positions
tests/test_fair_tuning.py::test_equal_prespecified_search_budget_and_model_specific_spaces
tests/test_headline_registry.py::test_studentlife_headline_registers_every_completed_fold
tests/test_metrics.py::test_binary_metrics_include_pr_auc_and_calibration
tests/test_participant_audio.py::test_participant_intervals_exclude_interviewer_and_invalid_rows
tests/test_receptive_field.py::test_measured_rf_matches_two_conv_formula
tests/test_receptive_field.py::test_manuscript_printed_rf_is_wrong
tests/test_registry_artifacts.py::test_completed_fit_writes_required_artifacts_without_subject_ids
tests/test_registry_artifacts.py::test_completed_fit_is_resume_safe_and_failure_is_preserved
tests/test_result_audit.py::test_result_audit_accepts_complete_consistent_summary
tests/test_result_audit.py::test_result_audit_rejects_incomplete_or_inconsistent_summary
tests/test_splits.py::test_subject_grouped_folds_have_no_leakage
tests/test_splits.py::test_leakage_detector_catches_overlap
tests/test_splits.py::test_grouped_stratification_does_not_reduce_multiclass_subjects_to_mean_label
tests/test_statistics.py::test_wilcoxon_n5_two_sided_cannot_reach_significance
tests/test_statistics.py::test_significance_reachable_at_n6
tests/test_statistics.py::test_cluster_bootstrap_ci_brackets_point_estimate
tests/test_statistics.py::test_paired_effect_sizes_sign
tests/test_statistics.py::test_multiplicity_monotone
tests/test_studentlife_imputation.py::test_forward_fill_never_uses_a_future_observation
tests/test_studentlife_imputation.py::test_forward_fill_all_missing_is_zero
tests/test_studentlife_imputation.py::test_train_mean_normalizer_ignores_missing_and_padding
tests/test_studentlife_imputation.py::test_early_v2_cache_filename_recovers_unambiguous_version
tests/test_studentlife_imputation.py::test_ambiguous_metadata_free_cache_remains_legacy
tests/test_timesnet.py::test_official_timesnet_shape_and_mask
```

One warning, benign and identical in both `codex` and `claude`:

```
src/dsctm/experiments/gate0.py:213: UserWarning: Converting a tensor with requires_grad=True
to a scalar may lead to unexpected behavior. Consider using tensor.detach() first.
```

`exp_0_4_causality` step (3) calls `model(X, s)` outside `torch.no_grad()` when measuring
determinism. Cosmetic — the comparison is still correct — but it should be fixed in Gate 1 for a
clean run log. Tracked as Gate 1 item **G1-W1**.

### 3.2 Coverage the Codex suite already provides (relevant to Gate 1)

| Gate 1 requirement | Existing test | Sufficient? |
|---|---|---|
| Altering padded values does not change logits | `test_padding_mask_makes_head_invariant_to_masked_tail` | ✅ |
| Normalization ignores padded timesteps | `test_train_mean_normalizer_ignores_missing_and_padding` | ✅ |
| No fill from a future observation | `test_forward_fill_never_uses_a_future_observation` | ✅ |
| All-missing channel fallback | `test_forward_fill_all_missing_is_zero` | ✅ |
| Preprocessing semantics versioned/hashed | `test_early_v2_cache_filename_recovers_unambiguous_version`, `test_ambiguous_metadata_free_cache_remains_legacy` | ✅ |
| Participants disjoint across splits | `test_subject_grouped_folds_have_no_leakage`, `test_leakage_detector_catches_overlap` | ✅ |
| Multiclass grouped stratification valid | `test_grouped_stratification_does_not_reduce_multiclass_subjects_to_mean_label` | ✅ |
| Strict causality / zero future leakage | `test_causality_and_reproducibility` | ✅ |
| RF = 61 / 481 / 1921 | `test_measured_rf_matches_two_conv_formula` | ✅ |
| **LSTM uses packed sequences** | — | ❌ **must add** |
| **Transformer uses a padding mask** | — | ❌ **must add** |
| **`lengths` validated and propagated** | partial (constructor validation only) | ⚠️ **must extend** |
| **Per-subject adapter cost = `d_s`** | — asserted only in a report string | ❌ **must add** |
| **Official DAIC-WOZ split used** | — | ❌ **must add** |
| **Duplicate-content / overlapping-window check** | — | ❌ **must add** |
| **Class weights from training data only** | — | ❌ **must add** |

Gate 1 therefore adds **7 new regression tests** and extends 1.

### 3.3 `source/` harness — collection error

```
ERROR collecting tests/test_model.py
src/dmstcn/model.py:132: in DMSTCN
    def forward(self, inputs: Tensor, subject_ids: Tensor, mask: Tensor | None = None)
E   TypeError: unsupported operand type(s) for |: 'torch._C._TensorMeta' and 'NoneType'
```

PEP-604 union syntax in an evaluated annotation with no `from __future__ import annotations`.
The package requires Python ≥ 3.10. This means the shipped `reviewer-package/` **cannot be
imported by a reviewer running Python 3.9** — a reproducibility defect in the artifact that was
sent to reviewers, independent of anything else in this audit. Recorded as **B-004**; a one-line
fix is queued for Gate 2 when the harness primitives are ported.

---

## 4. Reproducibility of this report

| Item | Value |
|---|---|
| Test invocation | §2 above, verbatim |
| Randomness | Codex tests seed via `dsctm.repro.set_seed`; no test depends on wall-clock or network |
| Determinism caveat | `torch.use_deterministic_algorithms` is **not** enabled in the Codex suite; on CUDA some kernels may vary. Not an issue for this CPU run. |
| Re-run cost | Codex suite ≈ 90 s wall-clock on this machine, CPU only |

---

## 5. Verdict

**Gate 0 baseline test condition: PASS.**

The Codex foundation passes 31/31 with no failures and no errors. The archived Claude tree
passes 11/11 and is frozen as evidence. The `source/` harness fails to import on the audit
Python and is treated as a donor of design patterns only, not as an execution path.

**These results validate single-process CPU logic only.** They assert nothing about CUDA
kernels, fp16 numerics, NCCL, multi-rank behaviour, or throughput. No claim of "PARAM-ready" is
made or implied at this gate.
