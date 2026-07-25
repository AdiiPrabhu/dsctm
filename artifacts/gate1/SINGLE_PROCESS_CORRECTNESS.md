# Gate 1 — Single-Process Correctness

Generated: 2026-07-26 · Branch `param-main` · Foundation `codex/dsctm/`
Evidence: `artifacts/gate1/gate1_tests.{xml,log}`

**Result: 59 tests passed, 0 failures, 0 errors** (was 31 at Gate 0 — 28 added).

---

## 1. Why this gate exists

DDP replicates whatever the single-process pipeline does, including its mistakes. A padding bug
or an imputation leak becomes *harder* to see once it is spread across ranks, not easier. Gate 1
pins every single-process property that Gates 2–10 will assume, so that any later discrepancy is
attributable to the distributed layer rather than to the science layer.

---

## 2. Coverage matrix

| Gate 1 requirement | Test | Pre-existing / added |
|---|---|---|
| **DAIC padding** | | |
| Normalization ignores padded timesteps | `test_train_mean_normalizer_ignores_missing_and_padding`, `test_normalizer_ignores_padding` | pre-existing + **added** |
| Pooled representations ignore padded timesteps | `test_padding_mask_makes_head_invariant_to_masked_tail` | pre-existing |
| Altering padded values does not change logits | `test_padding_mask_makes_head_invariant_to_masked_tail`, `test_zero_future_leakage_end_to_end` | pre-existing + **added** |
| LSTM uses packed sequences | `test_lstm_baseline_uses_packed_sequences` (+ negative control) | **added** |
| Transformer uses a padding mask | `test_transformer_baseline_applies_padding_mask` | **added** |
| True lengths validated and propagated | `test_contract_rejects_out_of_range_lengths`, `test_contract_rejects_mismatched_lengths_count`, `test_lengths_default_to_full_T_when_absent`, `test_lengths_propagate_into_loader_mask_and_zero_the_tail` | **added** |
| **StudentLife imputation** | | |
| Leading missing never filled from the future | `test_forward_fill_never_uses_a_future_observation` | pre-existing |
| Forward-fill uses only observations at or before `t` | same | pre-existing |
| All-missing channel has a documented fallback | `test_forward_fill_all_missing_is_zero` | pre-existing |
| Preprocessing semantics versioned and hashed | `test_early_v2_cache_filename_recovers_unambiguous_version`, `test_ambiguous_metadata_free_cache_remains_legacy` | pre-existing |
| **Dataset splits** | | |
| Participants disjoint across train/dev/test | `test_subject_grouped_folds_have_no_leakage`, `test_official_edaic_split_files_are_disjoint_and_correctly_sized` | pre-existing + **added** |
| Official split used, no dev+test merge | `test_official_edaic_split_files_are_disjoint_and_correctly_sized`, `test_shipped_split_is_not_the_manuscript_107_82_partition` | **added** |
| Grouped splitting valid for multiclass | `test_grouped_stratification_does_not_reduce_multiclass_subjects_to_mean_label` | pre-existing |
| Duplicate-content / overlapping-window check | `test_no_duplicate_window_content_across_grouped_folds` (+ positive control) | **added** |
| Class weights from training data only | `test_class_weights_use_training_labels_only`, `test_class_weights_are_independent_of_validation_labels`, `test_unweighted_loss_is_the_default`, `test_empty_class_does_not_divide_by_zero` | **added** |
| **Architecture** | | |
| Strict causal convolution, zero future leakage | `test_causality_and_reproducibility`, `test_zero_future_leakage_end_to_end` | pre-existing + **added** |
| RF = 61 / 481 / 1921, derived not typed | `test_receptive_fields_are_61_481_1921_derived_from_implementation` | **added** |
| Per-subject adapter cost `d_s`, not `2D` | `test_per_subject_adapter_cost_is_d_s_not_2D`, `test_full_model_per_subject_growth_is_d_s` | **added** |
| Logits unchanged when only masked padding is perturbed | `test_padding_mask_makes_head_invariant_to_masked_tail` | pre-existing |

Every added test that asserts an *invariance* is paired with a **negative control** proving the
invariance is caused by the mechanism under test and not by coincidence:

- `test_lstm_baseline_without_mask_is_affected_by_padding` — confirms the unmasked LSTM *is*
  corrupted by padding, so the masked pass proves something.
- `test_duplicate_window_detector_actually_fires` — plants a cross-subject duplicate window and
  confirms the detector catches it.

---

## 3. Changes made to the implementation

Three changes, all additive or warning-only. No numerical default changed.

### 3.1 `models/blocks.py::CSAG` — variant selection (DECISIONS.md D-006)

`CSAG.__init__` gains `nonlinearity: str | None = None`. When `None` (**the default**) the
forward pass is byte-for-byte the previous computation: `A = W_a(W_z(cat))`. When set to
`relu`/`gelu`/`tanh`, the activation is applied to `Z` before `W_α`.

A property `CSAG.is_manuscript_faithful` returns `True` iff `nonlinearity is None`, so downstream
reporting can state which variant produced a number.

Verified: `test_linear_csag_alias_is_numerically_identical_to_default` asserts `atol=0, rtol=0`
equality between `csag_mode="attention"` and `csag_mode="linear_csag"`.

### 3.2 `models/dmstcn.py` — `CSAG_MODES` and `csag_nonlinearity`

```python
CSAG_MODES = ("attention", "linear_csag", "nonlinear_csag", "mean", "static")
```

`csag_mode` default remains `"attention"`. `"linear_csag"` is an explicit alias for it.
`"nonlinear_csag"` is a declared deviation and requires `csag_nonlinearity` (default `"relu"`,
validated). Unknown modes and unknown activations both raise `ValueError`.

**The manuscript-faithful gate was not replaced, renamed or altered.** Any config that does not
name a mode continues to build exactly the gate the paper describes.

### 3.3 `experiments/gate0.py` — G1-W1 fixed

The determinism probe in `exp_0_4_causality` ran two forward passes outside `torch.no_grad()`,
producing `UserWarning: Converting a tensor with requires_grad=True to a scalar`. Wrapped in
`torch.no_grad()`. The measured quantity is unchanged; the run log is now clean. Confirmed absent
from `gate1_tests.log`.

---

## 4. Findings

### F1-1 — the shipped split files are **E-DAIC**, not DAIC-WOZ (corrects a Gate 0 statement)

`reviewer-package/data/{train,dev,test}_split.csv` were described at Gate 0 as "DAIC-WOZ official
split files". Direct inspection shows otherwise:

| Split | Rows | Positive rate | ID range |
|---|---:|---:|---|
| train | 163 | 22.7 % | 302–707 |
| dev | 56 | 21.4 % | 300–713 |
| test | 56 | 30.4 % | 600–718 |
| **total** | **275** | 24.0 % | — |

163 / 56 / 56 = 275 is the **E-DAIC (AVEC-2019)** partition. Classic DAIC-WOZ (AVEC-2017) is
107 / 35 / 47 = 189. The `PROVENANCE.md` in that directory confirms: *"these artifacts derive from
the USC E-DAIC distribution."*

Pairwise participant overlap is **zero** across all three splits, and dev+test do not sum to the
manuscript's merged 82. Both facts are now pinned by tests.

**Consequence.** The corpus-identity question (tracker V3-02, D0 open question 1) is now partly
answered by evidence: the only split definition present in this repository is E-DAIC. The
manuscript cites DAIC-WOZ 189 / 107-82. If the DAIC-WOZ corpus is to be used on PARAM, its
AVEC-2017 split CSVs must be obtained separately — they are **not** in this repository.

This upgrades claim `COH-EDAIC` in `artifacts/gate0/quarantined_claims.csv` from
`unverifiable_rederive_on_param` to **verified against in-repo split files** (cohort size and
class balance only; the audio and features still require staging).

### F1-2 — the Transformer padding mask takes PyTorch's nested-tensor fast path

`test_transformer_baseline_applies_padding_mask` emits:

```
torch/nn/modules/transformer.py:515: UserWarning: The PyTorch API of nested tensors is in
prototype stage ... (Triggered internally at .../NestedTensorImpl.cpp:182)
    output = torch._nested_tensor_from_mask(
```

This is **confirmation that the mask is reaching the encoder** — the fast path only engages when
`src_key_padding_mask` is present in eval mode with batch-first input. It is benign, but note that
this fast path does **not** engage during training or under DDP, so training-time and eval-time
attention take different code paths. Recorded for Gate 3: the DDP parity check must compare
eval-mode outputs against eval-mode outputs, never across modes.

### F1-3 — `test_lstm_baseline_uses_packed_sequences` needs a length ≥ 1 guarantee

`LSTMBaseline.forward` computes `lengths = mask.sum(1)` and calls `pack_padded_sequence`, which
raises on a zero-length row. The `WindowedDataset` contract enforces `lengths ∈ [1, T]`, so this
cannot occur through the supported path — and that contract is now tested
(`test_contract_rejects_out_of_range_lengths`). No code change needed; the dependency is recorded
so it is not accidentally removed.

---

## 5. Not covered by Gate 1 (deferred, with reasons)

| Item | Why deferred | Where it lands |
|---|---|---|
| Real StudentLife / DAIC cohort statistics | No dataset staged | Gate 5, on PARAM |
| FLOPs accounting | `thop` not installed | Gate 4 env, then Gate 5 |
| fp16 numerical behaviour | CPU fp16 is not representative of sm_70 | Gate 3, on PARAM |
| Anything multi-rank | No distributed layer yet | Gate 2 / Gate 3 |
| Dilation-schedule RF for the new schedules | Schedules not yet defined | Gate 6 (RF derived from implementation, never typed) |

---

## 6. Verdict

**Gate 1: PASS.**

| Condition | Result |
|---|---|
| DAIC padding semantics verified end to end | ✅ 8 tests |
| StudentLife imputation strictly causal | ✅ 5 tests |
| Split integrity, official partition, no merge | ✅ 6 tests |
| Architecture: causality, RF, adapter cost | ✅ 6 tests |
| Manuscript-faithful CSAG preserved, variant added beside it | ✅ 4 tests, `atol=0` alias equality |
| Full suite green | ✅ **59 passed / 0 failed / 0 errors** in 2.48 s |
| No architecture changed silently | ✅ default `csag_mode="attention"` unchanged |
