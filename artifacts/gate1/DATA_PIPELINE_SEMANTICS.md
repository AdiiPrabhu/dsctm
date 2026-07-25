# Gate 1 — Data Pipeline Semantics

Authoritative description of what the `codex/dsctm/` data path does, so PARAM runs can be
audited against a written contract rather than against reader memory.

Evidence: `artifacts/gate1/gate1_tests.{xml,log}` — 59 passed.

---

## 1. `WindowedDataset` contract  (`data/contract.py`)

| Field | Type | Meaning |
|---|---|---|
| `X` | `(N, T, F)` float32 | Right-padded windows. Padding is **zeros**. |
| `y` | `(N,)` | Label; `label_type ∈ {binary, multiclass, regression}` |
| `subject_id` | `(N,)` | Grouping key. The unit of splitting — never the window. |
| `lengths` | `(N,)` int64 | **Valid timesteps before right padding.** Defaults to `T` when absent. |
| `timestamp` | `(N,)` or None | Optional; length-checked when present |
| `n_classes`, `dataset`, `version`, `sampling_interval_s`, `feature_names` | — | Provenance |

**Validated at construction:** `X.ndim == 3`; `len(X) == len(y) == len(subject_id)`;
`len(timestamp) == len(X)` when present; `len(lengths) == len(X)`; and
**`1 ≤ lengths ≤ T` elementwise**.

`data_version_hash()` is a SHA-256 over `X`, `y`, `subject_id`, truncated to 16 hex chars. It is
the `data_version` recorded in every registry entry.

Tests: `test_contract_rejects_out_of_range_lengths`, `test_contract_rejects_mismatched_lengths_count`,
`test_lengths_default_to_full_T_when_absent`.

---

## 2. Padding semantics — the rule that must never regress

**Padding must be invisible to every computation.** Concretely:

| Stage | Rule | Implementation | Test |
|---|---|---|---|
| Normalization | μ, σ from valid positions only | `fit_normalizer(X, lengths)` builds a validity mask, uses `np.nanmean` / `np.nanstd` | `test_normalizer_ignores_padding` |
| Loader | Derive mask from lengths, zero the tail *after* normalizing | `_make_loader`: `mask = arange(T) < lengths[:,None]`; `X[~mask] = 0` | `test_lengths_propagate_into_loader_mask_and_zero_the_tail` |
| Batch | Mask travels with the batch | `TensorDataset(X, y, subj, mask)` | same |
| Convolution | Causal left-pad; future cannot reach the present anyway | `CausalConv1d` | `test_zero_future_leakage_end_to_end` |
| Pooling | Masked mean, not plain mean | `Head.forward(H, mask)`: `(H*w).sum(1) / w.sum(1).clamp_min(1)` | `test_padding_mask_makes_head_invariant_to_masked_tail` |
| LSTM baseline | `pack_padded_sequence` / `pad_packed_sequence` | `LSTMBaseline.forward` | `test_lstm_baseline_uses_packed_sequences` + negative control |
| Transformer baseline | `src_key_padding_mask = ~mask` | `TransformerBaseline.forward` | `test_transformer_baseline_applies_padding_mask` |
| iTransformer baseline | `X.masked_fill(~mask, 0)` before variate embedding | `ITransformerBaseline.forward` | (covered by shape tests) |
| TimesNet baseline | `h * mask` before flatten-and-project | `OfficialTimesNetBaseline.forward` | `test_official_timesnet_shape_and_mask` |

**Why this matters numerically.** DAIC sessions are padded to `T_MAX = 2000` with true lengths
reported between 830 and 2000. An unmasked mean over 2000 steps divides a 400-step session's
representation by 5 and a 2000-step session's by 1 — injecting session duration into the pooled
feature as a per-participant scale factor. In a clinical-severity task where interview length is
not independent of the label, that is a length shortcut, not a nuisance.

---

## 3. Imputation semantics — StudentLife  (`data/studentlife.py`)

`_ffill` is **strictly causal**: no value at time `t` may be filled from an observation at
`t' > t`.

```python
last  = np.where(valid, idx, -1)
np.maximum.accumulate(last, out=last)     # index of most recent valid observation at or before t
filled = np.zeros_like(x)
seen = last >= 0                          # leading prefix has no prior observation
filled[seen] = x[last[seen]]              # ... so it stays 0, it is NOT back-filled
```

**Leading missing prefix → 0**, explicitly. The prior implementation assigned
`filled[np.isnan(filled)] = filled[valid][0]`, i.e. the *first later* observation — backward-fill
leakage in a paper whose thesis is causal temporal ordering. That defect survives only in the
archived `claude/` tree.

Available conditions (`imputation=` argument), all fitted on training data only:

| Condition | Semantics | Cache version tag |
|---|---|---|
| `causal_ffill` (default) | Last observation at or before `t`; leading prefix 0 | `studentlife-v2-causal_ffill` |
| `train_mean` | Train-fold feature mean | `studentlife-v2-train_mean` |
| `zero` | Constant 0 | `studentlife-v2-zero` |
| `mask_aware_zero` | 0 plus an explicit missingness channel | `studentlife-v2-mask_aware_zero` |

Each cache embeds its semantic version and content hash. Metadata-free early caches are recovered
only when the filename is unambiguous, otherwise flagged legacy.

Tests: `test_forward_fill_never_uses_a_future_observation`, `test_forward_fill_all_missing_is_zero`,
`test_early_v2_cache_filename_recovers_unambiguous_version`,
`test_ambiguous_metadata_free_cache_remains_legacy`.

---

## 4. Split semantics  (`data/splits.py`)

**The participant is the unit of splitting. Never the window.**

| Scheme | Function | Notes |
|---|---|---|
| Grouped k-fold | `subject_grouped_kfold` | `StratifiedGroupKFold` over **window labels** grouped by participant. Valid for multiclass. |
| Official partition | `_read_splits` / `_read_daicwoz_splits` | Three-way train/dev/test from the released CSVs. **No dev+test merge.** |
| Original-protocol holdout | `stratified_holdout_by_subject` | Retained *only* to reproduce the manuscript's 80/20 protocol for comparison. Uses `round(mean(y))` — invalid for multiclass and deliberately unchanged, because its purpose is to reproduce the original. |

Every scheme returns a manifest with `split_hash` = SHA-256 of the sorted per-fold participant
lists, truncated to 16 hex chars.

Assertions available for any fold set: `assert_no_subject_overlap`, `assert_disjoint_indices`,
`audit_folds`.

Tests: `test_subject_grouped_folds_have_no_leakage`, `test_leakage_detector_catches_overlap`,
`test_grouped_stratification_does_not_reduce_multiclass_subjects_to_mean_label`,
`test_no_duplicate_window_content_across_grouped_folds` + positive control.

---

## 5. Personalization and the unseen subject

Subject indices are built from **training participants only**:

```python
subj_map = {s: i + 1 for i, s in enumerate(sorted(train_subjects))}   # index 0 reserved
```

Index **0 is a reserved "unknown subject" row**. During training a fraction `emb_dropout = 0.1`
of samples are remapped to 0, so row 0 is *trained*. Any evaluation participant absent from
`subj_map` resolves to 0 via `subj_map.get(s, 0)`.

Consequence to state plainly in the paper: **held-out participants receive a trained neutral
embedding, not an individualized one.** There is no test-time adaptation. This is the honest
description of the cold-start condition (tracker V3-03 / E4-12).

Ablation controls: `film_mode ∈ {subject, global, global_matched}` plus `use_film=False`.
`global_matched` keeps the parameter count of the subject variant while always indexing row 0, so
the personalization effect is separated from the capacity effect.

---

## 6. Loss and class imbalance

`_build_loss(cfg, y_train, n_classes, device)`:

- `cfg["class_weight"] == "balanced"` → `w_c = n / (C · count_c)` computed from **`y_train` only**.
  Empty classes are clipped to count 1 to avoid division by zero.
- list/tuple → explicit per-class weights.
- absent or `None` → **plain unweighted cross-entropy** (the default; no silent weighting).

Selection metric is dev macro-F1 in every protocol. Weighting changes the training objective only.

Tests: `test_class_weights_use_training_labels_only`,
`test_class_weights_are_independent_of_validation_labels`, `test_unweighted_loss_is_the_default`,
`test_empty_class_does_not_divide_by_zero`.

---

## 7. Evaluation protocol

| Protocol | Function | Test-set access |
|---|---|---|
| Grouped CV (StudentLife) | `headline_cv` | No separate test set; pooled out-of-fold predictions |
| Official split (DAIC) | `train_select_evaluate` | Best-dev `state_dict` frozen, restored, test evaluated **exactly once** |
| Fair tuning | `fair_tuning.run_fair_tuning` | 8 dev trials per model; `train_model` never receives test indices |

The archived `claude/` tree re-evaluated test at every dev improvement. The Codex path does not.

---

## 8. Invariants that must hold after the distributed layer is added (Gate 2/3)

These are the Gate 1 properties that DDP must not break. Gate 3 re-asserts each one at
`world_size > 1`:

1. Normalization statistics identical regardless of rank count (fit on the train fold, not per rank).
2. Padding mask travels with every sample through the sampler and the collate path.
3. Pooled representation of a given sample is independent of which rank processed it.
4. Subject→index map is built once from training participants and is **identical on every rank**.
5. Class weights computed once from the full training fold, not per-rank shards.
6. Evaluation covers each sample **exactly once** across all ranks — no sampler tail-padding
   duplicates.
7. Split hash and data hash identical on every rank; mismatch is a hard failure.
