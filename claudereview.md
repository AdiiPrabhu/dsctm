# D-MSTCN Resubmission — Code Audit

**Candidate A (Claude) vs Candidate B (Codex)**
Audit date: 2026-07-25 · Target venue: IEEE Access resubmission · Target hardware: PARAM Utkarsh (2× NVIDIA V100 SXM2 32 GB per node, SLURM, Intel Xeon Cascade Lake)

**Inputs audited**

| Artifact | Path |
|---|---|
| Original submitted PDF | `dsctm_original.pdf` (15 pp.) |
| Master reviewer tracker | `D_MSTCN_IEEE_Access_Resubmission_Tracker - Master Tracker (1).csv` (90 rows) |
| Candidate A | `cold/` — 3,352 LOC Python, 4 test files |
| Candidate B | `code/` — 4,861 LOC Python, 15 test files |

**Test suites executed on the audit machine (CPU, `CUDA_VISIBLE_DEVICES=''`):**
Candidate A **11/11 passed** · Candidate B **31/31 passed**

---

## Two premise corrections before the findings

### 1. This is not a temporal action segmentation paper

`dsctm_original.pdf` is *"Distributed Temporal Neural Architectures for Scalable Cognitive Modeling"* (D-MSTCN) — windowed **sequence-level classification** on StudentLife (3-class stress), DAIC-WOZ/E-DAIC (binary PHQ-8) and SEED (3-class emotion transfer). The manuscript's declared metrics (§IV-C) are macro-F1 (primary), top-1 accuracy, and AUC-ROC.

**F1@{10,25,50}, Edit score and MoF are frame-level segmentation metrics and are correctly absent from both codebases.** Auditing for them would flag a non-defect. This audit evaluates against the metrics the paper actually claims.

### 2. Neither codebase contains any distributed code

Grep across both trees for `DistributedDataParallel`, `DistributedSampler`, `init_process_group`, `all_gather`, `all_reduce`, `torchrun`, `GradScaler`, `#SBATCH`, `srun` returns **zero hits in source**. The only matches anywhere are prose mentions in markdown and one `torch.autocast` smoke test.

Section 3 of the audit brief therefore has a null answer for both candidates — the "winner" on multi-GPU readiness is a tie at zero. Both campaigns ran end-to-end on a single RTX 4060 Ti. The manuscript's 8-node A100 scaling claims (Table 3, Fig. 5–6, η ≥ 0.81, "57% lower per-epoch time at 8 nodes") are backed by **no code in either candidate**.

---

## 1. Reviewer requirement mapping (code-affecting items only)

| ID | Reviewer | Requirement | A (Claude) | B (Codex) | Winner |
|---|---|---|---|---|---|
| T2-02 | R2 | Exact per-branch receptive field | ✅ EXP-0.1, gradient-support measurement | ✅ + independent perturbation probe | **B** |
| T2-03 | R2 | Correct adapter parameter count | ✅ d_s = 8, not 2D | ✅ identical | Tie |
| T2-12 | R2, R3 | Unit/integration tests for TCP, SAP, RF, params | ⚠️ 11 tests, 4 files | ✅ **31 tests, 15 files** | **B** |
| V3-01 | R3, R4 | StudentLife participant-leakage audit | ⚠️ `StratifiedKFold` on `round(mean(y))` — invalid for 3-class | ✅ `StratifiedGroupKFold` on window labels grouped by subject | **B** |
| V3-02 | R6 | DAIC-WOZ 107/82 split correction | ✅ official 107/34/47, no dev+test merge | ✅ identical | Tie |
| V3-03 | R4, R3 | Unseen-subject embedding handling | ✅ index-0 unknown row + embedding dropout | ✅ + `global` / `global_matched` FiLM controls | **B** |
| **V3-05** | **R3** | **Forward-fill / imputation temporal leakage** | ❌ **`_ffill` back-fills the leading NaN prefix from a future observation** | ✅ **found it, fixed it, quarantined the invalid run, re-ran** | **B** |
| V3-06 | R3 | Overlapping-window / duplicate-content leakage | ✅ index + subject disjointness assertions | ✅ identical | Tie |
| V3-07 | R3 | Class distributions / imbalance handling | ✅ class-balanced CE, train-only weights | ✅ identical + PR-AUC reported | **B** |
| V3-08 | R3, R6 | Seeds, determinism, repeated-run protocol | ✅ `repro.py` (file identical to B) | ✅ + immutable per-fit environment capture | **B** |
| **E4-01** | **R2, R6** | **Baseline fairness audit** | ❌ TimesNet is a self-declared placeholder; **no equal-budget tuning** | ✅ **faithful THUML TimesNet pinned @ `4e938a1`** + `fair_tuning.py` (8 dev trials/model, test inaccessible during search) | **B** |
| E4-02 | R1, R2, R6 | Controlled DDP causality experiment | ❌ none | ⚠️ `delay_task.py` (synthetic short/medium/long XOR) — a proxy, not DDP | **B** (partial) |
| E4-03 | R2, R3, R6 | True multi-node experiments | ❌ none | ❌ none | **Neither** |
| E4-04 | R2, R3 | Synthetic/replicated larger-scale workloads | ⚠️ `synthetic.py` only | ⚠️ `synthetic.py` + delay generator | **B** (partial) |
| E4-05 | R2, R6 | Bandwidth / RTT / jitter / straggler benchmarks | ❌ none | ❌ none | **Neither** |
| E4-06 | R6 | Repeated throughput / efficiency measurements | ❌ none | ⚠️ single-device only, 30 synced repeats | **B** (partial) |
| E4-07 | R3 | Params, FLOPs, peak GPU memory, inference latency | ⚠️ params + FLOPs only | ✅ `run_single_device_profile.py`: synced median/p95/p99, throughput, peak MiB | **B** |
| E4-08 | R3 | Alternative dilation schedules ablation | ❌ none | ❌ none | **Neither** |
| E4-09 | R3 | Branch-count / kernel-size ablation | ⚠️ 3 branch-removal variants | ✅ all single-, pair- and full-branch combinations | **B** |
| E4-10 | R3 | CSAG vs alternative fusion | ⚠️ fixed-`mean` only | ✅ `mean` + learned-`static` + ½× / 2× attention temperature | **B** |
| E4-11 | R3 | FiLM vs lightweight personalization alternatives | ⚠️ on/off only | ✅ + `global` + **parameter-count-matched** global | **B** |
| E4-12 | R3, R4 | Cold-start / unseen-subject personalization | ✅ trained neutral row 0 | ✅ + explicitly documented in the mathematical record | **B** |
| E4-13 | R2, R3 | Redesigned significance-testing plan | ✅ bootstrap CI, Hodges–Lehmann, rank-biserial, Wilcoxon-reachability guard | ✅ same + **Holm and Benjamini–Hochberg multiplicity correction** | **B** |
| E4-14 | R2, R3 | Sufficient seeds for headline comparisons | ✅ 5 seeds + participant bootstrap | ✅ identical | Tie |
| E4-15 | R3, R6 | 95% CIs and effect sizes | ✅ present | ✅ present + multiplicity-adjusted | **B** |
| E4-16 | R1, R2 | δ_max / T_sync sensitivity analysis | ❌ invariants only, no sweep | ❌ invariants only, no sweep | **Neither** |
| E4-17 | R2, R6 | Instrumented communication-volume validation | ❌ none | ❌ none | **Neither** |
| E4-18 | R3 | Anonymized reproducibility package | ⚠️ configs + split/data hashes | ✅ immutable run registry + **fail-closed result auditor** + SHA-256 receipts | **B** |

**Score: B wins or ties every code-affecting item. A wins none.**

Six items are unaddressed in **both** candidates: E4-03 (multi-node), E4-05 (network robustness), E4-08 (dilation-schedule ablation), E4-16 (δ_max/T_sync sweep), E4-17 (communication instrumentation), and D1-03 (the core "DDP violates temporal causality" premise).

> **Note on `reviewer_to_experiment_map.csv`.** Both repos ship this file, but it is a stale Gate-P preflight product generated *before* implementation — A marks 67/90 rows "blocked", B marks 89/90. Neither reflects actual code state. The live ledgers are A's `HANDOFF.md` and B's `STATUS.md` / `METRICS.md`. The table above is built from the source code, not from these CSVs.

---

## 2. Model architecture & mathematical fidelity

### 2.1 Shared, and correct in both

`models/blocks.py` implements the manuscript equations faithfully:

| Eq. | Manuscript | Implementation | Status |
|---|---|---|---|
| (1) | `X' = W_in X + b_in`, `X' ∈ R^{T×D}`, D = 128 | `nn.Linear(input_dim, D)` | ✅ |
| (2) | `H_b^ℓ = H_b^{ℓ-1} + Conv_r(GELU(Conv_r(LN(H_b^{ℓ-1}))))` | `DilatedResidualBlock` — LN over channel dim, two causal convs at the same dilation, GELU between, identity residual | ✅ |
| — | Strict causality | `CausalConv1d` left-pads by `(K-1)·dilation`; asserted at **exactly 0** future leakage | ✅ |
| (3) | `Z = W_z·[H_s;H_m;H_l] + b_z`, `Z ∈ R^{T×3D}` | `nn.Linear(3D, 3D)` | ✅ |
| (4) | `A = W_α Z + b_α`, `A ∈ R^{T×3}` | `nn.Linear(3D, 3)` | ✅ |
| (5) | `α = softmax(A/√D)` | `softmax(A / sqrt(D), dim=-1)` over the branch axis | ✅ |
| (6) | `H = Σ_b α_{:,b} ⊙ H_b` | `(alpha.unsqueeze(-1) * stack).sum(dim=2)`, `(B,T,n,D)` broadcast correct | ✅ |
| (7)(8) | `γ(e_s) = W_γ ReLU(W_1 e_s + b_1) + b_γ`, β likewise | Shared generator MLP, identity init (γ≈1, β≈0) | ✅ |
| (9) | `H' = γ(e_s) ⊙ H + β(e_s)` | `gamma * H + beta` with `(B,1,D)` broadcast | ✅ |
| (10) | `ŷ = softmax(W_2 ReLU(W_1 H̄' + b_1) + b_2)` | Returns logits; softmax folded into `CrossEntropyLoss` | ✅ (numerically preferable) |
| (12) | `Ĝ_b = G_b ⊙ M`, `M_t = 1[t ≤ t_current]` | `causal_gradient_mask` | ✅ |

`train/tcp.py`, `eval/statistics.py`, `repro.py`, `config.py` and `configs/default.yaml` are **byte-identical** between the two candidates. The TCP staleness controller (Δ increment, HOLD trigger, HOLD-over-periodic precedence, Δ ≤ δ_max invariant) is the same simulation in both.

### 2.2 Manuscript errors both candidates correctly detect

**Receptive field (T2-02).** Measured RF by input-gradient support is **61 / 481 / 1921** for SSB/MSB/LSB, exactly matching `R = 1 + 2(K−1)·Σr` for the two-conv residual block of Eq. (2). The manuscript's printed **47 / 383 / 1535** match neither the one-conv formula (31/241/961) nor the two-conv formula.

For the correction note: the printed values follow `6·r_max − 1` (6·8−1 = 47, 6·64−1 = 383, 6·256−1 = 1535), which corresponds to **no standard dilated-TCN derivation**. Treat them as a formula error, not evidence of a different block design. The error propagates to Fig. 1 branch labels, Fig. 2, and the §III-F complexity constant `O(1536·T)` (1536 = 6·256).

**Adapter accounting (T2-03).** Per-subject *stored* cost is `d_s = 8` — one embedding row. The γ,β vectors (2D = 256) are **generated** by a shared MLP and are activations, not per-subject stored parameters. The manuscript's "the adapter adds 2D parameters per subject" is wrong by a factor of 32. Total model ≈ 1.36 M params (StudentLife config).

### 2.3 Manuscript weakness neither candidate flags

Eq. (3)–(4) compose two affine maps with **no intervening nonlinearity**, so `W_α W_z` collapses algebraically to a single linear map `R^{3D} → R^3`. The "learned Cross-Scale Attention Gate" is, mathematically, one linear layer followed by a softmax. Both implementations are faithful to the paper — the redundancy is in the paper. Worth pre-empting, since R3 attacked the novelty claim directly.

### 2.4 Silent bugs in Candidate A (all absent from B)

#### A-1 · Padding contaminates every DAIC-WOZ number

`data/daic.py` right-zero-pads sessions to `T_MAX = 2000` and records `true_len` per session. A then stores it as `ds._lengths` — an underscore attribute read **only** by `experiments/gate1.py` for a min/median/max provenance report. It never reaches training. Verified: the only three references to `_lengths` outside the loader are gate1 reporting lines.

Consequences, all in A:

- **`fit_normalizer`** (`train/trainer.py:20`) computes μ/σ over the full padded array. Padding zeros pull the mean toward 0 and shrink σ, by a factor that varies with each participant's session length.
- **`Head.forward`** (`models/blocks.py:143`) does `H.mean(dim=1)` over all 2000 steps. A 400-frame session has its pooled representation divided by 5; a 2000-frame session by 1. **Session duration is injected directly into the pooled feature as a per-participant scale factor** — a length shortcut in a clinical-severity task where interview duration is not independent of the label. Recorded session lengths: min 830, median 1821, max capped at 2000.
- **`LSTMBaseline`** runs unpacked across padding; **`TransformerBaseline`** attends over padding with no `src_key_padding_mask`.

Candidate B threads `lengths` through `WindowedDataset` (with `[1, T]` range validation), masks the normalizer (`np.nanmean` over valid positions only), zeroes the padded tail, packs the LSTM via `pack_padded_sequence`, passes `src_key_padding_mask` to the Transformer, and does masked mean-pooling in `Head`. It proves it with `test_padding_mask_makes_head_invariant_to_masked_tail`, which perturbs the masked tail by 100× and asserts the logits are unchanged.

**Measured impact.** B's mask-aware DAIC-WOZ rerun moved D-MSTCN from A's 2nd/6 (0.4854) to 3rd/6 (0.4818) and eliminated A's *only* CI-clears-zero result in the entire study (which had been against the simplified TimesNet placeholder — itself already flagged for replacement).

#### A-2 · Backward-fill leakage in `_ffill`

`data/studentlife.py:111`:

```python
filled[np.isnan(filled)] = filled[valid][0]
```

This fills the leading NaN prefix with the **first later valid observation**. In a paper whose entire thesis (P2, Theorem 1, TCP, the causal mask of Eq. 12) is temporal causal ordering, the data loader violates causality. StudentLife sensor missingness is ≈ 0.61 at 1-minute resolution, so leading prefixes are common, not rare.

This is precisely reviewer R3's tracker item **V3-05** (P0 Critical, "Audit forward-fill and imputation for temporal leakage"). A's own `OBSERVATIONS.md` describes the data as "forward-filled" and the pipeline as leakage-safe.

B's replacement comment names it explicitly: *"The previous implementation replaced a leading missing prefix with the first later observation, which was backward-fill leakage despite the function name."* B cancelled its in-flight run at 16 minutes on discovering this, quarantined the affected cache (hash `62de62987570bc40`), rebuilt as `studentlife-v2-causal_ffill` (hash `a9cbaa3a22c2bf4e`), and re-ran.

#### A-3 · Invalid stratification

`data/splits.py::subject_grouped_kfold` reduces each participant to `int(round(np.mean(y[subject_id == s])))` and feeds that to `StratifiedKFold`. On StudentLife the label is **3-class** {0,1,2} with counts 578/973/609 — rounding a mean of class *indices* does not produce a class.

B replaced this with `StratifiedGroupKFold` on window labels grouped by participant, with the inline note that the old approach *"is invalid for multiclass labels and produced severely unbalanced folds on StudentLife."* Both candidates retain the `round(mean)` form inside `stratified_holdout_by_subject`, which is intentional — that function exists to reproduce the manuscript's original 80/20 protocol.

#### A-4 · Test-set re-evaluation (minor)

A's `train_select_evaluate` calls `evaluate(..., te, ...)` at **every** dev improvement. Statistically harmless — selection is dev-only and test never drives early-stopping — but it repeatedly touches the held-out set and wastes compute. B checkpoints the best-dev `state_dict`, restores it after the loop, and evaluates test exactly once.

---

## 3. PARAM Utkarsh multi-GPU readiness

### 3.1 Audit result

| Requirement | A (Claude) | B (Codex) |
|---|---|---|
| `torch.nn.parallel.DistributedDataParallel` | ❌ absent | ❌ absent |
| `DistributedSampler` (+ `shuffle=False` for validation) | ❌ absent — plain `DataLoader(TensorDataset)` | ❌ absent |
| `torchrun` / `init_process_group` / NCCL backend | ❌ absent | ❌ absent |
| Rank-deadlock hazards | n/a — no ranks | n/a — no ranks |
| `torch.cuda.amp.autocast` in the training loop | ❌ absent | ❌ absent |
| `GradScaler` | ❌ absent | ❌ absent |
| AMP validated at all | ❌ nothing | ⚠️ one fp16 `autocast` finite-loss/finite-grad smoke test, `experiments/gate0.py:242`, CUDA-only |
| `torch.distributed.all_gather` metric aggregation | ❌ absent | ❌ absent |
| SLURM `sbatch` / `srun` launch scripts | ❌ absent | ❌ absent |
| Measured multi-GPU evidence | ❌ none (1× RTX 4060 Ti) | ⚠️ single-device FP32 profile only |

Both are pure single-device, single-process. Rank deadlocks and skewed evaluation logs are not present because ranks are not present. Neither codebase is a "port" — both require a **from-scratch DDP implementation** on top of an otherwise sound single-GPU trainer.

### 3.2 Three landmines for whoever writes that layer

1. **`nn.LazyLinear` in `ITransformerBaseline`** (present in both candidates). DDP requires all parameters materialized at wrap time; wrapping before a dry forward pass raises at construction. Run one dummy batch pre-`DDP()`, or replace with a fixed `nn.Linear(T, d_model)`.

2. **V100 is sm_70 — fp16 only, no bf16.** B's smoke test already uses `torch.float16`, which is the correct choice. `GradScaler` is mandatory, not optional, at fp16.

3. **DAIC-WOZ is 274 sessions and the test set is 47.** At 2 ranks with `DistributedSampler` and `batch_size=8`, per-rank batches are small and the sampler pads the final batch by **repeating samples**. Without `all_gather` of predictions *and* subject IDs followed by de-duplication, participants will be silently double-counted in evaluation. This is exactly the "skewed evaluation log" failure mode, and it will bite on the fixed official test set.

---

## 4. Verdict and recommendation

### Winner: Candidate B (Codex)

Decisively, and without a close call. B is a strict superset of A — same shared core, plus every fix. Specifically, B:

- found and repaired two P0 correctness defects in A (padding contamination of all DAIC-WOZ results; backward-fill causal leakage in StudentLife imputation);
- replaced A's self-declared placeholder TimesNet with a faithful port of the official THUML implementation, pinned to commit `4e938a1`;
- implemented the equal-budget fair-tuning protocol (E4-01) that A left as an open TODO — 8 dev trials per model, test provably inaccessible during search, configs frozen before 5-seed confirmation;
- added Holm and Benjamini–Hochberg multiplicity correction across the prespecified comparison family;
- added an immutable per-fold/per-seed run registry that also records *failed* trials as `model_failed` rather than silently dropping them;
- added a fail-closed result auditor (`audit_exp41_corrected.py`) that rejects an incomplete model family, wrong seeds/hashes, non-finite or out-of-range metrics, invalid CIs, or a fold mean that does not recompute — and emits a SHA-256 receipt;
- fixed the invalid multiclass stratification;
- carries 31 passing tests against A's 11.

B also quarantined its own invalid run rather than shipping it — the behaviour you want in a resubmission audit trail.

### Modules in B requiring work before PARAM Utkarsh

| Module | Required change |
|---|---|
| `train/trainer.py` | Write the DDP layer: `init_process_group("nccl")`, `DistributedSampler(shuffle=True)` for train / `shuffle=False` for val + test, `sampler.set_epoch(epoch)` each epoch, wrap in `DDP(model, device_ids=[local_rank])`. Add `autocast(dtype=torch.float16)` + `GradScaler` around the existing loss / backward / step. |
| `train/trainer.py::evaluate` | `all_gather` probabilities, labels **and subject IDs**; de-duplicate on subject ID before calling `classification_metrics`, or sampler tail-padding will inflate the 47-session DAIC-WOZ test set. |
| `train/trainer.py` early-stop | The `patience` / `best_dev` decision must be made on rank 0 and **broadcast**. Divergent per-rank break conditions are the classic collective deadlock in this pattern. |
| `models/baselines.py` | Materialize `ITransformerBaseline`'s `LazyLinear` with a dummy forward before the DDP wrap. |
| `experiments/*.py` | Guard every `write_completed_fit`, `Path(...).write_text` and `log()` with `if rank == 0`. Two ranks writing the same immutable run directory will corrupt the registry. |
| `scripts/*.sbatch` (new) | SLURM wrapper: `--gres=gpu:2 --ntasks-per-node=2`, `MASTER_ADDR` derived from `scontrol show hostnames`, `torchrun --nnodes=$SLURM_NNODES --nproc_per_node=2`. |
| `train/tcp.py` | Still a single-process simulation of Algorithm 1. Real TCP — per-branch staleness tracked across ranks, HOLD-triggered `all_reduce`, optimizer-state and partial-failure semantics — is unwritten in both candidates. B's own `STATUS.md` lists this as an open correctness gap. |

### The uncomfortable part — this is a scope decision, not a code decision

Both independent campaigns, across three corpora (StudentLife, E-DAIC, DAIC-WOZ) and two feature sets (23-dim LLD, 88-dim eGeMAPS functionals), found that **D-MSTCN does not win**:

| Run | D-MSTCN result | Rank | Inference |
|---|---|---|---|
| StudentLife, corrected leakage-safe (B, audit-passed) | 0.3428 ± 0.0067 | **4/6** | Beaten by transformer (0.3675), itransformer (0.3612), timesnet (0.3493). No pair statistically resolvable. |
| DAIC-WOZ 88-dim, mask-aware (B) | 0.4818 | **3/6** | Every paired participant-bootstrap 95% CI spans zero. |
| E-DAIC 23-dim (A) | 0.5529 ± 0.0918 | 2/6 | All CIs span zero. |
| E-DAIC 88-dim (A) | 0.5222 | 3/6 | All CIs span zero. |

Two further structural problems, independent of the numbers:

- With 5 folds or 5 seeds, a two-sided exact Wilcoxon signed-rank test **cannot reach p < 0.05** (minimum achievable p = 0.0625). Every "† p < 0.05" marker in Table 2 is unreachable as printed, regardless of what the data says. Both candidates encode this as an explicit guard.
- The systems half of the paper (Table 3, Fig. 5–6, Theorem 1's operational content, the SAP placement algorithm) rests on distributed code that has never existed in either candidate.

Building the DDP layer on PARAM Utkarsh gets you E4-03, E4-05, E4-06, E4-17 and the D1-03 causality premise — the systems half. **It will not produce the accuracy headline.** Make the reframing decision (tracker D1-01 / D1-05 / W5-01: reposition around causality, efficiency and the methodology corrections, versus defending the original accuracy claim) *before* spending V100 hours. Otherwise you will have rigorous scaling evidence attached to a claim the data does not support.

---

## Appendix — file-level inventory

### Modules byte-identical between candidates

`train/tcp.py` · `eval/statistics.py` · `repro.py` · `config.py` · `configs/default.yaml`

### Candidate B only

| File | Purpose | Tracker item |
|---|---|---|
| `models/timesnet.py` | Faithful THUML TimesNet @ `4e938a1` | E4-01 |
| `experiments/fair_tuning.py` | 8-trial equal-budget dev search per model | E4-01 |
| `experiments/preprocessing.py` | 4 leakage-safe imputation conditions | V3-05, V3-04 |
| `experiments/delay_task.py` | Controlled short/medium/long XOR dependency | E4-02 (proxy) |
| `experiments/result_audit.py` + `scripts/audit_exp41_corrected.py` | Fail-closed result admission | E4-18 |
| `scripts/run_single_device_profile.py` | Synced latency / throughput / peak memory | E4-07 |
| `scripts/build_daicwoz_participant_egemaps88.py` | Participant-only audio (excludes Ellie) | V3-04 |
| `scripts/plot_reliability.py` | Calibration / reliability figures | E4-15 |
| `artifacts/.../MATHEMATICAL_FORMULATION.md` | Full implemented-pipeline math record | T2-11 |
| 11 additional test files | — | T2-12 |

### Ablation coverage

| | A | B |
|---|---|---|
| Variants | 7 | **14** |
| Branch controls | full, noSSB, noMSB, noLSB, 1scale_SSB | + 1scale_MSB, 1scale_LSB |
| Fusion controls | attention, mean | + learned-static, ½× temp, 2× temp |
| Personalization controls | FiLM on/off | + global, parameter-matched global |
| Paired statistics | none in ablation | Hodges–Lehmann, rank-biserial, exact Wilcoxon, Holm, BH |
| Per-variant checkpointing | ❌ | ✅ |
