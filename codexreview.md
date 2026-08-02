# DSTCM / D-MSTCN Claude vs. Codex Comparative Audit

## Comparative audit verdict

Codex is the stronger research codebase, but neither codebase is ready for multi-GPU execution on PARAM Utkarsh.

- Mathematical/single-GPU correctness: Codex wins.
- Reviewer-experiment coverage: Codex wins.
- Multi-GPU/TCP/SAP implementation: neither implements it.
- Submit-time readiness: neither is ready.
- Recommended basis: Codex, after adding a real DDP/SLURM execution layer and rerunning all experiments.

## 1. Reviewer compliance matrix

Scores: 0 = absent, 3 = implemented but incomplete/unexecuted, 5 = complete with auditable evidence.

| Tracker requests | Claude | Codex | Audit finding |
|---|---:|---:|---|
| G0-01–05 references and claim/citation audit | 1 | 1 | Planning artifacts exist, but no completed authoritative 38-reference audit or corrected bibliography. |
| G0-06–07 result provenance and 68.7 verification | 1 | 3 | Codex has a stronger run registry/auditor, but supplied experiment artifacts are absent and the manuscript values are not reproduced. |
| G0-08–10 disclosure, immutable archive, sign-off | 0 | 1 | Cannot be completed through code; no signed integrity gate or final immutable evidence archive. |
| D1-01–02 distributed scope and dataset-scale alignment | 2 | 3 | Both correctly flag the mismatch. Neither provides physical multi-node evidence. |
| D1-03, E4-02 formal DDP causality test | 1 | 2 | Codex adds a synthetic delay task, but neither isolates a real DDP causality failure. |
| D1-04, T2-09–10 theorem/proof correction | 2 | 3 | Both recognize the theorem is unsupported. No valid bounded-bias convergence proof exists. |
| T2-01 η notation | 1 | 1 | Documentation concern remains outside executable evidence. |
| T2-02 receptive-field derivation/test | 5 | 5 | Both correctly implement and test the two-convolution receptive field. |
| T2-03 FiLM parameter count | 5 | 5 | Both correctly identify per-subject storage as `d_s`, not `2D`. |
| T2-04–05 TCP counter and HOLD precedence | 3 | 3 | Single-process state-machine semantics are tested; no optimizer or collective implementation. |
| T2-06–07 replication and N>3/N=16 semantics | 0 | 0 | No execution graph or implementation for replicated branches. |
| T2-08 dataset-specific temporal rationale | 2 | 2 | Correct RF observations exist, but manuscript rationale is not revised. |
| T2-11 computational/communication complexity | 2 | 2 | No validated collective-volume instrumentation. |
| T2-12 unit/integration tests | 3 | 4 | Claude: 11 passing tests; Codex: 31. Neither has distributed integration tests. |
| V3-01 StudentLife participant leakage | 4 | 5 | Both have subject-grouped splitting; Codex has stronger result auditing and data hashes. |
| V3-02 official DAIC-WOZ split | 4 | 4 | Official train/dev/test handling is implemented; data provenance still requires confirmation. |
| V3-03, E4-12 unseen-subject personalization | 3 | 4 | Both reserve an unknown-subject embedding; Codex adds global/matched adapter controls. No completed cold-start adaptation experiment. |
| V3-04 preprocessing documentation | 3 | 4 | Codex covers more preprocessing variants and records semantic data versions. |
| V3-05 imputation leakage | 2 | 5 | Claude retains the older pipeline. Codex implements strictly causal forward-fill and regression tests. |
| V3-06 overlapping-window audit | 2 | 2 | Subject separation exists, but no complete duplicate-content audit evidence. |
| V3-07 class imbalance and per-class metrics | 4 | 5 | Codex supports train-only balanced CE plus per-class precision/recall/F1. |
| V3-08 seeds/determinism | 4 | 4 | Both seed Python, NumPy, Torch and CUDA; no rank-synchronized distributed seeding. |
| V3-09 count reconciliation | 3 | 4 | Codex documents 46 StudentLife subjects and DAIC discrepancies more explicitly. |
| E4-01 baseline fairness | 2 | 4 | Codex adds equal-trial, development-only tuning and a faithful TimesNet baseline. Runs are incomplete. |
| E4-03 true multi-node experiments | 0 | 0 | Completely absent. |
| E4-04 large synthetic workload | 2 | 3 | Synthetic datasets exist, but no distributed scaling workload/results. |
| E4-05–06 network and repeated throughput benchmarks | 0 | 0 | No NCCL/interconnect experiments. |
| E4-07 parameters/FLOPs/memory/latency | 1 | 3 | Codex has a single-device profiler, not a full multi-GPU systems profile. |
| E4-08–11 architecture ablations | 2 | 4 | Codex has broad branch/fusion/FiLM runners; Claude has narrower component ablation. Results are not supplied. |
| E4-13, E4-15 statistical redesign | 4 | 5 | Codex includes participant/fold inference, bootstrap CIs, effect sizes and multiplicity correction. |
| E4-14 sufficient headline reruns | 1 | 3 | Codex documents reruns, but the referenced raw result tree is absent from this package. |
| E4-16 TCP sensitivity | 1 | 1 | No real TCP implementation, hence no valid δ/Tsync systems sensitivity result. |
| E4-17 communication instrumentation | 0 | 0 | Absent. |
| E4-18 reproducibility package | 2 | 4 | Codex has a much stronger registry and result auditor, but no final complete archive. |
| W5/F6/S7 manuscript, response and submission files | 0 | 1 | Mostly non-code tasks; final revised manuscript, response, highlighted PDF and approvals are absent. |

Approximate code-addressable compliance:

- Claude: **2.3/5**
- Codex: **3.2/5**
- Codex excluding unexecuted runners and status claims: approximately **2.7/5**

## 2. Mathematical and algorithmic correctness

### Receptive field

Both implementations follow Equation (2), with two causal convolutions per residual block:

```text
R = 1 + 2(K - 1) Σ r_l
```

For `K = 3`:

- SSB: `1 + 4(1 + 2 + 4 + 8) = 61`
- MSB: `1 + 4(8 + 16 + 32 + 64) = 481`
- LSB: `1 + 4(32 + 64 + 128 + 256) = 1921`

The manuscript’s 47/383/1535 values are incorrect and do not follow either the one- or two-convolution formula. The implementation is in `dsctm/code/dsctm/src/dsctm/models/blocks.py`.

### Causal convolutions

Left-only padding is correct, and the causal unit tests pass. No future input is used by the convolutional branches.

### Temporal padding and masking

This is a decisive Codex advantage.

- Claude normalizes and mean-pools padded timesteps as if they were valid.
- Codex excludes padding from normalization, zeros padded data, carries a validity mask and performs masked pooling.

Claude can therefore produce length-dependent features and biased normalization for variable-length DAIC sessions.

### Imputation leakage

Codex fixes a serious prior error: leading missing values are no longer filled from a future observation. Claude lacks the corresponding regression coverage.

### Loss

Codex correctly computes class weights using training labels only and passes them to `CrossEntropyLoss`.

No improper single-GPU loss scaling was found. Distributed loss scaling cannot be evaluated because DDP is absent.

### Personalization

Both avoid direct test-subject identity leakage by mapping unseen subjects to row zero and training that row through embedding dropout. Codex additionally supports:

- subject FiLM;
- global FiLM;
- parameter-matched global FiLM;
- no-FiLM ablation.

This is better aligned with V3-03/E4-12.

### Test-set access

Claude evaluates the test set every time development performance improves. Although the test result does not drive selection, repeated access is unnecessarily risky.

Codex snapshots the best development checkpoint and evaluates test exactly once afterward.

### TCP and the causality claim

The supplied TCP is not a training protocol. It is explicitly a single-process counter simulation.

Its `_allreduce()` only increments a Python version counter and resets dictionaries. It does not:

- invoke `dist.all_reduce`;
- synchronize parameters or optimizer state;
- communicate branch activations;
- implement SAP;
- apply the causal mask during training;
- handle branch replicas.

Moreover, masking “future” activation gradients does not establish that ordinary DDP violates temporal causality. For stateless complete samples, averaging gradients from different temporal windows is standard empirical-risk optimization. The manuscript’s general DDP-causality claim and convergence theorem must be removed or substantially narrowed unless a specific stateful/asynchronous mechanism is formally established.

## 3. Metrics

Both provide classification metrics including accuracy, balanced accuracy, macro/per-class F1, AUC, PR-AUC, Brier score and ECE.

Neither implements MoF or segmental Edit score.

However, MoF/Edit are action-segmentation metrics and do not naturally apply to the manuscript’s window/session-level stress and depression classification tasks. They should only be added if a reviewer explicitly requires framewise temporal segmentation. Otherwise, macro-F1, balanced accuracy, PR-AUC and calibration are the appropriate metrics.

## 4. PARAM Utkarsh readiness

| Requirement | Claude | Codex |
|---|---|---|
| Device-neutral `.to(device)` | Yes | Yes |
| Hardcoded `.cuda()` calls | None found | None found |
| DDP/process-group setup | No | No |
| Rank-local CUDA device | No | No |
| `DistributedSampler` | No | No |
| Sampler `set_epoch()` | No | No |
| `dist.barrier()` | No | No |
| Rank-synchronized seeding | No | No |
| `pin_memory=True` | No | No |
| Tuned `num_workers` | No | No |
| `persistent_workers`/prefetch | No | No |
| AMP/GradScaler training | No production implementation | No production implementation |
| Rank-zero-only artifact writes | No | No |
| Cross-rank metric gathering | No | No |
| SLURM `sbatch`/`srun` launcher | No | No |
| NCCL error/timeout configuration | No | No |
| Resume-capable distributed checkpoint | No | No |
| MoF/Edit/F1 structured rank-safe logging | No | No |

Both currently use an in-memory `TensorDataset` and single-process `DataLoader`. On PARAM, every rank would independently shuffle the complete dataset and concurrently write artifacts, producing duplicate training samples and corrupted/competing logs.

A two-V100 node also cannot directly place all three temporal branches one-per-GPU. The first valid deployment should therefore use ordinary two-rank DDP over the complete model. A genuine branch-parallel SAP/TCP design would require explicit placement, activation transfer and autograd-aware collectives and should be treated as a separate research implementation.

## 5. Test results

- Claude: **11 passed**, one warning.
- Codex: **31 passed**, one warning.
- Plain `pytest`: fails collection in both unless installed or run with `PYTHONPATH=src`.
- No multi-GPU tests were possible because no distributed implementation exists.

Passing tests demonstrate single-process behavior only; they do not validate scalability or TCP claims.

## Final recommendation

Use the Codex codebase as the sole foundation. Do not combine Claude’s model or trainer into it; doing so would reintroduce padding and evaluation weaknesses.

Current classification:

- Claude: not suitable for final experiments.
- Codex: strongest single-GPU research prototype.
- Codex on PARAM today: not executable as a correct multi-GPU campaign.
- Paper in current form: not ready for resubmission because the distributed claims remain unsupported and the corrected experiments do not substantiate the reported headline advantage.

Before submit-time execution, Codex needs:

1. A real `torchrun`/SLURM DDP trainer with `DistributedSampler`.
2. Rank-local device initialization and rank-zero logging.
3. Cross-rank prediction gathering before metric computation.
4. `pin_memory`, configurable workers, persistent workers and nonblocking transfers.
5. Distributed checkpoint/resume and failure handling.
6. NCCL profiling and repeated two-V100 measurements.
7. Either removal of TCP/SAP/convergence claims or a genuine distributed implementation.
8. Execution of the pending fair-tuning, ablation and sensitivity runners with immutable raw artifacts.
9. Revision of the manuscript receptive fields to 61/481/1921.
10. Honest reporting that existing corrected results do not establish D-MSTCN superiority.
