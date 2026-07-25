# Gate 3 — DDP Parity Report

Generated: 2026-07-26 · Branch `param-main`
Evidence: `DDP_TEST_RESULTS.xml`, `../gate2/ddp_parity_cpu_gloo.json`, `evaluation_uniqueness.json`

**Status: LOGIC-VERIFIED (CPU/gloo). Gate 3 is NOT passed.**

Gate 3's definition requires hardware validation on PARAM Utkarsh. The distributed layer is
implemented, integrated into the trainer, and proven correct over real multi-process
collectives on CPU. The NCCL / fp16 / V100 half runs when
`scripts/param/2gpu_ddp_smoke.sbatch` executes. Until then this gate stays open.

---

## 1. Test totals

| Suite | Tests | Result |
|---|---:|---|
| Full repository suite | **133** | 0 failures, 0 errors, 18.1 s |
| of which distributed units | 51 | pass |
| of which multi-process gloo | 22 | pass (world_size 2, 3, 4) |

---

## 2. Numerical parity, and why two different criteria are used

| Comparison | Criterion | Justification |
|---|---|---|
| replica vs replica | **bitwise** (SHA-256 state digest) | All-reduce delivers the identical buffer to every rank. Any difference is genuine divergence, not noise. |
| DDP vs single-process | **1e-6 absolute** | DDP forms the global gradient by all-reducing per-rank means; the reference takes one mean over the union. Float addition is not associative. |

Measured after one SGD step (CPU/gloo):

| world_size | per-rank batch | global batch | max abs Δparam | max rel Δparam | replicas identical |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 16 | 1.490e-08 | 1.069e-07 | ✅ |
| 3 | 8 | 24 | 1.490e-08 | 4.977e-07 | ✅ |
| 4 | 8 | 32 | 1.490e-08 | 1.155e-07 | ✅ |

1.490e-08 is ~2 ulp at unit scale — **67× inside** the declared tolerance. A genuine
sharding or reduction defect displaces parameters by 1e-3 or more, so the tolerance
discriminates rather than merely accommodating.

**Confound removed, not papered over.** The parity model runs with dropout disabled. A rank
holding an (8, F) shard and a reference holding the (16, F) union consume the RNG stream
differently and therefore draw different dropout masks. That divergence is correct PyTorch
behaviour and has nothing to do with DDP, so it is excluded from the parity measurement.
The checkpoint-resume test keeps dropout **on**, because that is the test whose job is to
prove RNG state is restored.

**On PARAM this same test re-runs over NCCL**, where the reduction tree differs again. The
tolerance is unchanged; the criterion still holds. Divergence beyond 1e-6 there would be a
real finding, not noise.

---

## 3. Required Gate 3 test matrix

| Required test | Status | Where |
|---|---|---|
| `test_ddp_forward_matches_single_gpu` | ✅ subsumed by the one-step test (forward is a prefix of it) | gloo |
| `test_ddp_one_step_matches_reference` | ✅ ws 2, 3 | gloo |
| `test_ddp_global_batch_semantics` | ✅ + 6 unit tests | both |
| `test_distributed_sampler_set_epoch` | ✅ ws 2, 3 | gloo |
| `test_eval_contains_no_duplicate_samples` | ✅ ws 2, 3, 4 | gloo |
| `test_eval_unique_count_matches_expected` | ✅ ws 2, 3, 4 | gloo |
| `test_rank_zero_only_writes_registry` | ✅ ws 2, 3 | gloo |
| `test_early_stop_decision_is_broadcast` | ✅ ws 2, 3 | gloo |
| `test_checkpoint_resume_matches_uninterrupted_run` | ✅ ws 2, with dropout | gloo |
| `test_fp16_loss_is_finite` | ⛔ **BLOCKED** — CPU fp16 is not representative of sm_70 | PARAM |
| `test_fp16_gradients_are_finite` | ⛔ **BLOCKED** — same | PARAM |
| `test_distributed_failure_terminates_all_ranks` | ✅ ws 2, 3 | gloo |
| `test_lazy_parameters_are_materialized` | ✅ 5 unit tests | units |

Two tests are hardware-blocked and are honestly marked so rather than being faked with a
CPU stand-in. `scripts/param/preflight.py --gpu` runs an fp16 autocast smoke check as the
first thing that happens on a real V100.

---

## 4. Trainer integration (this gate's code change)

`train/trainer.py` now accepts `ctx` and `precision`, both defaulting to the
single-process path so **the audited numerics are unchanged when `ctx is None`**.

| Change | Effect |
|---|---|
| `_build_tensor_dataset` | adds a fifth tensor: the dataset-global `sample_id`. Fold-local positions would collide across folds and make coverage auditing meaningless. |
| `_make_loader(..., ctx, train, seed)` | padded sampler for training (DDP needs equal step counts), `UnpaddedDistributedSampler` for evaluation |
| `evaluate(..., ctx, expected_n, autocast_dtype, subject_lookup)` | builds `PredictionRecord`s, all-gathers, validates coverage **before** any metric |
| `_prepare_distributed` | DDP wrap with lazy-parameter materialization, resolves AMP dtype and `GradScaler` |
| `_train_one_epoch` | AMP-aware; calls `sampler.set_epoch(epoch)` — without it every epoch reuses one shuffle |
| `train_model` | uses `EarlyStopCoordinator` when distributed, so all ranks leave the loop on the same epoch |

Regression: one Gate 1 test needed updating for the new five-tensor loader signature, and a
test was added asserting the sample ids are dataset-global rather than split-local.

---

## 5. What Gate 3 still needs from PARAM

Run `sbatch scripts/param/2gpu_ddp_smoke.sbatch`. It performs, in order:

1. `preflight.py --gpu` — CUDA present, 2 devices, **sm_70**, NCCL available, fp16 autocast finite.
2. `preflight.py --gpu --nccl` under torchrun — a live 2-rank NCCL all-reduce with a checked sum.
3. The full distributed suite on real GPUs.

Then `sbatch scripts/param/memory_probe.sbatch` to resolve **B-008** — the V100s are 16 GB,
not the 32 GB the brief assumed, so no batch size anywhere is trustworthy until measured.

Gate 3 is marked PASS only when those produce clean artifacts under
`results/param_utkarsh_authoritative/preflight/`.
