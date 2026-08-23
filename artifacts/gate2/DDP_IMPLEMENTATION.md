# Gate 2 — Full-Model DistributedDataParallel

Generated: 2026-07-26 · Branch `param-main`
Evidence: `artifacts/gate2/gate2_tests.{xml,log}`, `artifacts/gate2/ddp_parity_cpu_gloo.json`

**Result: 132 tests passed, 0 failures, 0 errors** (was 59 at Gate 1 — 73 added).

Status: **LOGIC-VERIFIED (CPU/gloo)**. This is *not* a hardware pass. NCCL, fp16 numerics,
V100 kernels and every performance figure remain Gate 3's hardware half — see
`DECISIONS.md` D-005.

---

## 1. Package

```
code/dsctm/src/dsctm/distributed/
├── __init__.py      public surface
├── runtime.py       process group, rank/device binding, seeding, precision, batch semantics
├── sampler.py       train sampler + UnpaddedDistributedSampler for evaluation
├── gather.py        prediction records, cross-rank gather, coverage validation
├── checkpoint.py    full-state save/resume incl. every RNG stream
├── logging.py       rank-0 write discipline, per-rank logs, run-directory contract
├── ddp.py           DDP wrap, lazy-parameter materialization, early-stop coordination
└── errors.py        synchronized failure propagation
```

Full-model DDP is the **control** implementation. SAP (Gate 8) and TCP (Gate 9) are
separate execution modes built on top of it. Nothing may be claimed faster or better than
a baseline that does not yet exist (`DECISIONS.md` D-007).

---

## 2. The four hazards this gate closes

### 2.1 Evaluation duplication — the DAIC-WOZ 47-session bug

`torch.utils.data.DistributedSampler(drop_last=False)` pads the index list up to a
multiple of `world_size` **by repeating samples from the front**. On the DAIC-WOZ official
test split at `world_size=2`:

| | value |
|---|---:|
| split size | 47 |
| stock padded sampler emits | **48** |
| duplicated sessions | **1** |
| `UnpaddedDistributedSampler` emits | 47 |

One session enters macro-F1 twice. Nothing in the output says so.

`UnpaddedDistributedSampler` partitions with `indices[rank::world_size]` — no padding, no
truncation, shard sizes differing by at most one. Unequal shards are safe in evaluation
because it runs under `no_grad` with no gradient collective.

Belt and braces: `assert_exact_coverage` runs *before* any metric is computed and raises
`EvaluationCoverageError` on a duplicate, a missing sample, or an unexpected id set. Proven
load-bearing by `test_stock_padded_sampler_is_caught_by_the_coverage_guard`, which
deliberately uses the stock sampler and asserts the guard fires.

The training path deliberately keeps the padded sampler: DDP requires every rank to run
the same number of backward passes or the gradient all-reduce deadlocks.

### 2.2 Global-batch semantics

The trap: take `batch_size` from the config, hand it unchanged to every rank, and silently
double the optimisation batch at `world_size=2`. The multi-GPU run is then a different
experiment from the single-GPU run it is compared against, with a different effective
learning rate.

`resolve_batch_semantics(scientific_global_batch, world_size, grad_accum_steps)` inverts
the relationship: the **global** batch is the fixed scientific quantity and the per-rank
size is derived. Indivisible combinations raise unless `allow_uneven=True` is passed
explicitly, and then the shortfall is recorded in `BatchSemantics.note`.

Every run records `per_rank_batch_size`, `world_size`, `grad_accum_steps`,
`effective_global_batch`, `scientific_global_batch`, `matches_scientific_intent`.

### 2.3 Divergent early stopping → collective deadlock

If rank 0 breaks the epoch loop and rank 1 does not, rank 1 blocks in the next gradient
all-reduce until the 72-hour wall-clock limit expires, holding a GPU-node reservation.

`EarlyStopCoordinator` decides on rank 0 and broadcasts an `EarlyStopDecision`.
`test_early_stop_decision_is_broadcast_from_rank_zero` feeds **every rank a different
score** and asserts all ranks execute the same number of epochs and see byte-identical
decisions.

### 2.4 `nn.LazyLinear` vs DDP

`ITransformerBaseline` uses `nn.LazyLinear`; DDP cannot wrap a module with
`UninitializedParameter`. `wrap_ddp(..., example_input=...)` runs a deterministic dry
forward under `no_grad` in eval mode, restores the training flag, and refuses to wrap if
any lazy parameter survives.

---

## 3. Declared numerical tolerance

`DDP_PARITY_ATOL = 1e-6` (absolute, fp32 parameters after one SGD step).

**Why not bitwise.** DDP forms the global gradient by all-reducing per-rank means; the
single-process reference forms it as one mean over the union. Floating-point addition is
not associative, so identical mathematics differs in the last ulp.

**Measured** (`artifacts/gate2/ddp_parity_cpu_gloo.json`):

| world_size | per-rank batch | global batch | max abs Δparam | max rel Δparam | replicas bitwise identical |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 16 | 1.490e-08 | 1.069e-07 | ✅ |
| 3 | 8 | 24 | 1.490e-08 | 4.977e-07 | ✅ |
| 4 | 8 | 32 | 1.490e-08 | 1.155e-07 | ✅ |

1.49e-08 is ~2 ulp at unit scale — 67× inside the tolerance. A genuine sharding or
reduction bug moves parameters by 1e-3 or more, so the tolerance discriminates.

Note the two different criteria, applied deliberately:

* **replica-vs-replica: bitwise.** All-reduce delivers the same buffer to every rank, so
  any difference is real divergence, not noise. Asserted via SHA-256 state digest.
* **DDP-vs-single-process: tolerance.** Reduction order genuinely differs.

The same tolerance re-applies on PARAM over NCCL, where the reduction tree differs again.

---

## 4. Test inventory (73 added)

`tests/test_distributed_units.py` — 51 single-process tests

| Area | Count | Notable |
|---|---:|---|
| Unpadded eval partition | 12 | parametrized over n ∈ {1,3,47,56,100,275,2160} × ws ∈ {1,2,3,4} |
| Global-batch semantics | 6 | refuses indivisible split; refuses ws > batch |
| Precision policy | 5 | **bf16 refused on sm_70**; fp16 → float16; unknown → raise |
| Prediction records / coverage | 8 | duplicate, missing, and wrong-id-set all rejected |
| Lazy parameters | 5 | iTransformer materialization; training mode preserved |
| Early stopping | 4 | max/min mode, patience, state round-trip |
| Run contract / atomic writes | 6 | required-file audit; no `.tmp` residue |
| DataLoader tuning | 2 | 4 workers/rank — does not oversubscribe 40 cores ÷ 2 ranks |
| Batch arithmetic | 3 | |

`tests/test_distributed_gloo.py` — 22 multi-process tests over real gloo collectives

| Test | world_size | Asserts |
|---|---|---|
| `test_ddp_one_step_matches_single_process_reference` | 2, 3 | replicas bitwise identical; vs reference within 1e-6 |
| `test_ddp_global_batch_semantics_are_preserved` | 2 | per-rank 8, global 16 |
| `test_eval_contains_no_duplicate_samples_and_covers_the_split` | 2, 3, 4 | 47 unique, 0 duplicates, per-rank counts sum to 47 |
| `test_stock_padded_sampler_is_caught_by_the_coverage_guard` | 2 | guard fires on the padded sampler |
| `test_early_stop_decision_is_broadcast_from_rank_zero` | 2, 3 | identical epoch count and decisions despite differing scores |
| `test_only_rank_zero_writes_shared_artifacts` | 2, 3 | one writer; every rank keeps its own log |
| `test_checkpoint_resume_matches_uninterrupted_run` | 2 | resumed digest == uninterrupted digest, **with dropout active** |
| `test_checkpoint_refuses_a_dataset_hash_mismatch` | 2 | refuses to resume onto different data |
| `test_failure_on_one_rank_terminates_every_rank` | 2, 3 | origin raises its own error; peers raise `RankFailure` |
| `test_split_hash_disagreement_across_ranks_is_fatal` | 2, 3 | ranks running different splits is fatal |
| `test_matching_split_hash_passes` | 2 | negative control |
| `test_train_sampler_set_epoch_reshuffles_and_covers_the_dataset` | 2, 3 | order changes; duplication is **exactly** the documented padding |
| `test_ddp_replicas_hold_identical_weights` | 2, 3 | |

Checkpoint resume runs **with dropout enabled on purpose** — it is the test that proves
RNG state is restored. The parity tests run with dropout disabled because a rank holding an
(8, F) shard and a reference holding the (16, F) union consume the RNG stream differently
and draw different masks; that divergence is correct PyTorch behaviour, so the parity test
removes the confound instead of papering over it.

---

## 5. What Gate 2 does NOT establish

| Claim | Status |
|---|---|
| NCCL correctness | ❌ NCCL unavailable locally — Gate 3 on PARAM |
| fp16 + GradScaler numerics | ❌ CPU fp16 is not representative of sm_70 — Gate 3 |
| V100 memory footprint | ❌ V100 is **16 GB HBM2**, not 32 GB — must be measured, see BLOCKERS B-008 |
| Throughput / scaling / efficiency | ❌ Gate 7 |
| InfiniBand behaviour | ❌ Gate 7 |
| Multi-node rank binding | ❌ single host only here — Gate 3/4 |
| SAP, TCP | ❌ Gates 8, 9 — not started |

---

## 6. Integration still to do (Gate 3 prerequisite)

The package is complete and tested, but `train/trainer.py` has **not** yet been rewired to
use it. That is deliberate: Gate 2 delivers a verified layer, Gate 3 integrates it and
proves parity end-to-end on real data. Remaining wiring:

1. `train_model` / `train_select_evaluate` accept a `DistContext` and build loaders via
   `make_train_sampler` / `make_eval_sampler`.
2. `evaluate` emits `PredictionRecord`s and routes through `gather_and_validate`.
3. `headline_cv` and the experiment runners guard every write with `ctx.is_main`.
4. `autocast` + `GradScaler` around the loss/backward/step.
5. `assert_agrees_across_ranks` on split hash, data hash and config hash at startup.
6. `EarlyStopCoordinator` replaces the local patience counter.
