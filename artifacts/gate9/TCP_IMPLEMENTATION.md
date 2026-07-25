# Gate 9 — Temporal Consistency Protocol (real)

Status: **IMPLEMENTED AND VERIFIED (CPU/gloo).** `src/dsctm/distributed/tcp_real.py`.
Evidence: 27 tests in `test_sap_tcp.py` + 108 in `test_theorem_invariant.py`.

## What changed

`train/tcp.py` is a single-process counter simulator: it increments a Python integer and
resets a dict. It performs no communication, synchronises no parameters and touches no
optimizer state. **It must not be described as a training protocol.** It is retained for the
Gate 0 invariant tests and is now explicitly labelled a simulator.

`tcp_real.py` performs actual `dist.all_reduce` on branch parameters within replica groups,
byte-counted into `CommStats`.

## Definitions committed to

The manuscript is imprecise about what is stale. This implementation states it:

| Term | Definition |
|---|---|
| `v_b` | branch parameter version — increments when `b` applies a local optimizer step |
| `V` | global version — increments on every synchronisation event |
| `Δ_b` | local steps applied by `b` since its last synchronisation |
| HOLD | `Δ_b ≥ δ_max` → `b` suspends updates until synchronisation completes |
| synchronisation | all-reduce of branch parameters in the replica group, then `Δ_b := 0`, `V += 1` |

Decisions are taken on the aggregator and **broadcast**. Ranks deciding independently is how
a collective deadlocks; a test asserts all ranks agree on all 20 decisions.

## Optimizer state — a recorded choice, not a silent one

Adam moments are branch-local and **not** synchronised by default. Averaging second-moment
estimates across replicas that saw different data is not well-defined. `sync_optimizer_state=True`
exists so the alternative can be *measured*, and the setting is written into every run
record and every checkpoint.

## Four execution modes

| Mode | Role |
|---|---|
| `full_model_synchronous_ddp` | **control** |
| `synchronous_sap` | isolates partitioning cost from asynchrony cost |
| `asynchronous_sap_without_tcp` | the failure mode TCP claims to fix; divergence unbounded by construction |
| `asynchronous_sap_with_tcp` | bounded divergence via HOLD + periodic sync |

## Blockers

| ID | Blocker |
|---|---|
| B-012 | Verified in simulation and 4-rank gloo. Real NCCL asynchrony on PARAM is unmeasured. |
| B-020 | `train/tcp.py` and `tcp_real.py` now coexist. The old module must be deleted once Gate 10 output exists, or a future reader will cite the simulator. |
