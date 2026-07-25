# Gate 8 — Scale-Aware Partitioner

Status: **IMPLEMENTED AND EQUIVALENCE-VERIFIED (CPU/gloo, world_size 4).**
Evidence: `../gate12/full_suite.xml` — `tests/test_sap_tcp.py`, 27 tests.

## What it is

Genuine branch-parallel execution, not DDP with extra steps. Each temporal branch lives on
its own rank; activations ship to an aggregator rank owning CSAG + FiLM + head.

```
rank 0  SSB          rank 2  LSB
rank 1  MSB          rank 3  aggregator (CSAG + FiLM + head + loss)
```

Minimum topology is 4 ranks. A PARAM node has 2 V100s, so this is `--nodes=2 --gres=gpu:2`.
`plan_placement` raises if given fewer.

## Placement implements Eq. (11)

`L_b = C_b^compute / (C_b^compute + C_b^comm)`, branches sorted descending, assigned to the
least-loaded ranks. Surplus ranks beyond `branches + 1` become data-parallel replicas of the
heaviest branches (§III-C, N > 3), recorded in `replica_groups`. Deterministic and tested.

## Cross-rank autograd — and why the obvious design fails

The natural implementation is a pair of autograd Functions whose backward issues
`dist.send`/`dist.recv`. **It does not work.** PyTorch runs backward on a dedicated worker
thread; gloo binds transport buffers per thread, so p2p from the autograd thread aborts:

```
gloo::EnforceNotMet: Cannot lock pointer to unbound buffer
```

`torch.autograd.set_multithreading_enabled(False)` did not reliably fix it either.

The working design cuts the graph at the rank boundary and re-joins it manually:

| Phase | Branch rank | Aggregator |
|---|---|---|
| forward | compute `H_b`, keep graph locally, send `H_b.detach()` | receive into a **leaf** with `requires_grad=True` |
| backward | receive grad, run `H_b.backward(grad)` | `loss.backward()`, read `leaf.grad`, send it back |

Every collective runs on the main thread where the process group was created. This is the
standard manual model-parallel pattern; recorded because it is a real constraint that will
bite anyone who assumes NCCL makes it go away (it does not — NCCL is merely more tolerant).

## Verified

| Property | Test |
|---|---|
| forward matches monolithic model | `max_abs_diff < 1e-5` at ws 4 |
| backward reaches every rank with non-zero gradients | asserted per rank |
| aggregator head gradient matches the monolithic reference | `< 1e-4` |
| ranks hold gradients **only** for what they own | branch ranks have no head grad; aggregator has no branch grad |
| communication volume measured, not estimated | aggregator receives exactly `3·B·T·D·4` bytes forward (tracker **E4-17**) |
| branch order deterministic across ranks | fixed `BRANCH_ORDER`, never set iteration |
| placement refuses a 2-GPU node | `PreflightFailure` |

Tolerance is 1e-5 rather than bitwise: activations make a p2p round trip and the aggregator
sums three received tensors in a different order than the fused monolithic stack. A genuine
wiring bug produces ~1e-1.

## Blockers

| ID | Blocker |
|---|---|
| B-018 | Equivalence verified on gloo/CPU only. NCCL on 2 PARAM nodes is unverified; p2p semantics differ. |
| B-019 | `replicate_gradients` calls `dist.new_group` per invocation — acceptable for correctness, wasteful in a hot loop. Cache the groups before any timing claim at world_size > 4. |
