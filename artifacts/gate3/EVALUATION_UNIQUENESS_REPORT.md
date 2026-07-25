# Gate 3 — Evaluation Uniqueness Report

Generated: 2026-07-26
Machine-readable: `evaluation_uniqueness.json`

**Claim: every evaluation sample is scored exactly once, at every world size, on every
split this campaign uses.**

---

## 1. The defect being prevented

`torch.utils.data.DistributedSampler(drop_last=False)` — the default choice, and the one
almost every DDP tutorial shows — pads the index list up to a multiple of `world_size`
**by repeating samples from the front**.

For training this is necessary: DDP requires every rank to execute the same number of
backward passes, or the gradient all-reduce deadlocks.

For evaluation it is a silent correctness bug. Repeated samples are scored twice and enter
the metric twice. On a 47-session split that is a ~2% distortion applied to whichever
sessions happen to sort first, and **nothing in the output records that it happened**.

---

## 2. Measured, per split, per world size

`unpadded` = emitted by `UnpaddedDistributedSampler` · `padded` = what the stock sampler
would emit · `dupes` = samples that would have been double-counted.

| Split | n | ws | unpadded | padded | dupes | exact coverage |
|---|---:|---:|---:|---:|---:|---|
| **DAIC-WOZ official test** | **47** | **2** | **47** | **48** | **1** | ✅ |
| DAIC-WOZ official test | 47 | 4 | 47 | 48 | 1 | ✅ |
| DAIC-WOZ official test | 47 | 8 | 47 | 48 | 1 | ✅ |
| DAIC-WOZ official dev | 35 | 2 | 35 | 36 | 1 | ✅ |
| DAIC-WOZ official dev | 35 | 8 | 35 | 40 | 5 | ✅ |
| DAIC-WOZ official train | 107 | 2 | 107 | 108 | 1 | ✅ |
| DAIC-WOZ official train | 107 | 8 | 107 | 112 | 5 | ✅ |
| E-DAIC official dev / test | 56 | 2, 4, 8 | 56 | 56 | 0 | ✅ |
| E-DAIC official train | 163 | 2 | 163 | 164 | 1 | ✅ |
| E-DAIC official train | 163 | 8 | 163 | 168 | 5 | ✅ |
| E-DAIC full | 275 | 8 | 275 | 280 | 5 | ✅ |
| StudentLife windows | 2160 | 2, 4, 8 | 2160 | 2160 | 0 | ✅ |
| StudentLife CV fold | 432 | 2, 4, 8 | 432 | 432 | 0 | ✅ |

**Note E-DAIC's 56.** It is divisible by 2, 4 and 8, so the padded sampler happens to be
harmless there. That is luck, not design — and it is precisely why a guard is needed rather
than a convention. DAIC-WOZ's 47 and 35 are not divisible by anything useful.

---

## 3. Two independent mechanisms

**Prevention.** `UnpaddedDistributedSampler` partitions with `indices[rank::world_size]`.
No padding, no truncation, shard sizes differ by at most one, deterministic and
order-stable. Unequal shards are safe in evaluation because it runs under `no_grad` with no
gradient collective — the reason DDP wants even shards does not apply.

**Detection.** `assert_exact_coverage` runs on the gathered records **before any metric is
computed** and raises `EvaluationCoverageError` on a duplicate, a count mismatch, or an
unexpected id set. Metrics cannot be produced from a corrupt gather.

The guard is proven load-bearing by
`test_stock_padded_sampler_is_caught_by_the_coverage_guard`, which deliberately wires up
the stock padded sampler at `world_size=2` on the 47-sample case and asserts the guard
fires with a message naming the cause.

---

## 4. Auditable output

Every distributed evaluation attaches `_coverage_audit` to its metrics dict:

```json
{
  "expected_n": 47,
  "gathered_n": 47,
  "unique_n": 47,
  "duplicates": 0,
  "per_rank_counts": {"0": 24, "1": 23},
  "covers_exactly_once": true
}
```

This lands in `metrics.json` in the run directory, so a reviewer can verify coverage
without rerunning anything.

Sample ids are **dataset-global**, not split-local — fold-local positions would collide
across folds and make the audit meaningless. Asserted by
`test_sample_ids_are_dataset_global_not_split_local`.
