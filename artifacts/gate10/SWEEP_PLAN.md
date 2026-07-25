# Gate 10 — SAP/TCP Systems Experiments

Status: **IMPLEMENTED, NOT EXECUTED.** `scripts/param/sap_tcp_sweep.py`. Smoke-tested at
world_size 1 (SAP modes correctly skipped with an explicit notice).

## Sweep

| Parameter | Grid |
|---|---|
| `delta_max` | 5, 10, 20, 50, 100, 200 |
| `T_sync` | 10, 50, 100, 200 |
| injected branch delay | 0, 1, 5, 20 ms |
| straggler rate | 0, 0.10, 0.25 |

Measured per cell: iteration median/mean/p95/std, throughput, final loss, communication
volume by direction, TCP invariants, HOLD counts, periodic-sync counts, staleness
distribution.

## Honesty constraints encoded in the script

* Every impairment is **injected by this script** and recorded exactly as configured.
  `"controlled_experiment": true` appears in every result row.
* Injected delay is recorded separately from measured time. Nothing is inferred or
  extrapolated.
* SAP modes are **skipped with an explicit printed notice** when `world_size < 4`, rather
  than silently reporting DDP numbers under a SAP label.

## Blockers

| ID | Blocker |
|---|---|
| B-013 | Needs ≥ 4 ranks = 2 PARAM nodes = 20 % of the cluster. This is the gating experiment for every TCP claim in the paper. |
| B-017 | Real network impairment (bandwidth/RTT/jitter/loss, tracker E4-05) needs `tc netem` and root. Application-level injected delay is **not** the same thing and is labelled as what it is. Either request admin assistance or withdraw the claim. |
| B-021 | `--full-grid` is 6×4×4×3 = 288 cells × 4 modes. At 2 nodes that is a large allocation; the default is an 8-cell subset and the full grid needs separate compute approval. |
