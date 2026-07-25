# Gate 7 — DDP Systems Baseline

Status: **IMPLEMENTED, NOT EXECUTED.** `scripts/param/scaling_benchmark.py` — the forward
reference flagged at Gate 4 is now closed. Smoke-tested at world_size 1 on CPU.

## Method

Strong and weak scaling measured separately and **never mixed in one claim**:

| Mode | Fixed | Question |
|---|---|---|
| strong | effective **global** batch | does adding GPUs make this run faster? |
| weak | **per-rank** batch | can I process more data by adding GPUs? |

Measured per configuration: iteration median / mean / std / p95 / p99 / min / max over
≥ 5 repetitions after 10 warmup steps, throughput, peak GPU memory. The **slowest rank**
defines the iteration (`all_reduce(MAX)`) because that is what a synchronous step costs.

Efficiency is reported only against a 1-GPU baseline measured on the same hardware. Without
one, `parallel_efficiency` is **null** with a recorded note — never assumed.

## Workloads

| Name | T | F | global batch | Purpose |
|---|---:|---:|---:|---|
| studentlife_like | 60 | 8 | 256 | small-window regime |
| daicwoz_like | 2000 | 88 | 32 | the activation-heavy real case |
| synthetic_large | 4000 | 128 | 64 | scale the real corpora cannot reach |

Every workload is flagged `"synthetic": true`. Replicated or synthetic samples are **never**
counted as additional scientific subjects.

## Committed matrix

| Config | GPUs | Share of cluster | Status |
|---|---:|---:|---|
| 1 GPU | 1 | 5 % | committed |
| 1 node | 2 | 10 % | committed |
| 2 nodes | 4 | 20 % | committed |
| 4 nodes | 8 | 40 % | best-effort |
| 8 nodes | 16 | 80 % | **dropped — not schedulable (B-009)** |

## Blockers

| ID | Blocker |
|---|---|
| B-009 | 16-GPU is 80 % of the cluster's 20 V100s. The manuscript's N=16 (Table 3) cannot be reproduced. Tracker T2-07 must be answered by restating the scaling section against what actually ran. |
| B-017 | Network impairment (bandwidth, RTT, jitter, packet loss — tracker E4-05) needs `tc netem`, which requires root. Not available to a normal HPC user. Either request an admin-assisted experiment or drop the claim; injected application-level delay is measured instead and is **not** the same thing. |
