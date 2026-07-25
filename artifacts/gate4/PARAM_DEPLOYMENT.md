# Gate 4 — PARAM Utkarsh Deployment Infrastructure

Generated: 2026-07-26 · Branch `param-main`

**Status: BUILT AND SYNTAX-VERIFIED. NOT EXECUTED.** No script here has run on PARAM.
Gate 4 passes when `2gpu_ddp_smoke.sbatch` produces a clean artifact set under
`results/param_utkarsh_authoritative/preflight/`.

---

## 1. Delivered

| File | Purpose |
|---|---|
| `env.sh` | conda env bootstrap, dataset roots, NCCL/InfiniBand settings |
| `preflight.py` | 23 hard/soft checks; exits non-zero on any hard failure |
| `stage_datasets.sh` | pinned dataset URLs, resumable, corrupt-archive quarantine |
| `verify_datasets.py` | `dataset_hashes.json` + split cross-check against the in-repo copies |
| `memory_probe.py` / `.sbatch` | resolves B-008 — the real batch ceiling on a 16 GB V100 |
| `launch_torchrun.sh` | rendezvous derived from SLURM; nothing hardcoded |
| `_header.inc` | shared sbatch preamble |
| `2gpu_ddp_smoke.sbatch` | **first GPU job**: env + 2×V100 + NCCL + fp16 + test suite |
| `1gpu_science.sbatch` | single-V100 scientific job |
| `2node_4gpu_ddp.sbatch`, `4node_8gpu_ddp.sbatch` | multi-node DDP |
| `extract_features.sbatch` | eGeMAPS extraction on the **cpu** partition |
| `tuning_array.sbatch`, `seeds_array.sbatch`, `ablation_array.sbatch` | throttled SLURM arrays |
| `scaling.sbatch` | Gate 7 strong/weak scaling |
| `README.md` | operator runbook |

All 3 Python scripts compile; all 11 shell/sbatch scripts pass `bash -n`. `preflight.py`
was dry-run locally and correctly reported 5 hard failures with fixes.

---

## 2. Design decisions forced by the cluster

### 2.1 Nothing is hardcoded

`launch_torchrun.sh` derives everything at runtime:

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))   # two jobs on a node cannot collide
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
```

`scontrol show hostnames` is required because `SLURM_JOB_NODELIST` uses the compact form
(`gpu[01-04]`), which is not a hostname.

### 2.2 Downloads on the login node, extraction on `cpu`

User Manual p.9: login nodes enforce CPU-time and memory limits and *terminate* offenders.
Access Guide p.7: "don't run any jobs in login nodes."

DAIC-WOZ is ~86 GB. Downloading is I/O-bound and low-CPU, so it fits within login limits and
is resumable (`wget -c`). eGeMAPS extraction over 189 sessions is CPU-heavy and would be
killed — it goes to the `cpu` partition with 48 cores. This split is also mandatory if
compute nodes turn out to lack internet egress, which is still unconfirmed.

### 2.3 Corrupt archives are quarantined, not silently skipped

The prior campaign lost DAIC-WOZ session 440 to a source-truncated zip. `stage_datasets.sh`
runs `unzip -t` before extracting and moves failures to `*.CORRUPT`, appending to
`corrupt_archives.txt`. A partially-extracted session that looks complete is worse than one
that fails loudly.

### 2.4 Arrays throttled to 4 concurrent tasks

`--array=0-47%4`. The cluster has 20 V100s shared across all users; an unthrottled 48-task
array would try to take most of it.

### 2.5 fp16, never bf16

V100 is sm_70. `autocast_dtype("bf16", ...)` raises. `GradScaler` is mandatory at fp16.

### 2.6 4 dataloader workers per rank

A GPU node has 40 cores and 2 ranks. 20 workers per rank oversubscribes once the main
process, NCCL threads and Lustre I/O are counted, and each worker forks the in-memory
`TensorDataset`. 4 is the default, overridable via `DSCTM_NUM_WORKERS`.

### 2.7 Lustre is bad at many small files

`RunLogger` opens the per-rank log once and appends; no per-step artifacts are written.
Shared artifacts are written once at their natural checkpoint, atomically.

---

## 3. Dataset sources pinned

| Dataset | Source | Notes |
|---|---|---|
| DAIC-WOZ (AVEC2017) | `https://dcapswoz.ict.usc.edu/wwwdaicwoz/` | ~86 GB, sessions 300–492, official 107/35/47 |
| E-DAIC (AVEC2019) | `https://dcapswoz.ict.usc.edu/wwwedaic/` | 275 sessions, official 163/56/56 |
| StudentLife | `kaggle datasets download -d dartweichen/student-life` | needs `~/.kaggle/kaggle.json` |

`verify_datasets.py` cross-checks downloaded E-DAIC split CSVs against the copies already
in `reviewer-package/data/` and **fails on a hash mismatch** — a corpus-identity problem
must surface before 48 GPU-hours are spent, not after.

It also guards tracker **V3-02** explicitly: if DAIC-WOZ dev+test sums to 82, it flags the
manuscript's merged split rather than letting it be revived.

---

## 4. Run-directory contract

Enforced by `distributed/logging.py::audit_run_directory` (15 required files):

```
command.txt  resolved_config.yaml  environment.json  git.json  slurm.json
hardware.json  dataset_hashes.json  split_hashes.json  stdout.log  stderr.log
metrics.json  predictions.parquet  checkpoint.pt  status.json  receipt.sha256
```

A run missing any file is **not complete**, regardless of exit code.

---

## 5. Open items before Gate 4 can pass

| Item | Owner | Blocking |
|---|---|---|
| Run `module avail`, confirm module names for `DSCTM_MODULES` | author | env.sh |
| Confirm `nvidia-smi` driver version → final torch pin | author | env.sh |
| Confirm compute-node internet egress | author | staging strategy |
| Confirm Lustre scratch path and quota | author | `DSCTM_SCRATCH` |
| Kaggle API token in place | author | StudentLife |
| Execute `2gpu_ddp_smoke.sbatch` | author | **Gate 4 pass** |
| Execute `memory_probe.sbatch` | author | B-008, all batch sizes |
| `scaling_benchmark.py` | Gate 7 | `scaling.sbatch` references it; not yet written |

The last row is a known forward reference: `scaling.sbatch` is delivered now so the SLURM
side is reviewable, but the benchmark it invokes is Gate 7 work and does not exist yet.
Submitting it today will fail with a missing-file error rather than doing something wrong.
