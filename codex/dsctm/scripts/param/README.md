# Running D-MSTCN on PARAM Utkarsh

`ssh -X <user>@paramutkarsh.cdac.in -p 4422` (captcha, then password)

## Cluster facts these scripts encode

| | |
|---|---|
| Scheduler | SLURM 20.11.8 |
| Partitions | `standard` · `cpu` · `gpu` · `hm` |
| GPU nodes | **10**, each 2 × V100 SXM2 **16 GB HBM2** (sm_70), 2 × Xeon G-6248 (40 cores), 192 GB RAM, 480 GB local SSD |
| Cluster GPU total | **20 V100s** — shared |
| CPU nodes | 75 × 2 × Xeon Platinum 8268 (48 cores), 192 GB |
| Interconnect | Mellanox InfiniBand HDR 100 Gbps |
| Storage | Lustre, 1.3 PB, 25 GB/s |
| OS | CentOS 7.9 (glibc 2.17) |
| Max walltime | `72:00:00` |
| Login nodes | CPU-time and memory limited; **do not run jobs there** |

## Order of operations

```bash
# 0. one time
git clone <repo> && cd <repo>/codex/dsctm
source scripts/param/env.sh                  # creates the conda env, sets all roots

# 1. login node: CPU-only checks
python scripts/param/preflight.py

# 2. login node: download (I/O bound, resumable, fits login limits)
bash scripts/param/stage_datasets.sh --all

# 3. cpu partition: feature extraction (CPU heavy - never on a login node)
sbatch scripts/param/extract_features.sbatch

# 4. gpu partition: FIRST GPU JOB. Validates env, both V100s, NCCL, fp16, test suite.
sbatch scripts/param/2gpu_ddp_smoke.sbatch

# 5. gpu partition: resolve BLOCKER B-008 - the real batch ceiling on a 16 GB V100
sbatch scripts/param/memory_probe.sbatch

# 6. science (single V100 each, arrays)
sbatch scripts/param/tuning_array.sbatch
sbatch scripts/param/seeds_array.sbatch
sbatch scripts/param/ablation_array.sbatch

# 7. systems (multi-GPU)
DSCTM_SCALING_MODE=strong sbatch scripts/param/scaling.sbatch
DSCTM_SCALING_MODE=weak   sbatch scripts/param/scaling.sbatch
```

Do **not** skip step 4. It is 30 minutes that prevents a 24-hour job failing at hour 23.

## Which jobs need multiple GPUs

Most of them do not. StudentLife is 2,160 windows; DAIC-WOZ is 275 sessions. These fit
comfortably on one V100, and a 1-GPU array gets scheduled far sooner than a 2-node
reservation on a cluster with 20 GPUs total.

| Use 1 GPU | Use multiple GPUs |
|---|---|
| tuning, confirmation seeds, ablations, transfer, calibration | DDP parity, scaling, communication profiling, SAP/TCP, synthetic large workloads |

## Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `DSCTM_ENV_PREFIX` | `~/.conda/envs/dsctm` | conda env location |
| `DSCTM_PY_VERSION` | `3.10` | |
| `DSCTM_MODULES` | `anaconda3/anaconda3 cuda/11.8 cuda/12.0` | run `module avail` and adjust |
| `DSCTM_TORCH_SPEC` | `torch==2.1.2 torchvision==0.16.2` | **cu118 chosen for glibc 2.17** — see below |
| `DSCTM_SCRATCH` | `~/scratch/dsctm` | point at Lustre scratch, not `$HOME` |
| `DSCTM_GPUS_PER_NODE` | auto via `nvidia-smi -L` | |
| `DSCTM_NUM_WORKERS` | `4` | per rank; 40 cores ÷ 2 ranks, leave room for NCCL and Lustre |
| `NCCL_DEBUG` | `WARN` | `INFO` when diagnosing — very noisy |

### Why torch 2.1.2 + cu118 by default

CentOS 7.9 ships **glibc 2.17**. Recent PyTorch wheels target `manylinux_2_28`
(glibc ≥ 2.28) and will either refuse to install or install and then fail at import inside
a queued job. cu118 builds are the newest still published as `manylinux_2_17`.

Check `nvidia-smi` for the driver version and override if the site supports newer:

```bash
DSCTM_TORCH_SPEC="torch==2.4.1" DSCTM_TORCH_INDEX="https://download.pytorch.org/whl/cu121" \
  source scripts/param/env.sh --rebuild
```

`preflight.py` verifies whatever gets installed and fails loudly if NCCL or CUDA is absent.

## Precision

**fp16 only.** V100 is sm_70 and has no bf16 tensor cores. `autocast_dtype("bf16", ...)`
raises rather than letting PyTorch silently emulate and produce numbers that look plausible
and are comparable to nothing. `GradScaler` is mandatory at fp16, not optional.

## Scaling matrix reality check

20 V100s exist in the whole machine:

| Config | GPUs | Share of cluster | Verdict |
|---|---:|---:|---|
| 1 GPU | 1 | 5 % | ✅ |
| 1 node | 2 | 10 % | ✅ |
| 2 nodes | 4 | 20 % | ✅ |
| 4 nodes | 8 | 40 % | ⚠️ long queue |
| 8 nodes | 16 | 80 % | ❌ not schedulable |

The committed matrix is **1 / 2 / 4** GPUs with 8 as best-effort. The manuscript's N = 16
is not reproducible here and its absence gets stated in the paper (tracker T2-07) rather
than papered over.
