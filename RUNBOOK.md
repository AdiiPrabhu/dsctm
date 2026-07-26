# D-MSTCN on PARAM Utkarsh — Operator Runbook

Everything you need to execute the campaign. Follow the phases in order; each one gates the
next. Estimated wall-clock is queue-dependent and excludes waiting.

**Login:** `ssh -X <user>@paramutkarsh.cdac.in -p 4422` → captcha → password.

---

## Cluster reference

| | |
|---|---|
| Scheduler | SLURM 20.11.8 |
| Partitions | `standard` · `cpu` · `gpu` · `hm` |
| GPU nodes | **10**, each 2 × V100 SXM2 **16 GB HBM2** (sm_70), 2 × Xeon G-6248 (40 cores), 192 GB RAM |
| **Cluster GPU total** | **20 V100s, shared** |
| CPU nodes | 75 × 2 × Xeon Platinum 8268 (48 cores), 192 GB |
| Interconnect | Mellanox InfiniBand HDR 100 Gbps |
| Storage | Lustre 1.3 PB, 25 GB/s |
| OS | CentOS 7.9 (glibc 2.17) |
| Max walltime | `72:00:00` |
| Login nodes | CPU/memory limited — **jobs are killed there** |

---

## Phase 0 — Clone and bootstrap  *(~20 min, login node)*

```bash
git clone git@github.com:AdiiPrabhu/dsctm.git
cd dsctm && git checkout param-main
cd codex/dsctm

module avail 2>&1 | tee ~/param_modules.txt      # SEE STEP 0a BEFORE CONTINUING
source scripts/param/env.sh
```

### 0a. Before the first `env.sh` — two values you must confirm

`env.sh` cannot guess these and guessing wrong wastes a queue slot.

```bash
module avail          # find the real anaconda / cuda module names
nvidia-smi            # driver version (run inside an allocation, or ask support)
```

Then re-run with overrides if the defaults were wrong:

```bash
DSCTM_MODULES="anaconda3/anaconda3 cuda/11.8" source scripts/param/env.sh --rebuild
```

**Why `torch==2.1.2+cu118` is the default.** CentOS 7.9 ships glibc 2.17. Recent PyTorch
wheels target `manylinux_2_28` (glibc ≥ 2.28) and will either refuse to install or install
and then **fail at import inside a queued job**. cu118 builds are the newest still
published as `manylinux_2_17`. If your driver supports newer CUDA:

```bash
DSCTM_TORCH_SPEC="torch==2.4.1" \
DSCTM_TORCH_INDEX="https://download.pytorch.org/whl/cu121" \
  source scripts/param/env.sh --rebuild
```

### 0b. Point scratch at Lustre, not `$HOME`

DAIC-WOZ audio alone is ~86 GB.

```bash
export DSCTM_SCRATCH=/scratch/$USER/dsctm     # confirm the real path with support
source scripts/param/env.sh
```

### 0c. Verify

```bash
python scripts/param/preflight.py
```

Expect dataset warnings (nothing staged yet). **Any hard failure must be fixed here** —
each one prints its own fix.

---

## Phase 1 — Stage datasets  *(hours, login node)*

```bash
# Kaggle token first (StudentLife only)
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
# copy kaggle.json from kaggle.com -> Account -> Create New API Token
chmod 600 ~/.kaggle/kaggle.json

nohup bash scripts/param/stage_datasets.sh --all > ~/staging.log 2>&1 &
tail -f ~/staging.log
```

| Dataset | Source | Size |
|---|---|---|
| DAIC-WOZ (AVEC2017) | `https://dcapswoz.ict.usc.edu/wwwdaicwoz/` | ~86 GB, sessions 300–492 |
| E-DAIC (AVEC2019) | `https://dcapswoz.ict.usc.edu/wwwedaic/` | 275 sessions |
| StudentLife | `kaggle datasets download -d dartweichen/student-life` | ~3 GB |

**Downloads run on the login node deliberately** — I/O-bound and low-CPU, so they fit
within login limits, and `wget -c` makes them resumable. Feature *extraction* is CPU-heavy
and is a batch job (Phase 2). If your login session drops, re-run the same command; nothing
restarts from zero.

Every archive is `unzip -t`-tested before extraction; failures are quarantined to
`*.CORRUPT` and listed in `_manifests/corrupt_archives.txt`. The previous campaign lost
DAIC-WOZ session 440 to a source-truncated zip — a half-extracted session that looks
complete is worse than one that fails loudly.

```bash
bash scripts/param/stage_datasets.sh --verify
```

This writes `dataset_hashes.json` and **cross-checks the downloaded E-DAIC splits against
the copies already in `reviewer-package/data/`**. A hash mismatch is a corpus-identity
problem and fails here rather than after 48 GPU-hours.

---

## Phase 2 — Extract features  *(~2–6 h, `cpu` partition)*

```bash
sbatch scripts/param/extract_features.sbatch
squeue --me
```

48 cores, `cpu` partition. **Never run this on a login node** — it will be killed part-way
and leave a cache that looks complete.

---

## Phase 3 — GPU validation  *(~30 min, `gpu` partition)*  ← DO NOT SKIP

```bash
sbatch scripts/param/2gpu_ddp_smoke.sbatch
```

Runs, in order:

1. `preflight.py --gpu` — CUDA present, 2 devices, **sm_70**, NCCL available, fp16 autocast finite
2. `preflight.py --gpu --nccl` under torchrun — a live 2-rank NCCL all-reduce with a checked sum
3. The full distributed test suite on real GPUs

```bash
cat $DSCTM_RESULTS_ROOT/preflight/preflight_gpu_*.json | python -m json.tool | head -40
```

Thirty minutes here prevents a 24-hour job failing at hour 23. **This is the job that
closes Gate 3.**

---

## Phase 4 — Memory probe  *(~1 h, `gpu` partition)*  ← SETS EVERY BATCH SIZE

```bash
sbatch scripts/param/memory_probe.sbatch
```

The engagement brief assumed 32 GB V100s. **PARAM has 16 GB.** Until this runs, every batch
size in every config — including `DAIC_CFG`'s `batch_size: 8` — is an assumption, and
DAIC-WOZ (T=2000, F=88, D=128, 3 branches) is the activation-heavy case.

```bash
python -m json.tool $DSCTM_RESULTS_ROOT/systems/memory_probe_*.json | tail -30
export DSCTM_BATCH_SIZE=<measured ceiling>
```

Closes **BLOCKER B-008**.

---

## Phase 4b — DRY RUN before the 294-task campaign  *(~1-4 h, `gpu` partition)*  ← DO THIS

Never submit a 294-task array on an estimate. Run five real tasks first, one per family,
fully instrumented.

```bash
sbatch scripts/param/1task_dryrun.sbatch
```

It prints the plan, runs one real task per family, extrapolates to the full campaign from
the **measured** time, and audits the contract on what it produced.

What it answers that an estimate cannot:

| Question | Why it matters |
|---|---|
| Does a task complete end-to-end on PARAM? | the whole pipeline, not just imports |
| Are all 15 contract files produced? | a run without them is not citable |
| What does one task actually cost? | replaces the a-priori bracket |
| How many epochs does early stopping really use? | the largest unknown in the estimate |

Then extrapolate and decide:

```bash
python scripts/param/calibrate.py --extrapolate $DSCTM_RESULTS_ROOT/calibration_<JOBID>.json
```

Want it faster and cheaper still? Cap the epochs — but the extrapolation then **understates**
real cost and is only good for proving the pipeline works:

```bash
python scripts/param/calibrate.py --sample 1 --max-epochs 3
```

**A note on "price".** PARAM Utkarsh is a national facility. You are not billed in currency;
you consume node-hours against your project's allocation. The number that matters is
GPU-hours (see `artifacts/gate5/compute_estimate.json`) and, socially, the share of the
cluster's **20 V100s** you hold at once. The arrays are throttled to `%4` for that reason.

---

## Phase 5 — Scientific campaign  *(days, `gpu` partition, 1 V100 per task)*

### 5a. Verify the array bounds — never type them

```bash
python scripts/param/plan.py
```

```
family                    tasks           --array  plan_digest
------------------------------------------------------------------
ablation                     78            0-77%4  7876bb29c494c34b
confirm-daicwoz              60            0-59%4  9f44701755325d6b
confirm-studentlife          60            0-59%4  6bfcb4986429ff6c
tuning-daicwoz               48            0-47%4  236956a0f6b99059
tuning-studentlife           48            0-47%4  99e8187f2cd00615
------------------------------------------------------------------
TOTAL                       294                    dcb6e197431c369d
```

If a `--array` bound in an sbatch file disagrees with this table, **the difference is
silently dropped work** and the campaign still reports success. A test enforces the match;
check it after any plan edit.

### 5b. Submit

```bash
sbatch scripts/param/tuning_array.sbatch      # 48 tasks, %4 concurrent
# wait for completion, then:
sbatch scripts/param/seeds_array.sbatch       # 60 tasks
sbatch scripts/param/ablation_array.sbatch    # 78 tasks
```

Order matters: confirmation seeds use configurations frozen by the tuning phase.

**One GPU per task, on purpose.** StudentLife is 2,160 windows and DAIC-WOZ is 275
sessions; these fit comfortably on one V100. A 1-GPU array gets scheduled far sooner than a
2-node reservation on a cluster with 20 GPUs total. `%4` throttles to 4 concurrent tasks —
20 % of the machine. Raise it only if the queue is empty.

### 5c. Inspect or re-run a single task

```bash
python scripts/param/plan.py --show ablation 40
python scripts/param/run_task.py --family ablation --index 40 --dry-run
python scripts/param/run_task.py --family ablation --index 40      # re-run one failure
```

---

## Phase 6 — Systems experiments  *(multi-GPU)*

```bash
DSCTM_SCALING_MODE=strong sbatch scripts/param/scaling.sbatch
DSCTM_SCALING_MODE=weak   sbatch scripts/param/scaling.sbatch
```

> **Not yet runnable.** `scaling.sbatch` invokes `scripts/param/scaling_benchmark.py`,
> which is Gate 7 work and does not exist. Submitting today fails with a missing-file error
> rather than doing something wrong.

### Scaling matrix reality

20 V100s exist in the entire machine:

| Config | GPUs | Share of cluster | Verdict |
|---|---:|---:|---|
| 1 GPU | 1 | 5 % | ✅ |
| 1 node | 2 | 10 % | ✅ |
| 2 nodes | 4 | 20 % | ✅ |
| 4 nodes | 8 | 40 % | ⚠️ long queue |
| 8 nodes | 16 | 80 % | ❌ not schedulable |

Committed matrix: **1 / 2 / 4** GPUs, 8 best-effort. The manuscript's N = 16 is not
reproducible here; its absence is stated in the paper (tracker T2-07) rather than papered
over.

---

## Monitoring

```bash
squeue --me
squeue --me -o "%.10i %.20j %.8T %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>
tail -f logs/dsctm-*.out

# campaign progress
ls $DSCTM_RESULTS_ROOT/*/ | wc -l
grep -l '"status": "completed"' $DSCTM_RESULTS_ROOT/*/*/status.json | wc -l
grep -l '"status": "model_failed"' $DSCTM_RESULTS_ROOT/*/*/status.json
```

---

## What a finished run looks like

Every run directory must contain all 15 files. A run missing any is **not complete**,
regardless of exit code — `finalize()` downgrades it to `infrastructure_failed` and records
why.

```
command.txt  resolved_config.yaml  environment.json  git.json  slurm.json
hardware.json  dataset_hashes.json  split_hashes.json  stdout.log  stderr.log
metrics.json  predictions.parquet  checkpoint.pt  status.json  receipt.sha256
```

`receipt.sha256` binds every evidence file. `status.json` quotes the receipt and records
the verdict, any waivers, and the coverage audit.

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `DSCTM_MODULES` | `anaconda3/anaconda3 cuda/11.8 cuda/12.0` | run `module avail` first |
| `DSCTM_TORCH_SPEC` | `torch==2.1.2 torchvision==0.16.2` | glibc 2.17 constraint |
| `DSCTM_SCRATCH` | `~/scratch/dsctm` | **point at Lustre** |
| `DSCTM_BATCH_SIZE` | `32` | set from the Phase 4 probe |
| `DSCTM_PRECISION` | `fp16` | **never bf16** — sm_70 has no bf16 tensor cores |
| `DSCTM_NUM_WORKERS` | `4` | per rank; 40 cores ÷ 2 ranks |
| `NCCL_DEBUG` | `WARN` | `INFO` when diagnosing; very noisy |
| `NCCL_IB_DISABLE` | `0` | `1` forces TCP fallback — slow, last resort |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: libc.so.6 version GLIBC_2.28` | torch wheel too new for CentOS 7.9 | `DSCTM_TORCH_SPEC="torch==2.1.2" source scripts/param/env.sh --rebuild` |
| `LOCAL_RANK exceeds visible CUDA device count` | `--ntasks-per-node` > `--gres=gpu:N` | tasks must not exceed allocated GPUs |
| Job hangs at ~0 % forever | a rank died inside a collective | `NCCL_DEBUG=INFO`, check `rank*.log` in the run dir |
| `PreflightFailure: bf16 requested on sm_70` | working as intended | use `fp16`; V100 has no bf16 tensor cores |
| `EvaluationCoverageError: duplicate prediction` | padded sampler reached evaluation | working as intended — report it, do not suppress |
| `IndexError: array index N outside family` | sbatch `--array` disagrees with the plan | `python scripts/param/plan.py --sbatch-array <family>` |
| Login session killed mid-download | login-node CPU/memory limit | re-run; `wget -c` resumes |
| `status.json` says `infrastructure_failed` after a clean run | a required file is missing | read `contract_violation` in `status.json` |

---

## Where results go

```
results/param_utkarsh_authoritative/   <- the ONLY citable evidence root
results/local_non_authoritative/       <- never cited, never merged
```

Nothing from the historical RTX 4060 Ti campaign may be cited: no raw artifact from it
exists anywhere in this repository. See `artifacts/gate0/OLD_RESULT_QUARANTINE.md` (27
registered claims). Only three prior findings survive, because they are properties of code
or arithmetic: receptive fields 61/481/1921, adapter cost `d_s`=8, and Wilcoxon n=5 min
p=0.0625.
