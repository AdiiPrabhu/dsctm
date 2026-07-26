# PARAM Utkarsh — Exact Run Order

You are logged in. Run these in order. Every step says what it needs, what it proves, and
what to check before moving on.

> ## ⚠ Two hard rules from the login banner
>
> **1. Never run anything substantial on a login node.**
> *"Please Don't run any jobs in Login Nodes. If you run, user will be disabled
> automatically."* Downloads, feature extraction and training all go through `sbatch`.
> Editing files, `git`, `module avail` and a single `curl -I` are fine.
>
> **2. Every job needs an account.** *"Use #SBATCH -A account name in your script."*
> All scripts carry a placeholder; `submit.sh` substitutes your real account at submit time
> so it never enters the repository.

---

## Corrected cluster facts

The banner corrects three things I had assumed:

| | Assumed | **Actual** |
|---|---|---|
| Partitions | `standard`, `cpu`, `gpu`, `hm` | **`standard*`, `gpu`, `hm`, `debug`** — there is **no `cpu` partition** |
| GPU nodes in `gpu` | 10 | **9** (`gpu002-010`); `gpu001` sits in `debug` |
| Max walltime | 72 h | **72 h** (`03-00:00:00`) ✓ · `debug` is capped at **1 h** |

`standard` is the default partition and contains **everything** — `cn006-107`, `gpu001-010`,
`hm001-039`. `debug` (`cn001-005`, `gpu001`, `hm001`) is a separate small pool, which makes
it ideal for a first smoke test: it does not queue behind production GPU work.

---

## Step 0 — Account and clone  *(login node, 2 min)*

```bash
sacctmgr show associations user=$USER format=Account,Partition,QOS
export DSCTM_ACCOUNT=<the account it prints>          # required on every job

git clone git@github.com:AdiiPrabhu/dsctm.git
cd dsctm && git checkout param-main && cd codex/dsctm
```

Put `export DSCTM_ACCOUNT=...` in your `~/.bashrc` so you cannot forget it.

---

## Step 1 — Find the real module names  *(login node, 2 min)*

```bash
module avail 2>&1 | tee ~/param_modules.txt
grep -iE "anaconda|conda|python|cuda" ~/param_modules.txt
```

Note the exact anaconda and cuda module names. You need them in the next step.

---

## Step 2 — Build the environment  *(login node, ~15 min)*

`pip install` is I/O-bound and short. This is the one heavy-ish thing that belongs on the
login node, and it still takes minutes, not hours.

```bash
export DSCTM_SCRATCH=/scratch/$USER/dsctm     # confirm the real Lustre path first
DSCTM_MODULES="<anaconda module from step 1>" source scripts/param/env.sh
```

**Why torch 2.1.2 + cu118 by default:** CentOS 7.9 ships glibc 2.17; newer PyTorch wheels
need ≥ 2.28 and will fail *at import inside a queued job*. Override only if `nvidia-smi`
shows a driver that supports newer CUDA:

```bash
DSCTM_TORCH_SPEC="torch==2.4.1" \
DSCTM_TORCH_INDEX="https://download.pytorch.org/whl/cu121" \
  source scripts/param/env.sh --rebuild
```

---

## Step 3 — Preflight  *(login node, 30 s)*

```bash
python scripts/param/preflight.py
```

Expect dataset warnings. **Fix every hard failure before continuing** — each prints its own
fix. Do not proceed with a red preflight; that is what it is for.

---

## Step 4 — Can compute nodes reach the internet?  *(login node, 10 s)*

```bash
bash scripts/param/check_egress.sh
```

Three HEAD requests, a few hundred bytes. This decides how Step 5 works — many HPC sites
block egress from compute nodes, and PARAM's is unconfirmed.

---

## Step 5 — Stage datasets  *(batch job, hours)*

```bash
export KAGGLE_API_TOKEN=KGAT_...              # StudentLife only
bash scripts/param/submit.sh stage_datasets.sbatch
squeue --me
```

Runs on `standard`, **not** the login node. It re-checks egress on the compute node and
**aborts with instructions** if blocked, rather than silently producing an empty tree.

If compute nodes are blocked but the login node is not, the fallback is `rsync` from a
machine you control:

```bash
rsync -avP --partial -e 'ssh -p 4422' ./datasets/ \
  $USER@paramutkarsh.cdac.in:$DSCTM_DATA_ROOT/
```

**Expire the Kaggle token once StudentLife lands** — it is in your shell history.

---

## Step 6 — First GPU job: smoke test  *(debug partition, ~20 min)*  ← DO NOT SKIP

```bash
bash scripts/param/submit.sh --debug 2gpu_ddp_smoke.sbatch
```

`--debug` sends it to the `debug` pool (1 h cap, `gpu001`) so it does not queue behind
production work. Validates env, both V100s, NCCL, fp16, and the distributed suite on real
GPUs.

```bash
squeue --me ; squeue --start          # estimated start time
tail -f logs/dsctm-ddp-smoke.*.out
```

**This closes Gate 3.** Twenty minutes here prevents a 24-hour job failing at hour 23.

If `debug` is busy, drop `--debug` to use the `gpu` partition.

---

## Step 7 — Memory probe  *(gpu partition, ~1 h)*  ← SETS EVERY BATCH SIZE

```bash
bash scripts/param/submit.sh memory_probe.sbatch
```

The brief assumed 32 GB V100s; PARAM has **16 GB**. Until this runs, every batch size in
every config is an assumption.

```bash
python -m json.tool $DSCTM_RESULTS_ROOT/systems/memory_probe_*.json | tail -30
export DSCTM_BATCH_SIZE=<measured ceiling>
```

---

## Step 8 — Extract features  *(standard partition, 2–6 h)*

```bash
bash scripts/param/submit.sh extract_features.sbatch
```

48 cores on `standard`. Needs Step 5 complete.

---

## Step 9 — DRY RUN  *(gpu partition, 1–4 h)*  ← BEFORE THE 294

```bash
bash scripts/param/submit.sh 1task_dryrun.sbatch
```

Five real tasks, one per family. Measures actual cost, extrapolates to the full campaign,
audits the contract on what it produced.

```bash
python scripts/param/calibrate.py --extrapolate \
  $DSCTM_RESULTS_ROOT/calibration_<JOBID>.json
```

**Decide here.** If the extrapolation is much larger than the ~91 GPU-hour estimate, adjust
scope before spending the allocation, not after.

Want it in minutes instead? `python scripts/param/calibrate.py --sample 1 --max-epochs 3`
proves the pipeline but **understates** cost — good for plumbing, not for sizing.

---

## Step 10 — Start monitoring  *(standard partition, background)*

```bash
bash scripts/param/submit.sh monitor.sbatch
# then, from your laptop:
scp -P 4422 $USER@paramutkarsh.cdac.in:$PWD/../../artifacts/monitoring/dashboard.html .
open dashboard.html
```

Self-contained HTML — no CDN, works offline.

---

## Step 11 — The campaign  *(gpu partition, days)*

Verify the array bounds first. **Never type them.**

```bash
python scripts/param/plan.py
```

```bash
bash scripts/param/submit.sh tuning_array.sbatch     # 48 tasks, %4
# wait for completion, then
bash scripts/param/submit.sh seeds_array.sbatch      # 60 tasks
bash scripts/param/submit.sh ablation_array.sbatch   # 78 tasks
```

Order matters: confirmation seeds use configurations frozen by tuning.

---

## Step 12 — Systems experiments  *(2 nodes)*

```bash
DSCTM_SCALING_MODE=strong bash scripts/param/submit.sh scaling.sbatch
DSCTM_SCALING_MODE=weak   bash scripts/param/submit.sh scaling.sbatch
bash scripts/param/submit.sh 2node_4gpu_ddp.sbatch
```

The `gpu` partition has **9 nodes = 18 V100s**. A 2-node job is 22 % of it; 4 nodes is 44 %.
The manuscript's N = 16 would be 89 % — not schedulable, which is the honest answer to
tracker T2-07.

---

## Step 13 — Evidence

```bash
python scripts/param/audit_campaign.py --all --aggregate --out audit.json
python scripts/param/build_evidence.py --out artifacts/final
```

The auditor is fail-closed: a family is admitted whole or not at all. If it admits nothing,
nothing is citable — that is the pipeline working, not a bug.

---

## Quick reference

```bash
squeue --me                       # my jobs
squeue --start                    # estimated start times
scontrol show job <id>            # full detail
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <id>
sinfo -p gpu                      # GPU partition state
tail -f logs/<jobname>.<jobid>.out
```

## If something fails

| Symptom | Fix |
|---|---|
| `Invalid account` | `export DSCTM_ACCOUNT=<real account>`; find it with `sacctmgr show associations user=$USER` |
| `Invalid partition: cpu` | already fixed — pull latest; the partition is `standard` |
| `GLIBC_2.28 not found` | `DSCTM_TORCH_SPEC="torch==2.1.2" source scripts/param/env.sh --rebuild` |
| Job pends forever | `squeue --start`; try `--debug` for short jobs; drop node count |
| Staging aborts on egress | see Step 5 fallback — `rsync` from a machine you control |
| `IndexError: array index N outside family` | `python scripts/param/plan.py --sbatch-array <family>` |
