# PARAM Utkarsh — Run Order (authoritative)

**Follow this file.** `RUNBOOK.md` is the original plan and is now partly stale — it was
written before the cluster was inspected. Everything below is verified against the real
machine on 2026-07-26.

---

## Verified facts

| | Value | How we know |
|---|---|---|
| Account | `nsmexternal` | `sacctmgr -P show associations` |
| Partitions | `standard*` · `gpu` · `hm` · `debug` | login banner — **there is no `cpu` partition** |
| `gpu` partition | 9 nodes (`gpu002-010`), 2×V100 **16 GB** each | banner + manual |
| `debug` partition | `cn001-005`, `gpu001`, `hm001`, **1 h cap** | banner |
| Max walltime | `03-00:00:00` (72 h) | banner |
| Scratch | `/scratch` Lustre, **548 TB free** | `df -h` |
| Conda env | `~/.conda/envs/dsctm`, **Python 3.10.18** | built on-cluster |
| `pypi.org` | ✅ reachable (200) | `curl -I` |
| `download.pytorch.org` | ❌ does **not** resolve | pip DNS failure |
| `dcapswoz.ict.usc.edu` | ✅ reachable (200) — **datasets can be staged on-cluster** | `curl -I` |
| GitHub | ✅ `git clone` / `git pull` work | done |
| Login nodes | jobs are killed; **"user will be disabled automatically"** | banner |

### Two rules that follow

1. **Nothing heavy on a login node.** `git`, `module`, editing, a single `curl` — fine.
   Installing torch, downloading datasets, extracting features, training — all `sbatch`.
   A 2.5 GB pip install *will* be killed and take your SSH session with it.
2. **Every job needs `-A nsmexternal`.** Always submit via `scripts/param/submit.sh`, which
   injects it. Never call `sbatch` directly.

---

## One-time setup

### 0. SSH keepalive — 💻 **MacBook**

```bash
cat >> ~/.ssh/config <<'EOF'

Host param
    HostName paramutkarsh.cdac.in
    User basavarajh
    Port 4422
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes
EOF
```

Then connect with `ssh param`.

### 1. Clone — 🖥 **PARAM**

```bash
cd ~ && git clone git@github.com:AdiiPrabhu/dsctm.git
cd ~/dsctm && git checkout param-main
```

Already cloned? `cd ~/dsctm && git pull`

### 2. Shell profile — 🖥 **PARAM**

```bash
cat >> ~/.bashrc <<'EOF'

# ---- D-MSTCN ----
export DSCTM_ACCOUNT=nsmexternal
export DSCTM_SCRATCH=/scratch/$USER/dsctm
export DSCTM_DATA_ROOT=$DSCTM_SCRATCH/datasets
export DSCTM_RESULTS_ROOT=$HOME/dsctm/results/param_utkarsh_authoritative
export PYTHONPATH=$HOME/dsctm/codex/dsctm/src:$HOME/dsctm/codex/dsctm
module load anaconda3/anaconda3 cuda/11.8 2>/dev/null
[ -d "$HOME/.conda/envs/dsctm" ] && conda activate "$HOME/.conda/envs/dsctm" 2>/dev/null
EOF
source ~/.bashrc
mkdir -p "$DSCTM_SCRATCH" "$DSCTM_DATA_ROOT" "$DSCTM_RESULTS_ROOT"
```

> **Check before every interactive command:** `which python`
> `…/.conda/envs/dsctm/bin/python` = good · `/usr/bin/python` = Python **2.7**, and every
> script will fail with `SyntaxError`.

### 3. Build the environment — as a JOB, ~20 min

```bash
cd ~/dsctm/codex/dsctm && mkdir -p logs
bash scripts/param/submit.sh install_env.sbatch
squeue --me
```

Safe to disconnect. When it finishes:

```bash
tail -40 logs/dsctm-install.*.out
```

Expect `torch : 2.1.2+cu121`, `nccl : True`, and preflight with **0 hard failures**.

### 4. Verify

```bash
source ~/.bashrc
cd ~/dsctm/codex/dsctm
python scripts/param/preflight.py
```

Dataset warnings are expected here. **Any hard failure must be fixed before continuing.**

---

## Campaign

### 5. Smoke test — `debug`, ~20 min  ← DO NOT SKIP

```bash
bash scripts/param/submit.sh --debug 2gpu_ddp_smoke.sbatch
squeue --me
tail -f logs/dsctm-ddp-smoke.*.out
```

Validates both V100s, NCCL, fp16 and the distributed suite on real GPUs. **Closes Gate 3.**
`--debug` uses the separate 1 h pool so you don't queue behind production work.

### 6. Memory probe — `gpu`, ~1 h  ← SETS EVERY BATCH SIZE

```bash
bash scripts/param/submit.sh memory_probe.sbatch
# when done:
python -m json.tool $DSCTM_RESULTS_ROOT/systems/memory_probe_*.json | tail -40
echo 'export DSCTM_BATCH_SIZE=<measured ceiling>' >> ~/.bashrc && source ~/.bashrc
```

The brief assumed 32 GB V100s; these are **16 GB**. Until this runs, every batch size is an
assumption. Closes **B-008**.

### 7. Stage datasets — `standard`, hours

```bash
export KAGGLE_API_TOKEN=<your token>      # StudentLife only
bash scripts/param/submit.sh stage_datasets.sbatch
```

The USC host is reachable, so this works on-cluster — no local download, no rsync. The job
re-checks egress on the compute node and aborts with instructions if that node is blocked.

**Expire the Kaggle token afterwards.**

### 8. Extract features — `standard`, 2–6 h

```bash
bash scripts/param/submit.sh extract_features.sbatch
```

### 9. Monitoring — optional, background

```bash
bash scripts/param/submit.sh monitor.sbatch
# from the MacBook:
scp -P 4422 basavarajh@paramutkarsh.cdac.in:~/dsctm/artifacts/monitoring/dashboard.html .
open dashboard.html
```

### 10. Dry run — `gpu`, 1–4 h  ← BEFORE THE 294

```bash
bash scripts/param/submit.sh 1task_dryrun.sbatch
# when done:
python scripts/param/calibrate.py --extrapolate \
  $DSCTM_RESULTS_ROOT/calibration_<JOBID>.json
```

Five real tasks, one per family. **Decide here.** If the measured extrapolation greatly
exceeds the ~91 GPU-hour estimate, cut scope before spending the allocation.

### 11. The campaign — `gpu`, days

```bash
python scripts/param/plan.py          # verify --array bounds; never type them
bash scripts/param/submit.sh tuning_array.sbatch      # 48 tasks
#   wait for completion, then:
bash scripts/param/submit.sh seeds_array.sbatch       # 60 tasks
bash scripts/param/submit.sh ablation_array.sbatch    # 78 tasks
```

Order matters — confirmation seeds use configurations frozen by tuning.

### 12. Systems — 2 nodes

```bash
DSCTM_SCALING_MODE=strong bash scripts/param/submit.sh scaling.sbatch
DSCTM_SCALING_MODE=weak   bash scripts/param/submit.sh scaling.sbatch
bash scripts/param/submit.sh 2node_4gpu_ddp.sbatch
```

9 GPU nodes = 18 V100s. 2 nodes = 22 %, 4 nodes = 44 %. The manuscript's N=16 would be
89 % — not schedulable, which is the honest answer to tracker T2-07.

### 13. Evidence

```bash
python scripts/param/audit_campaign.py --all --aggregate --out audit.json
python scripts/param/build_evidence.py --out artifacts/final
```

Fail-closed: a family is admitted whole or not at all. Admitting nothing is the pipeline
working, not a bug.

---

## Two open decisions — yours, not the code's

**B-015 — StudentLife windowing.** MSB's receptive field is 481 steps and LSB's is 1921,
but StudentLife windows are **60** steps. The medium and long branches cannot observe their
claimed timescales on that corpus. Re-window, restrict the multi-scale claim to DAIC-WOZ,
or report it as a limitation. **Decide before step 11** — it changes the ablation set.

**Corpus identity.** DAIC-WOZ (189) or E-DAIC (275)? The repo ships E-DAIC splits; the paper
cites DAIC-WOZ. Determines what step 11 runs.

---

## Reference

```bash
squeue --me                    # my jobs
squeue --start                 # estimated start times
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <id>
sinfo -p gpu
which python && python -V      # BEFORE any interactive command
```

| Symptom | Cause | Fix |
|---|---|---|
| `SyntaxError: invalid syntax` on `: list[dict]` | `python` is system 2.7 | `source ~/.bashrc` |
| every package `MISSING` | env half-built by a killed install | re-run step 3 |
| SSH drops mid-command | login-node limit killed your process | use `sbatch`, never install interactively |
| `Invalid account` | `-A` missing | use `submit.sh`, not `sbatch` |
| `Invalid partition: cpu` | stale checkout | `git pull` |
| Job pends forever | queue depth | `squeue --start`; try `--debug`; fewer nodes |
| `IndexError: array index N outside family` | sbatch `--array` disagrees with the plan | `python scripts/param/plan.py --sbatch-array <family>` |

| | Directory |
|---|---|
| 💻 MacBook | `~/Documents/phd/DSTCM_Resubmission/resubmit/dsctm/codex/dsctm` |
| 🖥 PARAM | `~/dsctm/codex/dsctm` |
| 🖥 PARAM data | `/scratch/basavarajh/dsctm/datasets` |
