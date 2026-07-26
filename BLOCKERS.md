# BLOCKERS

Open external blockers preventing gate completion. Each entry states exactly what is blocked,
what is needed to unblock, and what can proceed meanwhile.

Last updated: 2026-07-26 (Gate 0)

---

## B-001 · PARAM execution is remote — **DEPLOYMENT MODEL, not a blocker**

**Status changed 2026-07-26** after the author confirmed PARAM Utkarsh access. This is no longer
an access blocker; it is a two-environment workflow.

```
  THIS MACHINE (macOS/CPU)                 GITHUB                PARAM UTKARSH (SLURM, 2×V100)
  ────────────────────────                 ──────                ──────────────────────────────
  implement + CPU/gloo verify   ──push──►  param-main  ──pull──► env bootstrap, dataset staging,
  SLURM scripts, env bootstrap                                   sbatch, authoritative runs
                                                       ◄─push──  results/param_utkarsh_authoritative/
```

**Consequences for how the code must be written.** Because the author executes on PARAM and the
implementer cannot, every script must be **self-bootstrapping and self-checking**:

1. `scripts/param/env.sh` creates the venv and installs pinned deps; it must not assume a
   pre-existing environment.
2. `scripts/param/preflight.py` must verify — and refuse to proceed on failure — CUDA
   availability, `device_count`, compute capability **sm_70**, NCCL availability, PyTorch build,
   Python version, disk space, and dataset hashes.
3. Every failure mode must produce an actionable message, not a traceback, because the
   implementer will not be at the terminal.
4. Nothing may silently degrade. A missing `thop` currently turns FLOPs into the string
   `"unavailable"`; on PARAM that must be a hard error or an explicit recorded skip.

**Local verification remains bounded.** CPU/gloo passes are recorded as
`LOGIC-VERIFIED (CPU/gloo)` and never satisfy a hardware gate. Gates 3 and 7 must be re-run on
PARAM before any gate is marked PASS.

**Still required from the author** (needed to write Gate 4 correctly):

| Item | Why |
|---|---|
| SLURM partition/queue names and account/project code | `#SBATCH --partition` / `--account` cannot be guessed |
| Module system commands (e.g. `module load python/3.10 cuda/12.4`) | Determines `env.sh` |
| Whether login **and compute** nodes have internet egress | Many HPC centres block egress from compute nodes; if so, dataset pulls and `pip install` must happen on the login node in a staging step |
| Scratch / work filesystem path and quota | Datasets are ~90 GB+ for DAIC-WOZ audio alone |
| Max wall-clock per job and max concurrent array tasks | Sizes the SLURM arrays |
| Python version available | B-004 depends on it |

**What proceeds meanwhile.** Everything that does not require a GPU:

| Can be done now | Cannot be done now |
|---|---|
| Implement `src/dsctm/distributed/*` in full | Any NCCL collective |
| Multi-process DDP tests on **gloo/CPU** (`world_size` 2–4) | fp16 `autocast` + `GradScaler` numerics (CPU fp16 is not representative) |
| Sampler, gather, dedup, early-stop-broadcast, checkpoint-resume logic tests | Any throughput, latency, memory or scaling number |
| SAP placement, activation transfer, autograd-aware collectives (gloo) | Any V100-specific behaviour |
| TCP protocol implementation and state-machine tests | Straggler / bandwidth / RTT experiments |
| Write and lint every SLURM script | Execute any SLURM script |
| Result-auditor, table/figure generators | Admit any real result |

**Policy.** Gloo/CPU passes are recorded as `LOGIC-VERIFIED (CPU/gloo)`. They are **not**
sufficient for a gate to be marked PASS where the gate's definition requires hardware. No gate
from 3 onward may be marked PASS until re-run on PARAM.

---

## B-002 · Datasets staged on PARAM, not here — **AUTHOR-OWNED, needs links recorded**

**Status changed 2026-07-26.** The author holds the dataset links and will pull them directly on
PARAM. This is no longer an acquisition blocker; what remains is that the **exact sources must be
recorded in-repo** so the staging step is reproducible and hashable rather than manual.

**Needed:** the resolvable URLs (or the exact `wget`/`rsync` commands) for StudentLife, DAIC-WOZ
and E-DAIC, so `scripts/param/stage_datasets.sh` can be written with them pinned, and so
`dataset_hashes.json` in every run directory means something. Until those are in the repo,
Gate 5 cannot be reproduced by anyone but the author.

**Blocks (reduced):** Gate 5 and Gate 6 execution; the data-dependent half of Gate 1 is now
partly satisfied — see Gate 1 finding F1-1, which verified the E-DAIC split files that *are*
present in `reviewer-package/data/`.

**Detail.** `.gitignore` excludes `dataset/`. No raw data, no `.npz` caches, no extracted
eGeMAPS features exist in the repository. `opensmile` is not installed.

Present and usable: DAIC-WOZ official split and PHQ-8 label CSVs under `reviewer-package/data/`
(`train_split.csv`, `dev_split.csv`, `test_split.csv`, `Detailed_PHQ8_Labels.csv`,
`metadata_mapped.csv`, `detailed_lables.csv`, `PROVENANCE.md`).

**Still to stage on PARAM:**

| Dataset | Needed | Note |
|---|---|---|
| StudentLife | Full sensing + EMA release staged on PARAM | Prior campaign reported 46 usable subjects vs manuscript's 48 — must be re-derived, not assumed |
| DAIC-WOZ (AVEC2017) | 189 `*_P.zip` (~85.6 GB) | Prior campaign reported session 440 corrupt at source; re-verify independently |
| E-DAIC (AVEC2019) | Official release | Confirm test-label usage is authorized |
| SEED | Not present anywhere; never used by either implementation | Decide whether the SEED transfer experiment is in scope at all |
| `opensmile` ≥ 2.6.0 | pip install on PARAM | Required for 88-dim eGeMAPSv02 functionals |

**Open question for the author (blocking, not resolvable from code):** which corpus produced the
manuscript's headline — classic DAIC-WOZ (189, 107/82) or E-DAIC (274, official splits)? The two
give different numbers and the manuscript cites the former while the prior campaign found the
latter on disk.

---

## B-003 · No editable manuscript source and no original decision letter

**Blocks:** every W5-\*, F6-\*, S7-\* tracker row (32 of 90), and Gate 11's manuscript-side output.

**Detail.** Only `dsctm_original.pdf` (compiled, 15 pp.) is available. There is no `.tex`/`.docx`,
no `.bib`, and no raw reviewer report — only the derived tracker CSV.

**Needed:** the LaTeX or Word source, the bibliography, and the editorial decision letter with
the seven verbatim reviewer reports.

**Proceeds meanwhile:** all engineering evidence that those rows depend on.

---

## B-004 · `source/` DDP harness requires Python ≥ 3.10

**Blocks:** direct execution of `source/multi_gpu_validation/` on Python 3.9.

**Detail.** `source/src/dmstcn/model.py:132` uses `Tensor | None` in an evaluated annotation
without `from __future__ import annotations` → `TypeError` at import on 3.9.6. The
`reviewer-package/` copy has the same defect, meaning **the package shipped to reviewers does not
import on Python 3.9**.

**Needed:** confirm the Python version available on PARAM. Fix is one line
(`from __future__ import annotations`) and is queued for Gate 2 when harness primitives are
ported. Low severity for the campaign, non-trivial for the reviewer package.

---

## B-005 · Missing optional Python packages

**Blocks:** two named artifacts.

| Package | Blocks | Tracker item |
|---|---|---|
| `thop` | FLOPs in `exp_0_2_params_flops` (silently degrades to `"unavailable"`) | E4-07 |
| `pyarrow` | `predictions.parquet` in the Gate 4 run contract | E4-18 |
| `opensmile` | eGeMAPS extraction | see B-002 |

**Needed:** add all three to the PARAM environment; pin versions in `requirements-lock.txt`.

---

## B-006 · Compute allocation not approved

**Blocks:** Gate 5 onward at full scale.

**Detail.** Gate 5 + 6 + 7 span roughly: 2 datasets × 6 models × 8 tuning trials, then
6 models × 10 confirmation seeds, plus ≥ 14 ablation variants × 5 folds × 3 seeds, plus five
scaling configurations × 2 scaling modes × 5 repetitions. The prior campaign's own note states
the expanded Phase 5 alone grew from 105 to 210 fits and "requires revised compute approval".

**Needed:** node-hour budget, maximum wall-clock per job, queue limits, maximum concurrent array
tasks. A costed compute plan is a Gate 4 deliverable and must be approved before Gate 5 launches.

---

## B-007 · Prior results are unverifiable (informational, not blocking)

**Detail.** Every artifact cited by `codex/dsctm/METRICS.md`, `codex/dsctm/STATUS.md` and
`claude/dsctm/HANDOFF.md` is absent from the repository — no JSON, no `.npz`, no registry, no
figures. See `artifacts/gate0/OLD_RESULT_QUARANTINE.md`.

**Effect.** No prior number may be cited to reviewers. This does not block forward work (the
policy was already to rerun everything on PARAM) but it does mean there is **no fallback** if the
PARAM campaign under-delivers. Recorded so the risk is explicit.

**Optional recovery:** if the original RTX 4060 Ti machine still holds
`artifacts/resubmission/`, archiving it would restore the audit trail behind the ledgers. Worth
attempting before that machine is wiped.

---

## B-008 · PARAM V100s are **16 GB**, not 32 GB — task brief assumption corrected

**Source:** `PARAM_Utkarsh_User_Manual-v3.0-1.pdf` p.10 — *"2*nVidia V100 per node,
GPU Memory = 16 GB HBM2 per nVidia V100."*

The engagement brief specifies "2x NVIDIA V100 SXM2 32GB GPUs per node". The cluster has
**V100 SXM2 16 GB**. Half the assumed memory.

**Why it matters now.** DAIC-WOZ is the heavy case: T = 2000, F = 88, D = 128, three
branches of four residual blocks, activations retained for backward. Activation memory
scales as `batch x T x D x depth`. At 16 GB the workable per-rank batch is materially
smaller than at 32 GB, and `DAIC_CFG` currently sets `batch_size: 8` for a single device.

**Action:** a memory-probe job is now a Gate 3 deliverable — sweep per-rank batch on one
V100 until OOM, record the ceiling, and set the scientific global batch from the measured
ceiling rather than from an assumption. Until that runs, no batch size in any config is
trustworthy for PARAM.

---

## B-009 · The cluster has 20 V100s total — the 16-GPU scaling target is not schedulable

**Source:** manual p.10 — GPU Compute Nodes: **10**, each with 2 V100 → **20 V100s in the
entire machine**, shared by every user. (The architecture diagram on p.11 says 30 GPU
nodes and the Access Guide spec table says 10; CDAC's own documents disagree. The
preflight script queries `sinfo` and records the truth rather than trusting either.)

**Consequence for the Gate 7 scaling matrix:**

| Config | Nodes x GPUs | Share of cluster GPUs | Realistic? |
|---|---|---:|---|
| Single GPU | 1 x 1 | 5 % | ✅ |
| One-node DDP | 1 x 2 | 10 % | ✅ |
| Two-node DDP | 2 x 4 | 20 % | ✅ likely |
| Four-node DDP | 4 x 8 | **40 %** | ⚠️ long queue |
| Eight-node DDP | 8 x 16 | **80 %** | ❌ effectively unschedulable |

**This is also the answer to tracker T2-07.** The manuscript reports N = 16 on an
eight-GPU server and Table 3 reports efficiency at N = 16. Neither the original hardware
nor PARAM can produce a genuine 16-rank measurement. The scaling section must be restated
against what is actually schedulable — realistically 1, 2, 4 and possibly 8 GPUs — and the
node/GPU/rank conflation must be removed rather than carried forward.

**Action:** Gate 7 targets 1 / 2 / 4 GPUs as the committed matrix, with 8 as a
best-effort stretch. 16 is dropped unless CDAC grants a reservation, and its absence is
stated in the paper rather than papered over.

---

## B-010 · CentOS 7.9 (glibc 2.17) may not run modern PyTorch wheels

**Source:** manual p.10 and p.14 — OS CentOS 7.9, SLURM 20.11.8.

Recent PyTorch wheels are built against `manylinux_2_28` (glibc >= 2.28). CentOS 7.9 ships
glibc 2.17. A `pip install torch==2.6.0+cu124` may therefore fail outright or, worse,
install and then fail at import inside a queued job.

Available modules per the Access Guide include `anaconda3/pytorch` and
`python/conda-python/3.7`. Python 3.7 is too old for this codebase's dependency set
(numpy 2.x needs >= 3.9; `StratifiedGroupKFold` needs a recent scikit-learn).

**Action for `scripts/param/env.sh` (Gate 4):** do not assume. Try, in order —
(1) a conda env with Python 3.10 + a pinned torch known to work on glibc 2.17,
(2) the site `anaconda3/pytorch` module. Then have `preflight.py` **hard-fail** on: torch
import, CUDA availability, `device_count`, compute capability sm_70, NCCL availability,
and a live 2-rank NCCL all-reduce. Report versions rather than guessing them.

---

## B-011 · Login nodes enforce CPU/memory limits and kill offending processes

**Source:** manual p.9 — *"there will be a limit on the CPU time that can be used on a
login node by a user and there is a limit/user on the memory as well. If any of these are
exceeded, the job will get terminated."* Access Guide p.7 — *"Please don't run any jobs in
login nodes."*

Dataset staging is ~86 GB for DAIC-WOZ audio alone, plus eGeMAPS extraction over 189
sessions, which is CPU-heavy. Running that on a login node will be killed part-way and
leave a half-extracted cache that looks complete.

**Action:** `scripts/param/stage_datasets.sh` splits into two phases — download on the
login node (I/O-bound, low CPU, resumable with `wget -c`), then feature extraction as a
batch job on the **`cpu`** partition, not the GPU partition. Whether compute nodes have
internet egress is still unconfirmed and determines whether the download itself must also
be a login-node-only step.

---

# Blockers added in Gates 6–12

| ID | Gate | Blocker | Needs |
|---|---|---|---|
| **B-012** | 9, 11 | TCP invariant verified in simulation and 4-rank gloo only; real NCCL asynchrony unmeasured | Gate 10 on 2 PARAM nodes |
| **B-013** | 10 | Four-mode comparison needs ≥ 4 ranks = 2 nodes = 20 % of the cluster. **Gating experiment for every TCP claim in the paper.** | 2-node allocation |
| **B-014** | 11 | Retaining a convergence theorem requires writing one from scratch, about the protocol as implemented (HOLD suspends updates; no cited result models that) | author decision — mathematics, out of scope here |
| **B-015** | 6 | **StudentLife T=60 but MSB RF=481 and LSB RF=1921.** The medium/long branches cannot see their claimed timescales on that corpus. | author decision: re-window, restrict the claim to DAIC-WOZ, or report the limitation |
| **B-016** | 6 | `expanded` LSB has RF 10,881 > DAIC-WOZ T=2000; will be padding-dominated | kept as the upper bracket; degeneracy must be reported |
| **B-017** | 7, 10 | Real network impairment (bandwidth/RTT/jitter/loss — tracker **E4-05**) needs `tc netem` + root, unavailable to a normal HPC user | admin-assisted experiment, or withdraw the claim |
| **B-018** | 8 | SAP equivalence verified on gloo/CPU only; NCCL p2p semantics differ | 2-node PARAM run |
| **B-019** | 8 | `replicate_gradients` creates a process group per call — wasteful in a hot loop | cache groups before any timing claim at ws > 4 |
| **B-020** | 9 | `train/tcp.py` (simulator) and `tcp_real.py` now coexist; a future reader could cite the simulator | delete the simulator once Gate 10 output exists |
| **B-021** | 10 | `--full-grid` is 288 cells × 4 modes — a large 2-node allocation | separate compute approval; default is an 8-cell subset |
| **B-022** | 12 | `figures/` is empty — axes are not guessed for data that does not exist | generate once admitted data has a known shape |

## Standing blockers still open

B-002 (datasets to stage) · B-003 (no manuscript source) · B-005 (`thop`, `pyarrow`,
`opensmile` absent locally) · B-006 (compute allocation) · B-008 (**16 GB V100s — every
batch size is an assumption until `memory_probe.sbatch` runs**) · B-009 (**16-GPU target not
schedulable; tracker T2-07 must be restated**) · B-010 (glibc 2.17 vs modern PyTorch wheels)
· B-011 (login-node limits).



---

## Resolutions recorded 2026-07-26

**B-002 — RESOLVED (credentials + sources).** All three dataset sources are pinned in
`scripts/param/stage_datasets.sh`. Kaggle auth now accepts, in precedence order:
`KAGGLE_API_TOKEN` (env), `KAGGLE_USERNAME`+`KAGGLE_KEY` (env), or `~/.kaggle/kaggle.json`.
When only the single token is present, an ephemeral `KAGGLE_CONFIG_DIR` is created outside
the repository and removed on exit. No credential is written into the repo, echoed to a
log, or committed; `.gitignore` additionally blocks `kaggle.json`, `.env*`, `*_token`.

> **Action for the author:** the token was transmitted in plaintext and is in shell history.
> Expire it at kaggle.com → Settings → API once StudentLife is staged, and issue a fresh one.

Remaining under B-002: actually running the staging on PARAM, and confirming the E-DAIC
feature archives your EULA covers.

**B-006 — SIZED.** `scripts/param/estimate_compute.py` converts the 294-task plan into an
allocation request. A-priori bracket:

| | GPU-hours |
|---|---:|
| Science (294 tasks) | 3.2 – 10.1 |
| Systems experiments | 30 – 60 |
| **Grand total** | **33 – 70** |
| **Request with 30 % contingency** | **~91 GPU-hours** |

Also request: peak concurrency 4 GPUs plus one 2-node reservation; longest job 12 h (limit
is 72 h); ~6 h × 48 cores on the `cpu` partition for eGeMAPS extraction; ~250 GB Lustre
scratch.

The science half is small because the datasets are small — 2,160 StudentLife windows and
189 DAIC-WOZ sessions. **The systems experiments dominate the request**, which is the
correct shape for a paper whose contested claims are about scaling. Dominant uncertainty is
the DAIC-WOZ per-step time; re-run with `--probe` after `memory_probe.sbatch`.

**B-008 — NOT A DECISION.** It is resolved by running `sbatch scripts/param/memory_probe.sbatch`.
The job binary-searches the per-rank batch ceiling on a real 16 GB V100 for every model at
fp32 and fp16, and writes it to `results/param_utkarsh_authoritative/systems/`. Set
`DSCTM_BATCH_SIZE` from the output. Nothing to choose; something to run.


---

## Login-banner corrections (2026-07-26, after first PARAM login)

The MOTD contradicted three documented assumptions. All scripts were corrected.

| | Assumed from the PDFs | **Actual (login banner)** |
|---|---|---|
| Partitions | `standard`, `cpu`, `gpu`, `hm` | **`standard*`, `gpu`, `hm`, `debug`** — **no `cpu` partition exists** |
| `gpu` partition | 10 nodes | **9** (`gpu002-010`); `gpu001` is in `debug` |
| Account | not mentioned in either PDF | **`#SBATCH -A <account>` is required on every job** |
| Login-node policy | "please don't run jobs" | **"user will be disabled automatically"** |

**B-023 — `--partition=cpu` did not exist.** `extract_features.sbatch` would have been
rejected at submit. Corrected to `standard`. Had this not surfaced, the first attempt to
extract features would have failed with an unhelpful scheduler error.

**B-024 — every job requires an account.** Neither the Access Guide nor the User Manual
mentions this; the banner does. All 14 sbatch files now carry a placeholder and
`scripts/param/submit.sh` substitutes `$DSCTM_ACCOUNT` at submit time, so the account is
never committed. Find it with `sacctmgr show associations user=$USER`.

**B-025 — login-node staging is now an account-suspension risk, not a courtesy issue.**
The original plan ran an 86 GB `wget` on the login node. The banner says users are
**disabled automatically** for running jobs there. Staging moved to
`stage_datasets.sbatch` on `standard`, which first verifies egress on the compute node and
aborts with instructions rather than silently producing an empty dataset tree.
`scripts/param/check_egress.sh` (three HEAD requests) determines which path is available.

**B-026 — compute-node egress still unconfirmed.** If compute nodes are blocked and only
the login node has egress, there is no safe automated path: the fallback is `rsync` from a
machine the author controls, or a data-transfer route from CDAC support. Resolved by
running Step 4 of `RUN_ORDER.md`.

**Revised scaling arithmetic.** The `gpu` partition is 9 nodes = **18 V100s**, not 20.
2 nodes = 22 %, 4 nodes = 44 %, and the manuscript's N=16 would be **89 %** of the entire
GPU partition. B-009 stands and is now slightly worse than recorded.


---

## Module survey on PARAM Utkarsh (2026-07-26, login03)

Confirmed available and now the pinned defaults:

| Module | Status |
|---|---|
| `anaconda3/anaconda3` | ✅ the conda base `env.sh` uses |
| `cuda/11.8` | ✅ matches the `torch==2.1.2+cu118` pin exactly |
| `cuda/12.0` | available (cluster default), untested here |
| **`glibc/2.28`** | ✅ **available as a module** |
| `anaconda3/pytorch` | site-built PyTorch; version unverified |
| `python/conda-python/3.7`, `python/3.7.11` | too old for this dependency set |
| `horovod_python/3.9` | present, not used (we use native DDP) |
| `singularity/3.4.1` | present — a container fallback if the conda route fails |

**B-010 — DOWNGRADED from blocker to a documented choice.** CentOS 7.9 ships glibc 2.17,
which is why the default pin is `torch==2.1.2+cu118` (manylinux_2_17). But PARAM also
provides a **`glibc/2.28` module**, so a newer PyTorch is achievable if ever required.
It is deliberately not the default: loading an alternate glibc manipulates the dynamic
linker path at runtime and can break unrelated libraries in ways that surface deep inside a
queued job. The cu118 route needs no such intervention.

Escalation order if the conda build fails: (1) `anaconda3/pytorch` site module,
(2) `glibc/2.28` + a newer wheel, (3) `singularity/3.4.1` container.

**Confirmed environment facts**

| | |
|---|---|
| Account | `nsmexternal` (no partition restriction, QOS `normal`) |
| Scratch | `/scratch` — Lustre, 859 TB total, **548 TB free** |
| Login node | `login03` |


---

## B-026 CONFIRMED — PARAM has NO internet egress (2026-07-26)

`pip install` from `login03` fails at **DNS resolution**, not at the proxy or firewall:

```
NameResolutionError: Failed to resolve 'download.pytorch.org'
([Errno -2] Name or service not known)
```

This is the strongest form of the blocker. It is not "compute nodes are restricted, stage
from login" — **the login node itself cannot resolve external hostnames.**

### What it breaks

| Assumed | Reality |
|---|---|
| `pip install torch` in `env.sh` | ❌ cannot reach PyPI or download.pytorch.org |
| `wget` the DAIC-WOZ / E-DAIC archives (~86 GB) | ❌ cannot reach `dcapswoz.ict.usc.edu` |
| `kaggle datasets download` | ❌ cannot reach kaggle.com |
| `git clone` from GitHub | ❌ unless an internal mirror exists |
| `stage_datasets.sbatch` | correctly aborts on its egress check — working as designed |

### Paths that remain

**Environment** — three options, cheapest first:

1. `DSCTM_USE_SITE_TORCH=1 source scripts/param/env.sh` — use the cluster's
   `anaconda3/pytorch` module directly. No download at all. You inherit the site's Python
   and torch versions, which may be too old.
2. **Offline wheel bundle** (implemented): run `scripts/param/offline_bundle.sh` on a
   machine with internet, `rsync` the ~2.5 GB archive to PARAM, then
   `DSCTM_OFFLINE_WHEELS=~/dsctm_wheels source scripts/param/env.sh`. Wheels are fetched
   for `linux_x86_64 / cp310` regardless of the building host, so a Mac can produce a
   CentOS bundle.
3. Ask CDAC support for an HTTP/HTTPS proxy or a local PyPI mirror. Many national HPC
   facilities have one that is simply not advertised in the MOTD. **Ask before assuming
   options 1–2 are the only routes.**

**Datasets** — no automated path exists. Either:

1. CDAC provides a proxy or a data-transfer node, or
2. the author downloads ~90 GB locally and `rsync`s it:
   `rsync -avP --partial -e 'ssh -p 4422' ./datasets/ $USER@paramutkarsh.cdac.in:$DSCTM_DATA_ROOT/`

**This is now the critical-path blocker for the entire campaign.** Compute is allocated,
the environment has a workaround, but 90 GB of data cannot be moved by any script in this
repository without either a proxy or a long local download plus transfer.

### Recommended action

Raise a ticket with `utkarsh-support@cdac.in` asking specifically for:

* an HTTP/HTTPS proxy for compute or login nodes, or a local PyPI/conda mirror
* whether a data-transfer node with egress exists for staging research datasets
* whether `dcapswoz.ict.usc.edu` can be whitelisted for the duration of the project

The answer determines whether staging takes an afternoon or a week.
