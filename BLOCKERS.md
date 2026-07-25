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
