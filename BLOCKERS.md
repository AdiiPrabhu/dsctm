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
