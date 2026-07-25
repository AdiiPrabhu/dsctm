# Gate 0 — Repository Discovery

Generated: 2026-07-26
Branch: `param-main` · HEAD: `52ad6b1714c89d0e543c389004fe41c8c3ef1fc6`
Baseline tag: `baseline-flattened`

---

## 1. Git state at discovery

| Property | Value |
|---|---|
| Repository root | `/Users/adii/Documents/phd/DSTCM_Resubmission/resubmit/dsctm` |
| Remote | `git@github.com:AdiiPrabhu/dsctm.git` |
| Branches before Gate 0 | `main` only (no remote-tracking branches present) |
| Commit history | **one commit**: `0993ed0` "Initial commit: flattened DSCTM resubmission monorepo" |
| Tags before Gate 0 | none |
| Dirty state at discovery | 4 untracked files (tracker CSV, original PDF, `claudereview.md`, `codexreview.md`) |

### 1.1 Finding D0-1 — the requested baseline commit does not exist

The task specification asks for a tag `baseline-03cc9ec`. Verification:

```
$ git cat-file -t 03cc9ec
fatal: Not a valid object name 03cc9ec
```

Commit `03cc9ec` is **not present in this repository's history**. This repository is a
*flattened* monorepo with a single squashed root commit; the upstream `experimentation1` /
`experimentation2` history from which both implementations were derived was not carried over.

**Action taken:** `baseline-03cc9ec` was **not** fabricated. A truthful equivalent was created:

| Tag | Points at | Meaning |
|---|---|---|
| `baseline-flattened` | `52ad6b1` | Pre-PARAM baseline of the whole tree + audit evidence |
| `codex-single-gpu-audit` | `52ad6b1` | Path scope `codex/` — the audited single-GPU foundation |
| `claude-archived` | `52ad6b1` | Path scope `claude/` — read-only evidence, must not be merged |
| `source-ddp-harness` | `52ad6b1` | Path scope `source/`, `reviewer-package/code/` — pre-existing DDP harness |

All four tags reference the same commit because all trees live in that one commit; the tags
carry **path scope** in their annotation messages. The ancestry claim "derived from `03cc9ec`"
is recorded here as *documentation received from the upstream campaign*, not as a verifiable
Git fact in this repository.

**Consequence for reviewers:** no code in this repository can be cryptographically traced to a
pre-flattening ancestor. Provenance from this point forward is established by `baseline-flattened`
and the per-file SHA-256 inventory in `FILE_INVENTORY.csv`.

---

## 2. Finding D0-2 — there are three implementations, not two

The task brief describes two implementations (Claude and Codex). Discovery found **three
distinct Python packages**, plus a fourth copy.

| Tree | Package | Python files | LOC | Contains DDP? | Role assigned |
|---|---|---:|---:|---|---|
| `codex/dsctm/` | `dsctm` | 65 | 5,403 | **No** | **PARAM foundation** |
| `claude/dsctm/` | `dsctm` | 38 | 3,352 | No | Archived evidence — do not merge |
| `source/` | `dmstcn` | 8 | 851 | **Yes** | DDP harness reference |
| `reviewer-package/code/` | `dmstcn` | — | — | Yes | Byte-identical copy of `source/` except `README.md` |

`reviewer-package/code` vs `source/` differ in exactly one file:

```
$ diff -rq source reviewer-package/code
Files source/README.md and reviewer-package/code/README.md differ
```

`reviewer-package/` additionally carries the DAIC-WOZ split CSVs and PHQ-8 label files
(`train_split.csv`, `dev_split.csv`, `test_split.csv`, `Detailed_PHQ8_Labels.csv`,
`metadata_mapped.csv`, `detailed_lables.csv`) plus `data/PROVENANCE.md`. These are the only
dataset-derived files tracked in the repository and are the authority for the official
107 / 34 / 47 DAIC-WOZ split.

---

## 3. Codex foundation identification (by source inspection)

Identification was performed by reading source code, not by trusting directory names. All six
required markers are present in `codex/dsctm/` and **absent** from `claude/dsctm/`.

| Required marker | Present in `codex/` | Present in `claude/` | Verification |
|---|---|---|---|
| `models/timesnet.py` | ✅ 125 LOC | ❌ absent | Faithful THUML port, docstring pins upstream commit `4e938a1767106324dd753b2a44832bf870a0252e` |
| `experiments/fair_tuning.py` | ✅ 136 LOC | ❌ absent | 8 prespecified dev trials per model; `test_access_during_search: False` |
| `experiments/result_audit.py` | ✅ 76 LOC | ❌ absent | Plus `scripts/audit_exp41_corrected.py` (17 LOC) |
| `scripts/audit_exp41_corrected.py` | ✅ | ❌ absent | Fail-closed auditor entry point |
| Masking-aware DAIC handling | ✅ | ❌ | `contract.py` carries `lengths`; `trainer.py::_make_loader` builds `mask`; `blocks.py::Head.forward(H, mask)` does masked pooling; `baselines.py` uses `pack_padded_sequence` + `src_key_padding_mask` |
| Causal StudentLife forward-fill | ✅ | ❌ | `data/studentlife.py::_ffill` docstring: *"The previous implementation replaced a leading missing prefix with the first later observation, which was backward-fill leakage despite the function name."* |
| Immutable run registry | ✅ 229 LOC | ⚠️ 106 LOC, not wired into experiments | Codex `registry.py` exports `write_completed_fit` / `write_failed_fit`, imported by `headline.py`, `ablation.py`, `fair_tuning.py`, `delay_task.py`, `preprocessing.py`. Claude's `registry.py` is imported by **no** experiment module. |
| ~31 tests | ✅ **31 passed** | 11 passed | See `BASELINE_TEST_REPORT.md` |

Secondary confirmations unique to `codex/`:

- `data/contract.py` validates `lengths` in range `[1, T]` (claude's `WindowedDataset` has no `lengths` field at all; it writes an unread `ds._lengths` attribute).
- `data/splits.py` uses `StratifiedGroupKFold` on window labels; claude uses `StratifiedKFold` on `round(mean(y))`.
- `models/dmstcn.py` exposes `film_mode ∈ {subject, global, global_matched}` and `csag_mode ∈ {attention, mean, static}`; claude exposes only `csag_mode ∈ {attention, mean}`.
- `train/trainer.py::train_select_evaluate` restores a frozen best-dev `state_dict` and evaluates test **once**; claude re-evaluates test at every dev improvement.

**Conclusion: the Codex foundation is unambiguously `codex/dsctm/`.**

---

## 4. Finding D0-3 — a real DDP harness already exists in `source/`

`source/multi_gpu_validation/` (889 LOC across 5 files) is a **complete, well-structured DDP
validation harness** that neither review document mentions. It has never been executed:

> `source/multi_gpu_validation/STATUS.md`:
> "Local RTX 4060 Ti execution: intentionally not performed. Physical multi-GPU validation:
> pending rental hardware. Until the rental-machine reports pass, this directory is
> implementation—not experimental evidence."

### What it already implements correctly

| Capability | File | Assessment |
|---|---|---|
| `init_process_group(backend="nccl", init_method="env://")` | `common.py:61` | Correct |
| `torch.cuda.set_device(local_rank)` before init | `common.py:60` | Correct ordering |
| `RANK` / `LOCAL_RANK` / `WORLD_SIZE` discovery with fail-fast on missing | `common.py:44-58` | Correct |
| `LOCAL_RANK >= device_count` guard | `common.py:57` | Correct |
| `barrier()` then `destroy_process_group()` | `common.py:65-68` | Correct |
| Cross-rank state-digest equality assertion via `all_gather_object` | `common.py:98-102` | Reusable primitive |
| `all_reduce_mean` | `common.py:105-108` | Correct |
| Atomic JSON write (`.tmp` + `replace`) | `common.py:155-162` | Reusable |
| Hardware / git / CUDA / cuDNN metadata capture | `common.py:120-152` | Reusable |
| DDP-vs-single-process one-step parity check | `validate.py:86-135` | Directly reusable design |
| Strong- and weak-scaling benchmark separation | `validate.py:187-202` | Correct conceptual split |
| Checkpoint / resume equivalence check | `validate.py:205-259` | Directly reusable design |
| Rank-0-only artifact write | `validate.py:282-285` | Correct |
| Failure recorded to `failures.jsonl` before re-raise | `validate.py:286-298` | Correct |

### What it does not implement (and why it cannot be used as-is)

1. **It targets a different model.** It imports `from dmstcn import DMSTCN, DMSTCNConfig`
   (`source/src/dmstcn/model.py`, 150 LOC) — a minimal reimplementation with no branch
   ablations, no baselines, no `enabled_branches`, no `csag_mode` / `film_mode`. It is *not*
   the Codex `dsctm.models.DMSTCN`.
2. **No real data.** Every batch is `torch.randn` from `make_global_batch`. There is no
   `Dataset`, no `DistributedSampler`, no evaluation path, and therefore no duplicate-sample
   problem to solve — the exact hazard Gate 2 must address.
3. **No evaluation gathering, no metrics.** Nothing computes macro-F1 or gathers predictions.
4. **No AMP.** No `autocast`, no `GradScaler` anywhere.
5. **No SLURM.** `run_matrix.sh` calls `torchrun` directly; there is no `sbatch` wrapper and no
   `MASTER_ADDR` derivation from `scontrol`.
6. **Manual sharding, not a sampler.** `local_shard()` slices a global batch by rank. This is
   correct for a synthetic parity test and *wrong* as a training data path.
7. **It does not import on this Python.** See Finding D0-4.

### Decision recorded

The `source/` harness is adopted as a **design reference and primitive donor** for
`codex/dsctm/src/dsctm/distributed/`, not as an execution path. Reusing it does not violate the
"do not merge Claude pipeline code" rule — it is an independent third tree that shares no code
with `claude/`. Rationale and scope are recorded in `DECISIONS.md` as decision **D-002**.

---

## 5. Finding D0-4 — `source/` requires Python ≥ 3.10 and cannot import here

```
source/src/dmstcn/model.py:132: in DMSTCN
    def forward(self, inputs, subject_ids, mask: Tensor | None = None) -> DMSTCNOutput:
E   TypeError: unsupported operand type(s) for |: 'torch._C._TensorMeta' and 'NoneType'
```

The file uses PEP-604 union syntax in an evaluated annotation position without
`from __future__ import annotations`. It therefore requires Python ≥ 3.10.

The audit environment is **Python 3.9.6**. `codex/` and `claude/` both carry
`from __future__ import annotations` in every module using this syntax, so both import cleanly
on 3.9.

**Implication for PARAM:** the Python version on PARAM Utkarsh must be pinned and recorded
before any run. If PARAM provides Python 3.9, the `source/` harness is unusable as-is; this is
another reason it is being treated as a donor rather than an execution path. Tracked in
`BLOCKERS.md` as **B-004**.

---

## 6. Reviewer tracker and manuscript

| Input | Status |
|---|---|
| `D_MSTCN_IEEE_Access_Resubmission_Tracker - Master Tracker (1).csv` | Present, 90 rows, 20 columns, parsed |
| `dsctm_original.pdf` | Present, 15 pages, read |
| `claudereview.md` | Present (22.8 KB) — comparative audit |
| `codexreview.md` | Present (11.9 KB) — comparative audit with reviewer-compliance matrix |
| Editable manuscript source (LaTeX/Word) | **Absent** — blocks all W5/F6/S7 tracker rows |
| Original editorial decision letter | **Absent** |
| Reviewer reports (raw) | **Absent** — only the derived tracker is available |

Tracker phase distribution: Gate 0 integrity (10), scope decisions (8), theory/algorithm (12),
data/evaluation integrity (10), experiments/benchmarking (18), writing (13), formatting (8),
submission (11). Priority: 39 × P0-Critical, 40 × P1-Major, 10 × P2-Supporting, 1 × P3-Polish.

Both trees ship `artifacts/resubmission/reviewer_to_experiment_map.csv` (90 rows). **These are
stale Gate-P preflight products generated before implementation** — Claude marks 67/90 rows
`blocked`, Codex marks 89/90 `blocked`, and neither reflects the actual code state. They are
retained as evidence but must not be cited as a compliance record. A regenerated
`reviewer_to_artifact_map.csv` is a Gate 12 deliverable.

---

## 7. Dataset availability

| Dataset | Raw data in repo | Split/label files in repo | Status |
|---|---|---|---|
| StudentLife | ❌ (`.gitignore` excludes `dataset/`) | ❌ | Must be re-staged on PARAM |
| DAIC-WOZ | ❌ | ✅ `reviewer-package/data/*.csv` | Audio must be re-staged on PARAM |
| E-DAIC | ❌ | ❌ | Must be re-staged on PARAM |
| SEED | ❌ | ❌ | Not present anywhere; never used by either implementation |

No cached feature files (`.npz`), no extracted features, and no `artifacts/cache/` exist. Every
dataset must be re-acquired and re-hashed on PARAM before Gate 5. Tracked as **B-002**.

---

## 8. Gate 0 verdict

**Status: PASS**, with four findings recorded above.

| Gate 0 pass condition | Result |
|---|---|
| Codex implementation identified unambiguously | ✅ §3, eight independent source markers |
| Baseline test suite passes | ✅ 31/31, see `BASELINE_TEST_REPORT.md` |
| Old results cannot be confused with PARAM results | ✅ see `OLD_RESULT_QUARANTINE.md` — and note there are **no** old raw results to confuse |
| No Claude correctness defect merged into the working tree | ✅ `claude/` untouched; zero imports from `claude/` in `codex/`; verified by inventory |
