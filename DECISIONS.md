# DECISIONS

Append-only record of engineering decisions, with rationale and the evidence behind each.
Newest last. A decision is only reversed by a superseding numbered entry.

---

## D-001 · `codex/dsctm/` is the sole foundation; `claude/dsctm/` is frozen evidence

**Date:** 2026-07-26 (Gate 0) · **Status:** ACTIVE

Identification by source inspection, not directory name. All eight markers present in `codex/`
and absent from `claude/`: `models/timesnet.py` (THUML port pinned to `4e938a1`),
`experiments/fair_tuning.py`, `experiments/result_audit.py`, `scripts/audit_exp41_corrected.py`,
mask-aware DAIC handling (`lengths` in `contract.py` → mask in `_make_loader` → `Head(H, mask)` →
`pack_padded_sequence` / `src_key_padding_mask`), causal `_ffill`, a registry actually wired into
five experiment modules, and 31 tests. Full evidence: `artifacts/gate0/REPOSITORY_DISCOVERY.md` §3.

No file from `claude/` will be imported, copied or merged. `claude/` is tagged `claude-archived`
and retained read-only. Its defects (padding-unaware normalization and pooling, unmasked LSTM and
Transformer, backward-fill leakage in `_ffill`, `round(mean(y))` multiclass stratification,
repeated test-set evaluation, placeholder TimesNet, no fair tuning) are the reason.

---

## D-002 · Reuse `source/multi_gpu_validation/` as a design donor, not an execution path

**Date:** 2026-07-26 (Gate 0) · **Status:** ACTIVE

A third implementation exists that neither review document mentions:
`source/` (== `reviewer-package/code/` except `README.md`), package `dmstcn`, 851 LOC, containing
a **real, well-written DDP validation harness** that has never been executed.

**Decision:** port its primitives into `codex/dsctm/src/dsctm/distributed/`, retargeted at the
Codex `dsctm.models.DMSTCN`. Do not execute it, do not depend on it.

**Why this does not violate D-001.** `source/` is an independent third tree that shares no code
with `claude/`. The prohibition is on Claude's *pipeline* (trainer, loaders, splits, evaluation),
none of which is involved here.

**Adopted patterns:** `set_device` before `init_process_group`; fail-fast on missing
`RANK`/`LOCAL_RANK`/`WORLD_SIZE`; `LOCAL_RANK >= device_count` guard; `barrier()` then
`destroy_process_group()`; cross-rank state-digest equality via `all_gather_object`;
`all_reduce_mean`; atomic `.tmp`+`replace` JSON writes; hardware/git/CUDA metadata capture;
DDP-vs-single-process one-step parity design; strong/weak scaling separation; checkpoint-resume
equivalence design; rank-0-only artifact writes; failure appended to `failures.jsonl` before
re-raise.

**Explicitly NOT adopted:** `local_shard()` — manual by-rank slicing of a global batch. Correct
for a synthetic parity probe, wrong as a training data path. Gate 2 uses `DistributedSampler` for
training and a **non-padding** partition for evaluation.

**Also carried over:** the harness has no data, no sampler, no evaluation, no metrics, no AMP and
no SLURM. Those are net-new work in Gate 2 and Gate 4.

---

## D-003 · `baseline-03cc9ec` is not created; truthful substitutes are used

**Date:** 2026-07-26 (Gate 0) · **Status:** ACTIVE

`git cat-file -t 03cc9ec` → `fatal: Not a valid object name`. The repository has a single squashed
root commit `0993ed0`; the upstream branch history was not carried into the flattened monorepo.

Creating a tag named `baseline-03cc9ec` would assert an ancestry this repository cannot
substantiate. Instead: `baseline-flattened`, `codex-single-gpu-audit`, `claude-archived`,
`source-ddp-harness` — all annotated with their path scope, all pointing at `52ad6b1`.

Provenance from this point is established by `baseline-flattened` plus the per-file SHA-256
inventory in `artifacts/gate0/FILE_INVENTORY.csv` (212 files).

---

## D-004 · Repository layout for PARAM work

**Date:** 2026-07-26 (Gate 0) · **Status:** ACTIVE

| Location | Contents |
|---|---|
| `codex/dsctm/src/dsctm/` | All library code, including the new `distributed/` package |
| `codex/dsctm/scripts/param/` | SLURM and `torchrun` launchers |
| `codex/dsctm/tests/` | All tests, including distributed tests |
| `artifacts/gate<N>/` (repo root) | Per-gate governance artifacts |
| `results/local_non_authoritative/` | Non-PARAM output |
| `results/param_utkarsh_authoritative/` | PARAM output — the only evidence root |
| `STATUS.md`, `DECISIONS.md`, `BLOCKERS.md`, `EXPERIMENT_LEDGER.md` (repo root) | Campaign governance |

Rationale: library code stays inside the package it belongs to, so `PYTHONPATH=src` keeps working
and the package remains independently installable. Campaign governance sits at repo root because
it spans all four trees.

---

## D-005 · Gloo/CPU multi-process testing is the local proxy; it never satisfies a hardware gate

**Date:** 2026-07-26 (Gate 0) · **Status:** ACTIVE

NCCL is unavailable locally; gloo is available. All distributed *logic* (sampler behaviour,
prediction gathering, duplicate detection, early-stop broadcast, rank-0 write enforcement,
checkpoint resume, failure propagation) will be tested with gloo at `world_size` 2–4 on CPU.

Such passes are recorded as **`LOGIC-VERIFIED (CPU/gloo)`**, never as gate PASS where the gate
requires hardware. Gates 3 and 7 additionally require NCCL on two V100s and are re-run there.
fp16 `autocast` + `GradScaler` numerics are explicitly **not** validated by any CPU run — CPU fp16
autocast has different coverage and different overflow behaviour from sm_70.

---

## D-006 · CSAG: preserve the manuscript-faithful variant, add the nonlinear one beside it

**Date:** 2026-07-26 (Gate 0, planned for Gate 1) · **Status:** PLANNED

Manuscript Eq. (3)–(4) compose two affine maps with no intervening nonlinearity, so
`W_α W_z` collapses algebraically to a single linear map `R^{3D} → R^3`. The current Codex
implementation is faithful to the paper and will be **preserved unchanged** under the explicit
name `linear_csag`.

A separate `nonlinear_csag` variant with a declared activation between the two projections will be
added as a *new* `csag_mode`, never as a silent replacement. Both appear in the Gate 6 fusion
ablation so the manuscript can state what the nonlinearity is worth. Existing configs that do not
name a mode continue to resolve to the faithful variant.

---

## D-007 · Two-V100 nodes cannot host the paper's one-branch-per-GPU SAP; DDP is the control

**Date:** 2026-07-26 (Gate 0) · **Status:** ACTIVE

The manuscript's SAP places three branches on three nodes plus an aggregator. A PARAM node has
**two** V100s, so the paper's Fig. 3 topology needs a minimum of 4 ranks = 2 nodes.

Ordering is therefore fixed: full-model DDP first (Gate 2/3), benchmarked as the control
(Gate 7), and only then SAP (Gate 8) and TCP (Gate 9). No SAP or TCP claim will be made before
the DDP control exists, because "SAP is faster/better" is meaningless without it.

Corollary: the manuscript's `N = 16` on an 8-GPU server (tracker T2-07) cannot be reproduced as
described and must be restated in terms of ranks actually executed.

---

## D-008 · Prior results are treated as unverifiable, not merely non-authoritative

**Date:** 2026-07-26 (Gate 0) · **Status:** ACTIVE

The brief instructs that RTX 4060 Ti results are non-authoritative debugging evidence. Discovery
found something stronger: **no raw artifact from that campaign exists in this repository at all**
— no JSON, no cache, no registry, no figure, no log. Every number in the ledgers is narrative
text with no backing file, and every quoted SHA-256 is uncheckable.

Three findings survive because they are properties of code or arithmetic, re-verified at Gate 0:
receptive fields 61/481/1921; per-subject adapter cost `d_s` = 8; two-sided exact Wilcoxon with
n = 5 cannot reach p < 0.05. **These are the only prior findings citable to reviewers today.**

Register: `artifacts/gate0/quarantined_claims.csv` (27 claims).
