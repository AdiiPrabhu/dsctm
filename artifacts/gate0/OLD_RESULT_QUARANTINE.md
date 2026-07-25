# Gate 0 — Old Result Quarantine

Generated: 2026-07-26 · Branch `param-main` · Baseline tag `baseline-flattened`

---

## 1. Headline finding: there are no raw results to quarantine

An exhaustive search of the repository for run outputs returned **nothing**:

```bash
find . -not -path "./.git/*" \( -name "*.npz" -o -name "*.parquet" -o -name "*.pt" -o -name "*.ckpt" \)
find claude codex -name "*.json"
```

Both return zero files. Specifically, **every artifact cited by the narrative ledgers is absent**:

| Cited artifact | Cited in | Present? |
|---|---|---|
| `artifacts/resubmission/gate0/gate0_results.json` | codex `METRICS.md` | ❌ |
| `artifacts/resubmission/phase4/studentlife_headline_corrected.json` | codex `STATUS.md` (SHA-256 `abf7079f…f8a3a`) | ❌ |
| `artifacts/resubmission/phase4/studentlife_headline_corrected_audit.json` | codex `STATUS.md` | ❌ |
| `artifacts/resubmission/phase4/daicwoz_headline_egemaps88_masked.json` | codex `METRICS.md` (SHA-256 `b0f0427d…a74f`) | ❌ |
| `artifacts/resubmission/phase4/daic_headline.json` | claude `HANDOFF.md`, codex `METRICS.md` | ❌ |
| `artifacts/resubmission/phase4/daic_headline_egemaps88.json` | claude `HANDOFF.md` | ❌ |
| `artifacts/resubmission/phase4/daicwoz_headline_egemaps88.json` | claude `HANDOFF.md` | ❌ |
| `artifacts/resubmission/systems/single_device_profile.json` | codex `METRICS.md` (SHA-256 `1264699f…2e91`) | ❌ |
| `artifacts/cache/studentlife_causal_ffill_v2.npz` (hash `a9cbaa3a22c2bf4e`) | codex `STATUS.md` | ❌ |
| `artifacts/cache/daic_egemaps88/`, `daicwoz_egemaps88/` | both | ❌ |
| `artifacts/resubmission/runs/` (immutable registry) | codex `STATUS.md` | ❌ (directory does not exist) |
| `artifacts/resubmission/figures/` (reliability plots) | codex `METRICS.md` | ❌ |
| `SUMMARY.md` (cross-corpus table) | claude `HANDOFF.md` | ❌ |
| `artifacts/exp42_5seed.log`, `exp42_balanced.log` | claude `HANDOFF.md` | ❌ |

The only files under `claude/artifacts/` and `codex/artifacts/` are **Gate-P planning documents**
(preflight report, risk register, claim registry, campaign manifest, compute plan, reviewer map)
plus, in Codex, `MATHEMATICAL_FORMULATION.md` and two reproducibility-package scripts. There is
not a single measured number backed by a file.

### 1.1 What this means

The prior results are not merely *non-authoritative* (as the task brief already stipulates) —
they are **unverifiable**. No SHA-256 in either ledger can be checked. No metric can be
recomputed. No prediction file exists to re-derive a confidence interval from.

This **strengthens** the mandated rerun policy rather than complicating it: there is no
temptation to reuse old numbers because there is nothing to reuse. It also means the
`artifacts/resubmission/reviewer_to_experiment_map.csv` in both trees, and every "reproduced" /
"audited" state label in `codex/dsctm/METRICS.md`, are **claims without evidence in this
repository**.

### 1.2 Consequence for the response letter

Any reviewer response that cites a prior corrected result must either (a) be regenerated from a
fresh PARAM run, or (b) be dropped. Statements such as *"the confirmatory rerun was repeated from
committed code `07a78a1` with identical values; its JSON SHA-256 is `b0f0427d…a74f`"* cannot be
substantiated from this repository and **must not be repeated to reviewers** unless the artifact
is recovered from the original machine.

---

## 2. Result roots established

```
results/
├── local_non_authoritative/     # anything not produced on PARAM Utkarsh
│   └── README.md                # guard + policy
└── param_utkarsh_authoritative/ # the ONLY source for manuscript tables/plots
    └── README.md                # guard + admission policy
```

Both roots carry a `README.md` guard stating the policy. Gate 12 evidence generation reads
**exclusively** from `results/param_utkarsh_authoritative/`.

---

## 3. Quarantine register — narrative numeric claims

No files were deleted or moved. The documents below remain in place inside the tagged trees
(`claude-archived`, `codex-single-gpu-audit`) as audit evidence. They are hereby registered as
**QUARANTINED: narrative claim, no backing artifact**.

| Document | Path | Class |
|---|---|---|
| Claude campaign log | `claude/dsctm/HANDOFF.md` | Quarantined narrative |
| Claude observations | `claude/dsctm/OBSERVATIONS.md` | Quarantined narrative |
| Codex campaign status | `codex/dsctm/STATUS.md` | Quarantined narrative |
| Codex metrics ledger | `codex/dsctm/METRICS.md` | Quarantined narrative |
| Codex observations | `codex/dsctm/OBSERVATIONS.md` | Quarantined narrative |
| Codex campaign status copy | `codex/artifacts/resubmission/STATUS.md` | Quarantined narrative |
| Codex metrics log copy | `codex/artifacts/resubmission/METRICS_LOG.md` | Quarantined narrative |
| Codex run index | `codex/artifacts/resubmission/runs.csv` | Quarantined narrative |
| Reviewer→experiment maps (both) | `*/artifacts/resubmission/reviewer_to_experiment_map.csv` | Stale Gate-P preflight, superseded |
| Comparative audits | `claudereview.md`, `codexreview.md` | Retained as **valid** code review; their *code* findings are verifiable, their *result* citations are not |

### 3.1 Specific numeric claims now quarantined

All values below were produced on a **single RTX 4060 Ti** and none has a backing artifact.
Machine-readable register: `artifacts/gate0/quarantined_claims.csv`.

| Claim | Value | Source | Status |
|---|---|---|---|
| StudentLife EXP-4.1 corrected, D-MSTCN pooled macro-F1 | 0.3428 ± 0.0067, rank 4/6 | codex `STATUS.md` | Unverifiable |
| StudentLife corrected, transformer | 0.3675 ± 0.0047 | codex `STATUS.md` | Unverifiable |
| StudentLife corrected, itransformer | 0.3612 ± 0.0098 | codex `STATUS.md` | Unverifiable |
| StudentLife corrected, timesnet | 0.3493 ± 0.0073 | codex `STATUS.md` | Unverifiable |
| StudentLife corrected, temporal-cnn | 0.3243 ± 0.0077 | codex `STATUS.md` | Unverifiable |
| StudentLife corrected, lstm | 0.2970 ± 0.0041 | codex `STATUS.md` | Unverifiable |
| StudentLife split hash | `6208d08f0b8db52b` | codex `STATUS.md` | Unverifiable |
| StudentLife v2 cache data hash | `a9cbaa3a22c2bf4e` | codex `STATUS.md` | Unverifiable |
| StudentLife quarantined v1 cache hash | `62de62987570bc40` | codex `METRICS.md` | Unverifiable |
| DAIC-WOZ 88-dim masked, D-MSTCN macro-F1 | 0.4818, rank 3/6 | codex `METRICS.md` | Unverifiable |
| DAIC-WOZ 88-dim masked, iTransformer | 0.5090 | codex `METRICS.md` | Unverifiable |
| DAIC-WOZ 88-dim masked, LSTM | 0.4841 | codex `METRICS.md` | Unverifiable |
| DAIC-WOZ 88-dim unmasked, D-MSTCN | 0.4854 ± 0.0307, rank 2/6 | claude `HANDOFF.md` | Unverifiable |
| E-DAIC 23-dim, D-MSTCN | 0.5529 ± 0.0918, rank 2/6 | claude `HANDOFF.md` | Unverifiable |
| E-DAIC 88-dim, D-MSTCN | 0.5222, rank 3/6 | claude `HANDOFF.md` | Unverifiable |
| StudentLife EXP-4.1 pre-correction, D-MSTCN | 0.3233, rank 4/6 | claude `HANDOFF.md` | Unverifiable **and** leakage-affected |
| Single-device StudentLife inference | 1.427 ms median, 22,418 samples/s, 27.9 MiB peak | codex `METRICS.md` | Unverifiable |
| Single-device DAIC-WOZ inference | 7.545 ms median, 1,060 samples/s, 130.2 MiB peak | codex `METRICS.md` | Unverifiable |
| StudentLife cohort | 46 subjects, 2160 windows, classes 578/973/609 | both | Unverifiable — **re-derive on PARAM** |
| E-DAIC cohort | 275 sessions, official 163/56/56, 209 neg / 66 pos | both | Unverifiable — **re-derive on PARAM** |
| DAIC-WOZ cohort | 188/189 sessions (440 source-corrupt), 107/34/47 | claude `HANDOFF.md` | Partially checkable — split CSVs **are** in `reviewer-package/data/` |
| DAIC session lengths | median 1821, min 830, max capped 2000 | both | Unverifiable — re-derive |
| StudentLife sensor missingness | ≈ 0.61 | both | Unverifiable — re-derive |
| Total D-MSTCN parameters | ≈ 1.36 M | both | **Recomputable locally** — will be re-derived in Gate 1 |

### 3.2 Claims that survive quarantine

Three classes of prior claim remain valid because they are **properties of code**, reproducible
from source at any time, and were re-verified by the Gate 0 test run:

1. **Receptive fields 61 / 481 / 1921** — re-derived by `test_measured_rf_matches_two_conv_formula`
   at Gate 0 on this machine. Manuscript's 47 / 383 / 1535 remain wrong.
2. **Per-subject adapter cost = `d_s` = 8, not 2D = 256** — a structural property of
   `FiLMAdapter`. Gate 1 adds an explicit assertion (currently only asserted in a report string).
3. **Two-sided exact Wilcoxon with n=5 cannot reach p<0.05 (min p = 0.0625)** — re-verified by
   `test_wilcoxon_n5_two_sided_cannot_reach_significance`. This is arithmetic, not a measurement.

These three are the only prior findings that may be cited to reviewers today.

---

## 4. Anti-confusion controls

| Control | Implementation | Status |
|---|---|---|
| Physically separate result roots | `results/local_non_authoritative/`, `results/param_utkarsh_authoritative/` | ✅ created |
| Written guard in each root | `README.md` in both | ✅ created |
| No prior artifact can be mistaken for a PARAM artifact | Vacuously satisfied — zero prior artifacts exist | ✅ |
| Narrative ledgers registered as quarantined | This document + `quarantined_claims.csv` | ✅ |
| PARAM runs carry mandatory provenance | Gate 4 run-directory contract (`git.json`, `slurm.json`, `hardware.json`, `receipt.sha256`, …) | ⏳ Gate 4 |
| Manuscript tables generated only from admitted PARAM runs | Gate 12 generator reads one root only | ⏳ Gate 12 |
| Prior trees immutable | Git tags `claude-archived`, `codex-single-gpu-audit`, `source-ddp-harness` | ✅ created |

---

## 5. Verdict

**Gate 0 quarantine condition: PASS.**

Old results cannot be confused with PARAM results, for the strongest possible reason: **no old
raw result exists anywhere in this repository.** All prior numbers now live only as registered,
clearly-labelled narrative claims inside tagged evidence trees, and every one of them is marked
unverifiable.
