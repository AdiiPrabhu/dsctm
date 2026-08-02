# D-MSTCN Metrics Ledger

This is the human-readable, append-only index of measured experiment outcomes. Raw
JSON, predictions, curves, and run metadata are authoritative; every entry below must
point to them. Failed runs and negative findings remain visible.

## Verification state

- `reproduced`: rerun in this `experimentation2` checkout.
- `audited`: raw artifact and implementation checked, but training not rerun here.
- `imported`: inherited from commit `03cc9ec`; independent audit pending.
- `blocked`: required hardware, rights, or source input is unavailable.

## Correctness metrics

| Date | Experiment | Observation | Evidence | State |
|---|---|---|---|---|
| 2026-07-19 | EXP-0.1 | RF SSB/MSB/LSB = 61/481/1921 timesteps; gradient support equals `1 + 2(K-1)Σd` | `artifacts/resubmission/gate0/gate0_results.json` | reproduced |
| 2026-07-19 | EXP-0.4 | future-perturbation difference 0; batch max difference 8.94e-8; deterministic and checkpoint differences 0 | same Gate 0 JSON | reproduced |
| 2026-07-19 | tests | 11 passed | `PYTHONPATH=src ... python -m pytest -q` | reproduced |
| 2026-07-19 | tests after split/mask/metric fixes | 14 passed | same test command | reproduced |
| 2026-07-19 | full suite at handoff revision `d8a7b93` | 27 passed on CPU (`CUDA_VISIBLE_DEVICES=''`) | `PYTHONPATH=src ... python -m pytest -q` | reproduced |
| 2026-07-19 | corrected StudentLife cache audit | N=2160, T=60, F=8; classes 578/973/609; 46 participants; all finite; five validation folds 436/422/432/436/434; zero subject overlap | `studentlife_causal_ffill_v2.npz`, data hash `a9cbaa3a22c2bf4e`, split hash `6208d08f0b8db52b` | reproduced |

## Headline quality metrics inherited from immutable result JSON

## Corrected StudentLife live progress (provisional; not headline evidence)

| Emitted | Model | Seed | Completed scope | Pooled macro-F1 | Evidence/status |
|---|---|---:|---|---:|---|
| 2026-07-19 23:30 IST | D-MSTCN | 0 | all 5 grouped folds | 0.3376 | live stdout preserved in `studentlife_headline_corrected_live.log`; provisional 1/3 seeds |
| 2026-07-20 01:15 IST | D-MSTCN | 1 | all 5 grouped folds | 0.3524 | same live log; provisional 2/3 seeds |
| 2026-07-20 02:49 IST | D-MSTCN | all (0–2) | 3 seeds × 5 folds complete | 0.3428 ± 0.0067 (pooled mean±std) | model partial JSON `studentlife_headline_corrected_partial.json`; seed-2 individual pooled value not durably captured (stdout→pts, not a file); fold-level macro-F1 mean 0.3347, 95% CI [0.3142, 0.3539]; per-fold avg-over-seeds [0.3668, 0.3360, 0.2988, 0.3389, 0.3332]; provisional 1/6 models |
| 2026-07-20 02:51 IST | LSTM | all (0–2) | 3 seeds × 5 folds complete | 0.2970 ± 0.0041 (pooled mean±std) | same partial JSON (`completed_models=[dmstcn, lstm]`); fold-level macro-F1 mean 0.2935, 95% CI [0.2664, 0.3188]; per-fold avg-over-seeds [0.3265, 0.3062, 0.2441, 0.3175, 0.2730]; provisional 2/6 models |
| 2026-07-20 03:51 IST | temporal-cnn | all (0–2) | 3 seeds × 5 folds complete | 0.3243 ± 0.0077 (pooled mean±std) | same partial JSON (`completed_models=[dmstcn, lstm, temporal-cnn]`); fold-level macro-F1 mean 0.3187, 95% CI [0.3040, 0.3342]; per-fold avg-over-seeds [0.3289, 0.3467, 0.3000, 0.3155, 0.3021]; provisional 3/6 models |
| 2026-07-20 03:53 IST | transformer | all (0–2) | 3 seeds × 5 folds complete | 0.3675 ± 0.0047 (pooled mean±std) | same partial JSON (`completed_models=[…, transformer]`); fold-level macro-F1 mean 0.3630, 95% CI [0.3528, 0.3733]; per-fold avg-over-seeds [0.3761, 0.3720, 0.3477, 0.3492, 0.3701]; provisional 4/6 models. **Exceeds D-MSTCN's provisional pooled 0.3428** — preserve as-is; paired participant-level comparison and audit pending |
| 2026-07-20 03:57 IST | timesnet | all (0–2) | 3 seeds × 5 folds complete | 0.3493 ± 0.0073 (pooled mean±std) | same partial JSON (`completed_models=[…, timesnet]`); fold-level macro-F1 mean 0.3446, 95% CI [0.3321, 0.3546]; per-fold avg-over-seeds [0.3362, 0.3541, 0.3542, 0.3231, 0.3551]; provisional 5/6 models. Also slightly above D-MSTCN's provisional pooled 0.3428 |
| 2026-07-20 03:58 IST | itransformer | all (0–2) | 3 seeds × 5 folds complete | 0.3612 ± 0.0098 (pooled mean±std) | final JSON `studentlife_headline_corrected.json`; fold-level macro-F1 mean 0.3573, 95% CI [0.3447, 0.3704]; per-fold avg-over-seeds [0.3816, 0.3629, 0.3361, 0.3522, 0.3536]; **6/6 models complete — run finished, PID 59422 exited normally**. Also above D-MSTCN's pooled 0.3428 |

These values are retained to observe every emitted metric, but they must not be ranked, averaged,
or interpreted until all six models complete and the final fail-closed audit passes.

**Quarantine notice:** inherited StudentLife metrics below used dataset hash
`62de62987570bc40`, whose leading-prefix imputation was later proven to backward-fill
from a future observation. It is retained for audit history only and must not be cited.

| Experiment | Dataset/features | D-MSTCN primary metric | Rank | Primary inference | Evidence | State |
|---|---|---:|---:|---|---|---|
| EXP-4.1 | StudentLife, 8 sensor features | fold macro-F1 0.3233 | 4/6 | No significant pairwise result; exact two-sided Wilcoxon minimum with five non-zero pairs is 0.0625 | `artifacts/resubmission/phase4/studentlife_headline.json` | imported |
| EXP-4.2 | E-DAIC, 23-dim LLD | test macro-F1 0.5529 ± 0.0918 | 2/6 | Every paired participant-bootstrap 95% CI spans zero | `artifacts/resubmission/phase4/daic_headline.json` | imported |
| EXP-4.2b | E-DAIC, 88-dim eGeMAPS | test macro-F1 0.5222 ± 0.0795 | 3/6 | Every paired participant-bootstrap 95% CI spans zero | `artifacts/resubmission/phase4/daic_headline_egemaps88.json` | imported |
| EXP-4.2c | DAIC-WOZ, 88-dim eGeMAPS | test macro-F1 0.4854 ± 0.0307 | 2/6 | CI clears zero only against the simplified, weakest TimesNet implementation; not headline evidence | `artifacts/resubmission/phase4/daicwoz_headline_egemaps88.json` | imported |

## Mask-aware confirmatory DAIC-WOZ rerun

EXP-4.2c was rerun after excluding right padding from normalization and temporal
pooling. Official split is 107 train / 34 dev / 47 test; one source-corrupt dev
session is prespecified and excluded. Five seeds, class-balanced cross-entropy using
train-only class counts, dev-only checkpoint selection, and one test evaluation per run.

| Model | Macro-F1 | Accuracy | Balanced acc. | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| D-MSTCN | 0.4818 | 0.5574 | 0.4916 | 0.5065 | 0.3517 | 0.5116 | 0.1577 |
| LSTM | 0.4841 | 0.5574 | 0.4833 | 0.4450 | 0.3311 | 0.4882 | 0.0715 |
| Temporal-CNN | 0.4511 | 0.5149 | 0.4530 | 0.4316 | 0.3230 | 0.5059 | 0.1177 |
| Transformer | 0.4303 | 0.5149 | 0.4284 | 0.3974 | 0.2663 | 0.5620 | 0.1595 |
| TimesNet (simplified) | 0.4427 | 0.5234 | 0.4426 | 0.4234 | 0.3140 | 0.4974 | 0.0730 |
| iTransformer | 0.5090 | 0.5787 | 0.5190 | 0.5134 | 0.3499 | 0.4984 | 0.1342 |

D-MSTCN is 3rd/6 by mean macro-F1 after masking. Its 95% paired participant-bootstrap
interval spans zero against every baseline: LSTM −0.0024 [−0.056,+0.051], Temporal-CNN
+0.0307 [−0.017,+0.078], Transformer +0.0514 [−0.032,+0.138], simplified TimesNet
+0.0390 [−0.013,+0.088], and iTransformer −0.0272 [−0.162,+0.107]. Evidence:
`artifacts/resubmission/phase4/daicwoz_headline_egemaps88_masked.json`.

D-MSTCN per-class seed-mean precision/recall/F1 is class 0:
0.6987/0.6545/0.6708 and class 1: 0.2793/0.3286/0.2927. No superiority claim is
supported. The run was repeated after commit `07a78a1` with identical results.
Confirmatory JSON SHA-256:
`b0f0427d4a319b36a433e3d1dd7791987dc760dca4030dd1b78c4d9a3fd9a74f`.
Reliability evidence is under `artifacts/resubmission/figures/`.

## Cancelled/excluded runs

| Date | Experiment | Outcome | Prespecified reason | Quality inspected? |
|---|---|---|---|---|
| 2026-07-19 | corrected StudentLife launch 1 | cancelled at ~16 min before first seed metric | Audit found leading-prefix backward-fill leakage in cached data hash `62de62987570bc40`; cache quarantined | no |

Replacement StudentLife cache: `studentlife_causal_ffill_v2.npz`, data hash
`a9cbaa3a22c2bf4e`; Gate-1 split hash remains `6208d08f0b8db52b`, 18/18 tests pass.
The first metadata-free v2 cache is recognized by its unambiguous prespecified filename as
semantic version `studentlife-v2-causal_ffill`; its numerical content and content hash are
unchanged. Future headline JSON records both semantic version and content hash.

## Hardware / systems metrics

| Date | Observation | Value | State |
|---|---|---:|---|
| 2026-07-19 | Physical GPU inventory | 1 × NVIDIA RTX 4060 Ti, 16380 MiB | reproduced |
| 2026-07-19 | Genuine 2–8 GPU scaling | unavailable on this host | blocked |
| 2026-07-19 | StudentLife inference, batch 32, FP32 | median 1.427 ms; p95 1.541; p99 1.624; 22,418 samples/s; peak 27.9 MiB | reproduced |
| 2026-07-19 | DAIC-WOZ inference, batch 8, FP32 | median 7.545 ms; p95 7.795; p99 7.800; 1,060 samples/s; peak 130.2 MiB | reproduced |

Single-device evidence: `artifacts/resubmission/systems/single_device_profile.json`,
SHA-256 `1264699fa44c3e48d1f09fe1b686594df27bed8a0e84f250a60525a6940e2b91`.

## Required fields for subsequent runs

Record run ID, experiment/condition, dataset and feature version, split hash, config
hash, fold/repeat/seed, start/end time, status/failure class, selection metric,
per-class and aggregate metrics, calibration metrics, timing where relevant, prediction
path, and confidence interval/effect-size method.

The corrected EXP-4.1 summary must additionally pass
`scripts/audit_exp41_corrected.py`; its audit receipt records the source SHA-256 and fails
closed on an incomplete model/comparison family, wrong seeds or hashes, non-finite/out-of-range
metrics, invalid confidence intervals, or a fold mean that does not recompute.
