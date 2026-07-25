# D-MSTCN Metrics Log

Last updated: 2026-07-19 21:15 IST

This file is the append-only human-readable index of observed metrics. Machine-readable
run artifacts remain authoritative. Values are recorded only with an evidence path and
verification state.

## Imported candidate results (awaiting independent audit)

Source checkout: `/mnt/adissd/phd/dsctm-resubmission/claude/dsctm` at local commit
`03cc9ec`. These are observations from existing JSON artifacts, not yet independently
reproduced in the Codex checkout.

| Experiment | Dataset/features | D-MSTCN result | Rank | Statistical conclusion | Evidence | State |
|---|---|---:|---:|---|---|---|
| EXP-4.1 | StudentLife, 8 sensor features | fold macro-F1 0.3233 | 4/6 | No significant comparison; two-sided exact Wilcoxon cannot reach 0.05 with five non-zero pairs | `phase4/studentlife_headline.json` in candidate checkout | imported_unverified |
| EXP-4.2 | E-DAIC, 23-dim LLD | test macro-F1 0.5529 ± 0.0918 | 2/6 | All paired participant-bootstrap 95% CIs span zero | `phase4/daic_headline.json` in candidate checkout | imported_unverified |
| EXP-4.2b | E-DAIC, 88-dim eGeMAPS | test macro-F1 0.5222 ± 0.0795 | 3/6 | All paired participant-bootstrap 95% CIs span zero | `phase4/daic_headline_egemaps88.json` in candidate checkout | imported_unverified |
| EXP-4.2c | DAIC-WOZ, 88-dim eGeMAPS | test macro-F1 0.4854 ± 0.0307 | 2/6 | Only comparison with simplified TimesNet has CI above zero; not a credible headline baseline | `phase4/daicwoz_headline_egemaps88.json` in candidate checkout | imported_unverified |

## Runtime and hardware observations

| Observation | Value | Evidence | State |
|---|---|---|---|
| Available physical GPUs | 1 | `nvidia-smi`, RTX 4060 Ti 16380 MiB | verified_2026-07-19 |
| Multi-GPU scaling capability | unavailable locally | only one physical GPU enumerated | verified_2026-07-19 |

## Logging rule

Every new run must record configuration hash, split hash, seed/fold, status, timing,
loss/selection metric, final metrics, confidence intervals where applicable, and the
immutable artifact path. Failed runs remain visible.
