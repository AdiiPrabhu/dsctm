# Paper-Relevant Observations Log

Running notes captured DURING experimentation that should feed the revised manuscript,
response letter, or limitations. Each entry: what was observed, evidence, and where it
matters. Nothing here is invented — every claim is measured or read from data/code.
(See also HANDOFF.md for status and the always-on no-fabrication rule.)

## Architecture / correctness (Gate 0)
- **Receptive field is 61 / 481 / 1921** (SSB/MSB/LSB), measured by input-gradient support
  and equal to the two-conv formula `1+2(K-1)Σr`. The manuscript's printed **47 / 383 / 1535
  are wrong** (match neither one- nor two-conv). → Fix RF numbers + Fig 1/2 labels (T2-02).
- **FiLM per-subject cost = d_s = 8 stored params** (one embedding row); γ,β (2D=256) are
  *generated* by a shared MLP, i.e. activations, not per-subject storage. The "adds 2D
  parameters per subject" wording is wrong. → Correct parameter accounting (T2-03).
- Total D-MSTCN params ≈ **1.36 M** (StudentLife config, 48 subjects).
- Causality holds exactly (0 future leakage), model is deterministic and
  checkpoint-equivalent; batch-invariant to ~1e-7. → Supports reproducibility statement.
- TCP invariants (Δ increment/reset, HOLD, HOLD-over-periodic precedence, Δ≤δ_max) hold in
  simulation. The invariants are real; the *performance* claims still need the 8-GPU server.

## Datasets / provenance (Gate 1)
- **DAIC on disk is E-DAIC (AVEC-2019): 275 sessions, official 163/56/56.** Manuscript cites
  classic DAIC-WOZ (189, 107/82). → Report official splits; this *removes* the 107/82 problem.
  Needs author confirmation of which corpus the paper used.
- **eGeMAPS = openSMILE 2.3.0, 23 LLDs @ 100 Hz** on disk; manuscript says 88-dim @ 0.5 s
  (v3.0). Exact 88-dim reproduction not possible from this release. → Describe features as
  actually used; flag discrepancy.
- **DAIC is class-imbalanced: 209 negative / 66 positive (~24%).** → Headline metric must be
  macro-F1 / PR-AUC, not accuracy; report class balance.
- DAIC session lengths (0.5 s frames): median 1821, max capped at T=2000, min 830.
- **StudentLife = 46 subjects** with sufficient Stress-EMA (manuscript says 48).
- **StudentLife stress scale is non-monotonic**; mapped to 3 classes {4,5}=low, {1}=moderate,
  {2,3}=high → 578/973/609 (balanced). Mapping is a documented design choice.
- **Sensor missingness ≈ 0.61** at 1-min resolution (duty-cycled sensors), forward-filled. →
  State imputation + realized-context honestly. Realized context ≤ 60 min regardless of RF.

## Statistics
- With **5 seeds OR 5 folds**, a two-sided exact Wilcoxon signed-rank **cannot reach p<0.05**
  (min p = 0.0625). Every "† p<0.05 (5 seeds)" marker is unreachable as printed. → Replace
  significance stars with effect sizes (rank-biserial, Hodges-Lehmann) + bootstrap CIs (E4-13/15).
- Primary unit = participant/fold, not seed (seeds = stability repetitions).

## Results (filled in as runs complete)
- _(headline StudentLife grouped-CV + E-DAIC official-split results will be appended here,
  with per-fold uncertainty — no numbers until runs finish.)_
