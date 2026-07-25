# D-MSTCN — Resubmission Experimentation Harness

Reference implementation and experiment harness for the D-MSTCN IEEE Access
resubmission. Built to the manuscript equations (pp. 7–8) and to the rigor rules in
`D_MSTCN_ONE_FILE_MASTER_PROMPT.md` (two execution modes, immutable run registry,
leakage-safe splits, corrected statistics).

Branch: **`experimentation1`** (Claude's line of work; Codex uses `experimentation2`).

## Status

| Gate / phase | Runs without data? | State |
|---|---|---|
| **Gate 0 — implementation correctness** (RF, params/FLOPs, TCP invariants, causality) | ✅ yes | **runs now, passing** |
| Gate 1 — data integrity (provenance, leakage audit) | needs data | code ready; awaits StudentLife/DAIC |
| Phase 2 — baseline reproduction & fair baselines | needs data | models ready; awaits data |
| Phase 3 — synchronization ablation | needs data (+GAP-5 for systems realism) | scaffold |
| Phase 4 — locked headline eval + statistics | needs data | splits + stats ready |
| Phase 5 — architecture / personalization ablations | needs data | model supports ablation flags |
| Phase 6 — systems scaling | needs **8-GPU server** | not runnable on this host |

## Quick start

```bash
# shared venv (one level up, shared with the codex workspace)
source ../../venv/bin/activate          # /media/adii/adissd/phd/dsctm-resubmission/venv
pip install -e .

python scripts/run_gate0.py             # Gate 0 evidence → artifacts/resubmission/gate0/
pytest -q                               # correctness + statistics + leakage tests
```

## What Gate 0 establishes today (no data required)

- **EXP-0.1 receptive field** — measured RF = **61 / 481 / 1921** (two-conv formula
  `1+2(K-1)Σr`); the manuscript's printed **47 / 383 / 1535** match neither formula
  and are corrected here (reviewer T2-02).
- **EXP-0.2 parameters** — per-subject *stored* cost = **d_s = 8** (one embedding
  row); the γ,β vectors (2D=256) are *generated* by a shared FiLM MLP, not stored
  per subject (corrects the "2D parameters per subject" claim, T2-03).
- **EXP-0.3 TCP** — staleness increment/reset, HOLD activation, HOLD-over-periodic
  precedence, and Δ ≤ δ_max all verified in single-process simulation.
- **EXP-0.4 causality** — strict causal padding (0.0 future leakage), batch
  invariance (~1e-7), bit-exact determinism, variable-length support, checkpoint
  equivalence.

## Package layout

```
src/dsctm/
  repro.py        seeding, determinism modes, environment capture
  config.py       YAML resolve + hash
  registry.py     immutable run registry (run identity + per-run dirs)
  data/
    contract.py   WindowedDataset — the canonical shape every loader emits
    synthetic.py  multi-scale synthetic generator (pipeline runs before real data)
    splits.py     subject-grouped CV, holdout, leakage assertions
  models/
    blocks.py     causal conv, dilated residual block, CSAG, FiLM, head
    dmstcn.py     D-MSTCN (with ablation flags)
    baselines.py  LSTM, Temporal-CNN, Transformer, TimesNet, iTransformer, DP-LSTM, FedAvg-LSTM
    timesnet.py   faithful official TimesNet classification pathway (pinned upstream commit)
  train/tcp.py    TCP staleness-protocol simulation (Algorithm 1)
  eval/
    metrics.py    macro-F1, accuracy, AUC, PR-AUC, Brier, ECE
    statistics.py participant/fold bootstrap CIs, paired effect sizes, Wilcoxon guard, Holm/BH
  experiments/gate0.py   Phase-0 experiments + runner
```
TimesNet is adapted from the official THUML implementation pinned in
`models/timesnet.py`. Historical JSON produced before commit `cc723c8` used the retained
`TimesNetBaseline` placeholder and remains explicitly labeled non-confirmatory.

## Plugging in real data

A loader only needs to emit a `dsctm.data.contract.WindowedDataset`:
`X (N,T,F) float32`, `y (N,)`, `subject_id (N,)`, optional `timestamp (N,)`, plus
`n_classes` / `label_type` / `sampling_interval_s`. See `configs/data/*.yaml` for the
expected StudentLife (F=8, T=60, 3-class) and DAIC-WOZ (F=88, 2-class, official
107/35/47 split) contracts. **Never commit raw data or subject IDs** (see `.gitignore`).

## Corrections baked in (vs. the rejected manuscript)

Receptive field reported as measured; FiLM parameter accounting separated
(stored vs. generated); single-server branch-parallel framing (no multi-node claim);
subject-grouped CV + official DAIC splits (no 107/82 merge); statistics refuse
unreachable significance (n≤5 two-sided Wilcoxon cannot reach p<0.05).
