# D-MSTCN — Resubmission Experimentation Harness

Reference implementation and experiment harness for the D-MSTCN IEEE Access
resubmission. Built to the manuscript equations (pp. 7–8) and to the rigor rules in
`D_MSTCN_ONE_FILE_MASTER_PROMPT.md` (two execution modes, immutable run registry,
leakage-safe splits, corrected statistics).

Active branch: **`param-main`**. The former Codex and Claude implementation trees were
consolidated into `code/` (canonical) and `cold/` (archived comparison) on 2026-07-26.

## Current status

The implementation and local CPU/gloo validation are complete: **316 tests pass**. No
fresh PARAM result is citable yet. The fail-closed evidence pipeline currently admits
zero result families because the authoritative PARAM campaign has not run.

| Gate | State |
|---|---|
| 0–1: repository, model, data and mathematical correctness | **PASS locally** |
| 2–3: DDP correctness | logic-verified on CPU/gloo; **NCCL/V100 run pending** |
| 4: PARAM/SLURM infrastructure | built; **cluster execution pending** |
| 5: scientific campaign | 294-task plan built; **not executed** |
| 6: ablations | 78-task PARAM plan built; **not executed** |
| 7–10: scaling, SAP and TCP systems evidence | implemented; **hardware runs pending** |
| 11: theorem/formal claims | Outcome B recorded; manuscript revision pending |
| 12: evidence generation | implemented; no result family admitted yet |

The authoritative operational instructions are [RUN_ORDER.md](../../RUN_ORDER.md).
[RUNBOOK.md](../../RUNBOOK.md) is retained for history but is partly stale.

## Pending work

### 1. Immediate prerequisites

- [ ] On PARAM, pull the merged `param-main` branch and confirm the canonical path is
  `~/dsctm/code/dsctm`.
- [ ] Ask CDAC for an HTTP/HTTPS proxy, local PyPI/conda mirror, or data-transfer node.
  PARAM was observed without general internet/DNS egress.
- [ ] If no package mirror is available, transfer and install the prepared offline Linux
  wheel bundle.
- [ ] Stage StudentLife and DAIC-WOZ (approximately 90 GB) under
  `$DSCTM_DATA_ROOT`; use `rsync --partial` if CDAC provides no transfer route.
- [ ] Confirm that the available E-DAIC archives and labels may be used under their EULA.
- [ ] Rotate the Kaggle API token after StudentLife staging.
- [ ] Obtain/confirm the approximately 91 GPU-hour allocation, peak four-GPU concurrency,
  and one two-node reservation.

### 2. Author decisions required before the campaign

- [ ] Select the paper's primary depression corpus: classic DAIC-WOZ (189 participants,
  cited by the manuscript) or E-DAIC (275 participants).
- [ ] Resolve StudentLife windowing: re-window beyond 60 steps, restrict the multi-scale
  claim to DAIC-WOZ, or report that the medium/long receptive fields exceed the window.
- [ ] Decide whether SEED remains in scope; neither a dataset nor a working experiment is
  currently present.
- [ ] Decide whether to retain any convergence theorem. The implemented HOLD protocol is
  not covered by the rejected manuscript's theorem; Outcome B currently withdraws that
  unsupported theorem claim.

### 3. PARAM validation and calibration

Run these strictly in the order specified by `RUN_ORDER.md`:

- [ ] Build or verify the Python/PyTorch environment through a batch job.
- [ ] Run `2gpu_ddp_smoke.sbatch` on the debug partition to validate V100, NCCL, fp16,
  and distributed correctness; this closes the hardware half of Gate 3.
- [ ] Run `memory_probe.sbatch`. PARAM has 16 GB V100s, so no configured batch size is
  authoritative until the measured ceiling is recorded.
- [ ] Verify dataset hashes and extract eGeMAPS features on a compute node.
- [ ] Run `1task_dryrun.sbatch` and recalibrate the GPU-hour estimate before launching
  arrays.

### 4. Scientific experiments

- [ ] Complete equal-budget development tuning: 96 tasks total (48 StudentLife and
  48 DAIC-WOZ).
- [ ] Freeze the selected configurations, then complete confirmation runs: 120 tasks
  total (60 StudentLife and 60 DAIC-WOZ).
- [ ] Complete the planned architecture, dilation, fusion, personalization and
  preprocessing ablations: 78 tasks.
- [ ] Preserve immutable configs, environment records, predictions and run receipts for
  every completed or failed task.
- [ ] Do not reuse the older single-GPU headline numbers as confirmatory evidence; they
  remain non-authoritative and mostly do not support the rejected accuracy claim.

### 5. Multi-GPU systems evidence

- [ ] Measure strong and weak scaling on 1, 2 and 4 GPUs; treat 8 GPUs as best effort.
- [ ] Restate or withdraw the manuscript's 16-GPU claim. Sixteen ranks would consume
  roughly 89% of PARAM's GPU partition and is not realistically schedulable.
- [ ] Validate SAP forward/backward equivalence with NCCL on real GPUs.
- [ ] Run the four-mode comparison (synchronous DDP, synchronous SAP, asynchronous SAP
  without TCP, asynchronous SAP with TCP) on at least two nodes.
- [ ] Run the approved TCP/SAP sweep. The full 288-cell grid requires separate compute
  approval.
- [ ] Ask CDAC for an admin-assisted `tc netem` experiment, or withdraw claims about
  controlled bandwidth, RTT, jitter and packet loss.
- [ ] Cache SAP process groups before making timing claims above world size four.
- [ ] Remove or clearly quarantine the TCP simulator after real Gate 10 evidence exists.

### 6. Audit and final evidence

- [ ] Run `audit_campaign.py --all --aggregate`; resolve every rejected or incomplete
  family without manually bypassing admission checks.
- [ ] Run `build_evidence.py` to generate final tables, figures, manifests and receipts.
- [ ] Populate the currently empty figures output only from admitted PARAM results.
- [ ] Update the experiment ledger, gate dashboard, blocker register and reviewer tracker
  with immutable artifact paths and hashes.

### 7. Manuscript and reviewer response

- [ ] Add the editable LaTeX or Word manuscript, bibliography, original decision letter,
  and seven verbatim reviewer reports. Only the compiled rejected PDF and derived tracker
  are currently available.
- [ ] Rewrite the results and systems sections using admitted evidence only.
- [ ] Correct receptive-field, FiLM parameter, data-split, hardware/rank and statistical
  claims throughout the manuscript.
- [ ] Report negative or inconclusive findings honestly; do not seed-pick, tune only
  D-MSTCN, merge DAIC development/test sets, or select metrics after seeing test results.
- [ ] Prepare the point-by-point reviewer response and final reproducibility package.

Full blocker detail is maintained in [BLOCKERS.md](../../BLOCKERS.md); campaign state is
maintained in [STATUS.md](../../STATUS.md), and planned/completed experiment identifiers
are tracked in [EXPERIMENT_LEDGER.md](../../EXPERIMENT_LEDGER.md).

## Quick start

```bash
# Local CPU/gloo verification
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/run_gate0.py             # Gate 0 evidence → artifacts/resubmission/gate0/
pytest -q                               # correctness + statistics + leakage tests
```

Do not use this local quick start to produce manuscript numbers. Authoritative runs must
write beneath `results/param_utkarsh_authoritative/` on PARAM and pass the result auditor.

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
