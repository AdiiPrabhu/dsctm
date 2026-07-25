# D-MSTCN Resubmission — Handoff / Running Log

Living document. Newest update on top. Covers what is set up, what is built, dataset
provenance findings, and what is next. Companion to `README.md` (usage) and the
master-prompt gate protocol.

Claude continuation instructions are consolidated in `instruction.md`. Treat the live
process and raw artifacts as authoritative if any timestamp or progress detail here ages.

---

## Codex continuation (2026-07-19 21:25 IST)

An isolated continuation exists in `/mnt/adissd/phd/dsctm-resubmission/codex/dsctm`
on branch `experimentation2`, based on this checkout at commit `03cc9ec`. Gate 0 was
rerun there (11/11 tests passing). The continuation keeps `STATUS.md`, `METRICS.md`,
this handoff, and `docs/DMSTCN_ALGORITHM.md` as separate live records.

Audit findings before any confirmatory rerun: DAIC sequences are zero-padded but temporal
pooling is currently unmasked; mask/AMP coverage claimed by Gate 0 is incomplete; TimesNet
is only a simplified placeholder; and true multi-GPU evidence is impossible on the one-GPU
host. These are implementation gaps, not hidden caveats.

Update 21:40 IST: masking, valid-timestep normalization, live dataset roots, balanced
grouped folds, dev-only checkpoint capture, one-time test evaluation, and PR-AUC logging
are implemented; 14/14 tests pass. Gate 1 now yields StudentLife split hash
`6208d08f0b8db52b` with validation sizes 422–436. The first mask-aware DAIC-WOZ rerun
completed: D-MSTCN macro-F1 0.4818 (3/6); every paired bootstrap CI spans zero. See
`METRICS.md`. Transformer emitted a nondeterministic attention-kernel warning.

The run was repeated from committed code `07a78a1` and reproduced identically. Its
confirmatory JSON SHA-256 is `b0f0427d4a319b36a433e3d1dd7791987dc760dca4030dd1b78c4d9a3fd9a74f`;
reliability tables/figure are under `artifacts/resubmission/figures/`.

EXP-6.1 is complete: StudentLife batch-32 median/p95/p99 latency
1.427/1.541/1.624 ms (22,418 samples/s, 27.9 MiB peak); DAIC-WOZ batch-8
7.545/7.795/7.800 ms (1,060 samples/s, 130.2 MiB peak), FP32, 30 synchronized
samples after 10 warmups. Gate 0 now also passes per-branch perturbation RF and AMP
finite loss/gradient checks. The next jobs exceed 30 minutes and require the approval
table recorded in `STATUS.md` / the active assistant handoff.

Update 22:05 IST: the user explicitly approved continuation. The approximately
17 GPU-hour campaign is authorized; corrected StudentLife grouped-CV headline evaluation
starts first. No multi-GPU or external submission action is authorized.

Baseline-fidelity update: `models/timesnet.py` now adapts the official THUML TimesNet
classification path pinned at upstream commit `4e938a1` (FFT period discovery, 2D
inception blocks, spectral weighting, residual, masked classification head). The targeted
CPU shape test passes. Do not reinterpret historical simplified-TimesNet metrics; rerun
fairness comparisons with the faithful model.

Phase-5 runner hardening: `scripts/run_phase5_ablation.py` is ready; the experiment
module checkpoints after each variant and produces paired effect sizes plus Holm/BH
multiplicity corrections. Targeted statistics tests pass. Launch it after EXP-4.1 frees
the GPU.

Critical preprocessing finding: `_ffill` used the first later observation for leading
NaNs, which is backward-fill leakage. The active EXP-4.1 job was cancelled before its
first metric at ~16 minutes. The fix leaves leading missing values at zero and only
propagates observations forward; regression tests cover leading and fully missing cases.
Use only `studentlife_causal_ffill_v2.npz` and never reuse `studentlife.npz` for claims.

EXP-1.3 is ready in `experiments/preprocessing.py` / `run_exp13_preprocessing.py`.
Conditions are causal forward fill, observed-only train-fold mean, zero, and zero plus
observed-mask channels. Each condition records its data version/hash and uses the same
subject folds. Launch after EXP-4.1/Phase-5 according to the approved campaign queue.

Run-registry update: `registry.write_completed_fit` now creates the required immutable
directory for every completed fold/seed without participant IDs. EXP-1.3 and Phase-5
invoke it through `headline_cv` callbacks. A test asserts the required artifact set.

EXP-3.3 is ready in `experiments/delay_task.py`. Delays 4/64/192, exact XOR construction,
full/SSB/MSB/LSB controls, and success/falsification thresholds were written before any
training result. The mathematical ledger §9.2 records the implemented task exactly.

EXP-2.2/2.3 equal-budget tuning is ready in `experiments/fair_tuning.py`: 8 model-specific
dev trials for each of 6 models, then five-seed confirmation of each frozen winner. Search
uses `train_model` (no test index/loader); confirmation uses the test-once path. Use
`scripts/run_exp22_fair_tuning.py` with `DSCTM_DAICWOZ_CACHE` set to the complete cache.

Participant-only audio refinement is ready in
`scripts/build_daicwoz_participant_egemaps88.py`. It extracts only transcript-labelled
Participant intervals, concatenates them without invented silence, and builds the same
88-dimensional/0.5-second masked contract. Its public aggregate manifest contains counts
and length summaries but no coded participant IDs.

Registry recovery/failure behavior is now explicit: an identical completed run is reused
without overwrite or duplicate registry row, while a failed tuning trial gets an immutable
`model_failed` directory and is not silently replaced or granted extra search budget.

---

## ⏸ PAUSED HERE (2026-07-19 ~20:45 IST) — awaiting user decision on next step

State is fully committed & saved. **Nothing running.** Resume by picking one of the options below.

- **Repo:** `claude/dsctm` @ `experimentation1`. Two new commits, **local, NOT pushed**:
  - `b74d7a8` EXP-4.2c DAIC-WOZ 88-dim results + imbalance-fix/loader + pipeline + summarizer.
  - `ef658aa` docs/DMSTCN_ALGORITHM.md (model math, expected-vs-observed, improvement levers).
- **Headline (final, robust):** D-MSTCN 1st in **none** — StudentLife 4/6, E-DAIC-23d 2/6,
  E-DAIC-88d 3/6, DAIC-WOZ-88d 2/6; no credible significant win anywhere. See `SUMMARY.md`
  (cross-corpus table) and `docs/DMSTCN_ALGORITHM.md` §7.
- **Open options offered (user paused to decide):**
  1. `git push` the two commits.
  2. **Draft the equal-budget HP-sweep runner for all 6 models** (dev-select / test-once, same
     protocol; the single most legitimate lever to move D-MSTCN's rank). Show GPU-hour estimate +
     approval table first (master-prompt §5). This is the concrete next experiment.
  3. Draft the reviewer-response framing around the **reframed contribution** (efficiency d_s=8 vs
     2D / strict causality / cheap FiLM personalization / imbalance+RF+param corrections) rather
     than a headline accuracy win.
- **Legit improvement levers** (apply to ALL 6 models, pre-specify): see `docs/DMSTCN_ALGORITHM.md`
  §8.1. **Fabrication (do NOT do):** §8.2 (seed-picking, D-MSTCN-only tuning, dev-as-test, 107/82
  merge, metric shopping).

---

## ✅ RESUME COMPLETE — DAIC-WOZ downloaded + EXP-4.2c pipeline run (2026-07-19 ~20:00 IST)

**DONE — nothing pending here.** DAIC-WOZ (classic AVEC2017) download finished at **188/189**
sessions (session **440**'s AVEC2017 *source* zip is truncated to ~12.5% — full byte count matched
the server's `Content-Length` exactly, so it's corrupt at source, not a network cut; unrecoverable.
440 is a **dev** session → excluded, dev 35→34, **test 47 unaffected**. Quarantined +
documented: `dataset/DAIC-WOZ/{PROVENANCE_440.txt, 440_P.CORRUPT_TRUNCATED, 440_P.zip.CORRUPT}`).
The full pipeline ran green via `scripts/run_daicwoz_pipeline.sh` (extract → EXP-4.2c → summarize →
**11/11 pytest**). **Result: D-MSTCN 2nd/6, no credible win** — see the EXP-4.2c update-log entry
and `SUMMARY.md`. ⚠ **Bug fixed:** the documented `PYTHONPATH=src` invocation below is WRONG for
the cross-script `from scripts...` import (only resolves from stdin); use **`PYTHONPATH=src:.`**
(the pipeline script exports `"$PWD/src:$PWD"`). Historical in-flight state kept below for context.

---

**In-flight when power dropped:** DAIC-WOZ (classic AVEC2017) download+extract — **~72 of 189
sessions done** (~72 GB). The job is resumable & integrity-checked; a shutdown loses nothing on
disk. Nothing was committed to git, but all files are on disk (survive a clean shutdown; a git
commit is NOT required to resume). Data lives under `dataset/DAIC-WOZ/` and caches under
`claude/dsctm/artifacts/cache/` (both outside git).

**To resume, run these in order from a normal shell:**
```bash
# 1) FINISH THE DOWNLOAD (resumable: wget -c, re-checks all, skips complete zips)
cd /mnt/adissd/phd/dsctm-resubmission/dataset/DAIC-WOZ
nohup bash fetch_extract_daicwoz.sh >> fetch.log 2>&1 &
#    wait for "DAICWOZ_FETCH_DONE"; expect 189 *_P/ folders each with *_AUDIO.wav:
#    ls -d *_P/ | wc -l   (== 189)

# 2) THE PIPELINE (venv import needs PYTHONPATH=src; roots auto-resolve /media->/mnt)
cd /mnt/adissd/phd/dsctm-resubmission/claude/dsctm
export PYTHONPATH=src
VP=../../venv/bin/python
$VP -u scripts/build_daicwoz_egemaps88.py    # ~10 min, resumable (skips cached npz)
$VP -u scripts/run_exp42_daicwoz.py          # ~8 min -> artifacts/resubmission/phase4/daicwoz_headline_egemaps88.json
$VP    scripts/summarize_phase4.py           # regenerate SUMMARY.md (auto-adds DAIC-WOZ + comparison if the summarizer is extended)
$VP -m pytest -q                             # sanity: 11 tests
```
**Env reminders:** activate/venv at `../../venv`; `import dsctm` FAILS without `PYTHONPATH=src`
(editable .pth points at the dead `/media` mount). Dataset roots default to `/media` too but every
runner resolves the live `/mnt` path itself. GPU: 1× RTX 4060 Ti 16 GB.

**All DAIC-WOZ code is already built, validated on partial data, tests green** — nothing to
re-write, just run the 3 commands above once the download finishes. Details in the 2026-07-19
(DAIC-WOZ) update-log entry below.

---

## Status dashboard

| Area | State |
|---|---|
| Repo / branch | `AdiiPrabhu/dsctm` @ **`experimentation1`** (Claude); Codex on `experimentation2` |
| Shared venv | `/media/adii/adissd/phd/dsctm-resubmission/venv` (torch 2.6.0+cu124) |
| GPU | 1× RTX 4060 Ti 16 GB (Phases 0–5 here; Phase 6 needs 8-GPU server) |
| **Gate 0** correctness | ✅ passing (RF, params, TCP, causality) — no data needed |
| **Gate 1** StudentLife | ✅ built + leakage-free (46 subj, 2160 windows) |
| **Gate 1** E-DAIC | ✅ built + leakage-free (275 sessions, official 163/56/56) |
| **Phase 4** EXP-4.1 StudentLife | ✅ ran — D-MSTCN mid-pack (4th/6), nothing significant (see 07-19 entry) |
| **Phase 4** EXP-4.2 E-DAIC (23-dim LLD) | ✅ imbalance fix + 5 seeds + participant bootstrap — collapse gone; D-MSTCN 2nd/6, **no significant advantage** (all paired CIs span 0) |
| **Phase 4** EXP-4.2b E-DAIC (**88-dim, paper's features**) | ✅ rebuilt eGeMAPSv02 88-dim functionals from raw audio; **D-MSTCN 3rd/6, still no significant win** — feature fidelity does not rescue the headline (see 07-19 88-dim entry) |
| **DAIC-WOZ** (classic AVEC2017, corpus the paper cites) | ✅ **downloaded 188/189** (440 source-corrupt, excluded); 88-dim extracted; EXP-4.2c ran (5 seeds + participant bootstrap) → `daicwoz_headline_egemaps88.json`. **D-MSTCN 2nd/6 (test macro-F1 0.4854), no credible win** (only paired CI above 0 is vs the weakest/simplified TimesNet) |
| Phase 2 baselines (matched budget) | ✅ ran as the 6-model comparison inside Phase 4 |
| Multi-GPU (Phase 6) | ⬜ deferred to rented 8-GPU server |

---

## Update log

### 2026-07-19 (DAIC-WOZ RESULT) — EXP-4.2c ran on the paper's actual corpus+features — D-MSTCN 2nd/6, still no credible win
Download resumed after the power loss and finished at **188/189** sessions; ran the full pipeline
(`scripts/run_daicwoz_pipeline.sh`: extract 88-dim → EXP-4.2c 5-seed + participant bootstrap →
summarize → pytest 11/11). Raw: `daicwoz_headline_egemaps88.json`; tables now in `SUMMARY.md`
(EXP-4.2c block + cross-corpus placement table).
- **Ranking (5-seed mean test macro-F1, DAIC-WOZ 88-dim):** temporal-cnn 0.5055 > **dmstcn
  0.4854** > transformer 0.4819 > lstm 0.4692 > itransformer 0.4630 > timesnet 0.4066. D-MSTCN is
  **2nd/6** — behind temporal-cnn (the same model that led E-DAIC-23d).
- **Paired participant bootstrap (primary):** 4 of 5 baseline CIs span 0. The **only** exception in
  the entire study is **vs TimesNet** (Δ +0.079, 95% CI **[+0.005, +0.153]**, P=0.98) — but that is
  the **weakest** model and the **simplified TimesNet baseline already flagged for replacement**, so
  it supports **no** headline claim. Nothing significant vs temporal-cnn/transformer/lstm/itransformer.
- **Net:** the negative headline is now robust across **both corpora (StudentLife, E-DAIC, DAIC-WOZ)
  AND both feature sets (23-dim, 88-dim)**. D-MSTCN is 1st in none, mid-pack everywhere. Same fair
  matched-budget protocol for all 6 models; no seed-picking / D-MSTCN-only tuning.
- **Data provenance:** session **440** excluded — its AVEC2017 source zip is truncated to ~12.5%
  (byte count matched server `Content-Length` exactly → corrupt at source, not network). It's a dev
  session, so test 47 is unaffected; dev 35→34. Quarantined + logged in `PROVENANCE_440.txt`.
- **Tooling:** fixed the broken `PYTHONPATH=src` invocation (needs `src:.` for `from scripts...`);
  extended `summarize_phase4.py` to render the DAIC-WOZ block + a cross-corpus placement table, and
  made the "N official test sessions" note corpus-accurate (E-DAIC 56 / DAIC-WOZ 47).

### 2026-07-19 (DAIC-WOZ) — pulling in the corpus the paper actually cites; full pipeline BUILT, download in progress
Goal: re-run on **classic DAIC-WOZ (AVEC2017)** — the corpus the manuscript cites (189 sessions),
vs the E-DAIC we had on disk. Downloading all 189 `*_P.zip` (85.6 GB) from
`dcapswoz.ict.usc.edu/wwwdaicwoz/` into `dataset/DAIC-WOZ/` + AVEC2017 split CSVs.
- **Status at power loss: ~72/189 downloaded+extracted.** Resume: see the ⚡ block at top.
- **Splits (verified):** official **train 107 / dev 35 / test 47** (test labels in
  `full_test_split.csv`; blind `test_split` has none). Using proper train/dev-select/test with
  **NO dev+test merge** — this is what answers the reviewer's "107/82" objection. ~30% positive
  (imbalanced → class-balanced CE, same as E-DAIC).
- **Code built + validated on partial data (tests green, 11/11):**
  - `data/daic.py`: `_read_daicwoz_splits`, `build_daicwoz88` (shares new `_assemble88` helper
    with `build_daic88`; E-DAIC path unchanged, re-verified N=275/F=88).
  - `scripts/build_daic_egemaps88.py`: refactored to a corpus-agnostic `extract_corpus(...)`.
  - `scripts/build_daicwoz_egemaps88.py`: extracts 88-dim eGeMAPS from DAIC-WOZ audio →
    `artifacts/cache/daicwoz_egemaps88/`.
  - `scripts/run_exp42_daicwoz.py`: EXP-4.2 on DAIC-WOZ, 5 seeds + participant bootstrap →
    `daicwoz_headline_egemaps88.json`.
- **Provenance caveat to record:** DAIC-WOZ `*_AUDIO.wav` is the full WoZ interview (includes the
  Ellie interviewer); E-DAIC reused the SAME recordings (session 300 identical length 1297), so
  whole-recording eGeMAPS is applied identically to both — fair, but a participant-only variant
  (via transcript) is a legit refinement.
- **Pending after download:** run the 3 pipeline commands (⚡ block) → get the DAIC-WOZ headline,
  then compare across StudentLife / E-DAIC-23d / E-DAIC-88d / DAIC-WOZ-88d and update docs.
- **Context:** this answers "how to get a D-MSTCN win as stated in the paper" — testing the
  paper's actual corpus after its 88-dim features (E-DAIC) already showed no win. Same fair
  protocol for all 6 models; no seed-picking / D-MSTCN-only tuning.

### 2026-07-19 (88-dim) — EXP-4.2b on the paper's stated feature set — D-MSTCN still does NOT win
Rebuilt the manuscript's **88-dim eGeMAPS functionals** (eGeMAPSv02, openSMILE 3.0 via
`opensmile` 2.6.0) from the on-disk raw audio over 0.5 s windows — the biggest fidelity gap to
the paper (disk shipped only 23-dim LLDs). New: `scripts/build_daic_egemaps88.py`, cache
`artifacts/cache/daic_egemaps88/`, loader `daic.build_daic88`, runner `scripts/run_exp42_88.py`;
raw `daic_headline_egemaps88.json`. All 275 sessions, 0 missing audio, Gate-1 leakage-free.
Same split/protocol/loss/5-seeds/bootstrap as the 23-dim run — only features changed.
- **Result:** with the paper's own features, **D-MSTCN is 3rd/6** (test macro-F1 0.5222) —
  it *dropped* from 2nd (23-dim, 0.5529); LSTM (0.5403) and TimesNet (0.5343) are ahead. Every
  paired participant-bootstrap CI still spans 0; point estimate is negative vs LSTM and TimesNet.
- **Interpretation:** the richer functionals helped some baselines more than D-MSTCN. Feature
  fidelity — the single most legitimate lever toward the paper's headline — **does not produce a
  win**. The negative headline is now robust across **both corpora AND both feature sets**.
- Context: this was in response to "how to get a D-MSTCN win as stated in the paper." Pursued
  the honest path (reproduce the paper's inputs, evaluate all models identically); no
  seed-picking / D-MSTCN-only tuning / metric or split shopping. Remaining legitimate levers all
  need author input (exact eval protocol, exact 88-dim config, equal-budget HP search for all 6).

### 2026-07-19 (5-seed) — EXP-4.2 with proper statistics — imbalance FIXED, but D-MSTCN advantage does NOT hold
Re-ran EXP-4.2 at **5 seeds** with the primary inference on the **fixed official test set**
(participant bootstrap, n_boot=10000; seeds secondary) — matching the §8 plan and EXP-4.1's
rigor. Log `artifacts/exp42_5seed.log`; detail `phase4/OBSERVATIONS.md` (5-seed section);
tables `SUMMARY.md`; raw `daic_headline.json` (now 5-seed schema).
- **This corrects the 2-seed reading below.** The earlier "D-MSTCN 1st/6, +0.195" was a
  2-seed artifact: it rode a lucky seed0 (test 0.6889). D-MSTCN's 5-seed test macro-F1 is
  **0.5529 ± 0.0918** (seeds 0.41–0.69 — the *widest* spread of any model).
- **Ranking (5-seed mean test macro-F1):** temporal-cnn 0.5631 > **dmstcn 0.5529** > lstm
  0.5258 > timesnet 0.5242 > transformer 0.5151 > itransformer 0.5101. D-MSTCN is **2nd, not
  1st**.
- **Primary paired test (participant bootstrap of the test-F1 difference):** vs every baseline
  the **95% CI spans 0** (Δ from −0.010 vs temporal-cnn to +0.043 vs itransformer;
  P(D-MSTCN better) 0.39–0.78). D-MSTCN is **statistically indistinguishable from all five
  baselines** on E-DAIC test — the 56-session set is too small to resolve these gaps.
- **What the fix DID achieve (report this):** class-balanced CE removes the majority collapse
  for **all 6 models** (Δtest +0.06…+0.15; both classes predicted). A real methodological
  correction — but it does **not** rescue a D-MSTCN headline claim. Net: **D-MSTCN shows no
  headline advantage on either corpus** (mid-pack + n.s. on both). Negative headline stands.

### 2026-07-19 (re-run) — EXP-4.2 imbalance fix: class-balanced CE — collapse resolved
Re-ran **only EXP-4.2** (E-DAIC) with the imbalance fix; EXP-4.1 untouched and NOT re-run.
Script `scripts/run_exp42.py`; log `artifacts/exp42_balanced.log`; ≈3 min, 12 model-fits.
Full detail + before/after table: `artifacts/resubmission/phase4/OBSERVATIONS.md` (07-19 re-run
section) and `SUMMARY.md`; raw: `daic_headline.json` (new) vs `daic_headline_plainCE.json` (old).
- **The fix:** training loss `CrossEntropyLoss()` → **class-balanced** CE, weights
  `n/(C·count_c)` from **TRAIN labels only** (leakage-safe; `trainer._build_loss`, opt-in via
  `cfg["class_weight"]="balanced"` — now set in `DAIC_CFG`; StudentLife cfg unchanged). Single
  principled change: same split, bs, seeds, dev-select/test protocol, argmax decoding.
- **Result:** the 5/6 **majority collapse is eliminated** — every model now predicts both
  classes (dev macro-F1 0.44 → ~0.59–0.635). On test (mean of 2 seeds), **D-MSTCN is now 1st/6**
  (macro-F1 **0.6055**, bal-acc 0.6192), the biggest gainer from the fix (Δtest **+0.195**).
- **Honest caveat (do not skip in the letter):** only **2 seeds**, wide per-seed spread
  (D-MSTCN test 0.522–0.689), dev scores tightly bunched. Indicative, **not** yet statistically
  robust. Next: ≥5 seeds + CIs / paired test before any "D-MSTCN wins on E-DAIC" claim.
- Also fixed: loader `DAIC_ROOT_DEFAULT` pointed at the dead `/media/...` mount → first launch
  built an empty dataset; re-run script now resolves the live `/mnt/...` root. No metric affected.

### 2026-07-19 (later) — Phase 4 headline eval RAN — ⚠ negative result (reported honestly)
Ran `scripts/run_phase4.py` to completion on this GPU (EXP-4.1 StudentLife 5-fold×3 seeds,
EXP-4.2 E-DAIC official split×2 seeds; 6 models each). Wall-clock ≈5 h. Detail +
per-model-seed log: `artifacts/resubmission/phase4/OBSERVATIONS.md`; tables: `SUMMARY.md`;
raw: `{studentlife,daic}_headline.json`.
- **First run died at 16.5 min** — orphaned/killed by its launching session + progress stuck
  in an unflushed file buffer (plain `print`, block-buffered redirect). NOT OOM. Relaunched
  with `python -u` + `nohup` and added **per-model partial-JSON checkpoints** to `headline.py`
  (purely additive; no metric changed).
- **EXP-4.1 StudentLife:** D-MSTCN ranks **4th/6** (fold macro-F1 0.3233). transformer (0.3617)
  and itransformer (0.3484) beat it; paired rank-biserial vs those two is negative (−0.87,
  −0.73). Nothing significant — 5 folds can't reach p<0.05 (min exact p 0.0625; guard held).
  All scores ≈ 3-class chance (~0.33).
- **EXP-4.2 E-DAIC:** **5/6 models collapse to the majority baseline** (dev 0.4400/test 0.4105,
  byte-identical across models & seeds). Only itransformer learns on dev (0.6323) but is weak
  on test (0.4476). Cause: 24%-positive imbalance + plain CE, no class weighting. D-MSTCN
  collapses too.
- **Bottom line:** under a fair matched-budget protocol, **D-MSTCN shows no headline advantage**
  on either corpus. Reported as-is (no-fabrication rule). Legit follow-ups: imbalance-aware
  E-DAIC training; confirm corpus/feature identity with author; reconsider the manuscript's
  headline claim. NO number was adjusted to look better.

### 2026-07-19 — Datasets ingested; Gate 1 started
- Found both datasets on disk:
  - `dataset/StudentLife/` (3.3 GB) — sensing + Stress EMA + surveys.
  - `dataset/daicwoz/` (342 GB) — **E-DAIC (AVEC-2019)**, 274 sessions, official splits.
- Built **StudentLife loader** (`data/studentlife.py`) → `WindowedDataset` (T=60, F=8).
  - Gate 1: **46 subjects, 2160 windows, 3-class {578/973/609}, leakage-free** grouped 5-fold
    (split_hash `d7bfd972ce2833bc`). Sensor missingness ≈ 0.61 (forward-filled).
- Built **E-DAIC loader** (`data/daic.py`) + cached all eGeMAPS (0.5s-aggregated).
  - Gate 1: **275 sessions, official 163/56/56, T=2000, F=23, leakage-free** (no
    cross-split participant overlap). 2-class **imbalanced {0:209, 1:66}** (~24% positive
    → report PR-AUC / macro-F1, not accuracy). True lengths: median 1821, max capped 2000.
- Added **Gate 1 module** (`experiments/gate1.py`): provenance + leakage for both.

### 2026-07-18 — Repo + Gate 0 harness
- Cloned repo (SSH), created `experimentation1`, shared venv, installed torch cu124.
- Built D-MSTCN reference implementation + 7 baselines + harness (see README).
- **Gate 0 passing (11 tests)**: RF 61/481/1921 (manuscript 47/383/1535 wrong);
  per-subject params = d_s=8 not 2D; TCP invariants hold; causal/deterministic. Pushed.

---

## ⚠ Dataset provenance findings (important for the response letter)

These are honest manuscript-vs-actual discrepancies. They are recorded, not worked around.

1. **DAIC identity.** On disk is **E-DAIC (AVEC-2019): 274 sessions, official
   163/56/55 train/dev/test.** The manuscript cites classic **DAIC-WOZ: 189 sessions,
   107/82.** → Use the official E-DAIC splits and report them. This *resolves* the
   reviewer's 107/82 objection (no dev+test merge). **Confirm with the author which
   corpus the paper actually used.**
2. **eGeMAPS version/dim.** Disk provides eGeMAPS **LLD 23-dim @ 100 Hz, openSMILE
   2.3.0**. Manuscript states **88-dim @ 0.5 s, openSMILE 3.0**. We aggregate LLDs to
   0.5 s frames (F=23). Exact reproduction of the 88-dim setup is not possible from this
   release — a finding in itself.
3. **StudentLife subjects.** 46 subjects have ≥ min Stress-EMA responses (manuscript
   says 48). Two/three users dropped for insufficient labels. Documented.
4. **StudentLife stress label mapping.** 5-point non-monotonic scale → 3 classes:
   `{4,5}=low, {1}=moderate, {2,3}=high`. Documented + configurable (`DEFAULT_STRESS_MAP`).
5. **Sensor missingness ≈ 0.61** at 1-min resolution (StudentLife sensors are
   duty-cycled). Forward-filled; per-window missingness recorded. Realized context is
   ≤ 60 min regardless of theoretical RF (master-prompt §7.1).
6. **DAIC test access (GAP-6).** E-DAIC ships test labels here, so test evaluation is
   possible — **confirm this is authorized** before reporting test numbers; otherwise
   dev-only.

---

## StudentLife feature contract (F=8, 1-min bins over 60 min before each EMA)
`activity_mean, audio_mean, conversation_frac, dark_frac, phonelock_frac,
phonecharge_frac, gps_speed_mean, gps_moving_frac`. Windows carry RAW features;
normalization/imputation are applied AFTER the subject split (leakage-safe).

## How to run
```bash
source ../../venv/bin/activate && cd claude/dsctm   # or use venv/bin/python directly
python scripts/run_gate0.py                          # Gate 0 correctness (no data)
python -c "from dsctm.data.studentlife import build_studentlife as b; \
           from dsctm.experiments.gate1 import run_gate1_studentlife as g; g(b(cache='artifacts/cache/studentlife.npz'))"
python scripts/build_daic.py                         # E-DAIC cache + Gate 1 (heavy, ~minutes)
pytest -q                                            # 11 correctness/stats/leakage tests
```
Caches live under `artifacts/cache/` (gitignored). Raw data and subject IDs are never committed.

## Next / TODO
- [x] Complete a code-to-equation audit of `docs/DMSTCN_ALGORITHM.md`. It now includes
      model, data, loss, training, baseline, ablation, metric, uncertainty, multiplicity,
      reproducibility, registry, and admission-audit formulations, with evidence boundaries.
      Important interpretation: held-out participants all map to trained unknown FiLM row 0;
      grouped evaluation has no individualized test-time subject adaptation.
- [x] Live EXP-4.1 (launch commit `10b6c48`) **completed 03:58 IST** (PID 59422 exited
      normally) and **passed `audit_exp41_corrected.py` 03:59 IST** (`checks_passed: true`,
      no errors). Final JSON `studentlife_headline_corrected.json` SHA-256
      `abf7079fe189cd7b53239aebbbd3bcd4a7608a8412010ecefd5589c3734f8a3a`; audit receipt
      `studentlife_headline_corrected_audit.json`. Pooled macro-F1 (fold-level 95% CI):
      transformer 0.3675 [0.3528,0.3733] · itransformer 0.3612 [0.3447,0.3704] · timesnet
      0.3493 [0.3321,0.3546] · **D-MSTCN 0.3428 [0.3142,0.3539] (4/6)** · temporal-cnn
      0.3243 [0.3040,0.3342] · lstm 0.2970 [0.2664,0.3188]. Paired D-MSTCN-vs-baseline
      family: no comparison statistically resolvable (5 folds → two-sided exact Wilcoxon
      min p 0.0625; `significance_reachable: false` for all). Conclusion: **no reproducible
      headline advantage for D-MSTCN**; transformer-family baselines numerically beat it.
      Preserved as-is. Limitation: launch revision predates in-file data-hash embedding
      (`embedded_data_hash` null; tied to cache by independently audited hash
      `a9cbaa3a22c2bf4e`). Per-seed D-MSTCN values not durably captured (stdout→pts); only
      runner-emitted model-level aggregates recorded, none inferred.
      This monitoring/logging pass made **documentation-only** edits (this file, `STATUS.md`,
      `METRICS.md`, ignored live log); no source/experiment code was written or modified,
      and no git commit/push has been run yet (awaiting user decision).
- [x] Add `scripts/audit_exp41_corrected.py`, a fail-closed final-result validator and
      SHA-256 receipt generator. Run it immediately after the corrected JSON appears and
      before copying metrics into `METRICS.md`; focused accept/reject tests pass.
- [x] Audit the corrected cache directly and repair its metadata contract. Early v2 NPZs
      omitted the semantic version, so the known corrected filename now maps explicitly to
      `studentlife-v2-causal_ffill`; numerical content remains hash `a9cbaa3a22c2bf4e`.
      Future headline JSON embeds both semantic version and content hash. The live launch
      predates that output-field patch, so associate it using the independently audited hash.
- [x] Run the complete post-launch regression suite: 27/27 tests pass on CPU at
      `d8a7b93`. This is code-handoff verification, not provenance for the already-live
      EXP-4.1 process at `10b6c48`.
- [x] Remove overclaims from the mathematical record: historical results are not described
      as equal-budget tuned, the old StudentLife metric is visibly quarantined, and tested
      TCP utilities are not presented as a completed distributed implementation.
- [x] Add immutable per-fold preservation to future EXP-4.1 headline executions. The
      live corrected run started at 22:19:20 IST from `10b6c48` and predates this patch;
      do not attribute its eventual artifact to a later commit.
- [x] Implement the complete Phase-5 control family: seven branch combinations,
      dynamic/mean/static CSAG, half/double temperature, and no/global/subject/
      parameter-matched-global FiLM. The resulting 14 variants require 210 fits rather
      than the previously approved 105, so obtain revised approval before launching it.
- [x] Finish E-DAIC caching + Gate 1.
- [x] Training loop (scientific mode) + Phase-4 locked eval on StudentLife (grouped CV).
- [x] Phase-2 baselines ran as the 6-model matched-budget comparison inside Phase 4.
- [ ] **Decide how to handle the (robustly negative) headline** — D-MSTCN is mid-pack +
      non-significant on **both** corpora AND **both** feature sets (StudentLife 4th/6; E-DAIC
      23-dim 2nd/6; E-DAIC 88-dim/paper's-features 3rd/6; all paired CIs span 0). The most
      legitimate lever toward the paper's claim (feature fidelity) has been tried and does not
      win. Central resubmission question: reframe the contribution away from a headline accuracy
      win (toward TCP/causality/efficiency properties, or the honest methodology/provenance
      findings), or challenge the manuscript's original claim. Do NOT bury it.
- [x] **E-DAIC imbalance handling** — class-balanced CE (`trainer._build_loss`); re-ran EXP-4.2.
      Collapse resolved for all 6 models.
- [x] **E-DAIC: ≥5 seeds + CIs + paired test** — done (5 seeds, participant bootstrap). Result:
      D-MSTCN advantage is NOT statistically resolved (see 5-seed entry). Optional further step:
      more seeds still won't shrink the CI much — it's the 56-session test set that limits power.
- [x] **E-DAIC 88-dim feature-fidelity reproduction** — extracted eGeMAPSv02 88-dim functionals
      from raw audio; re-ran EXP-4.2b. D-MSTCN 3rd/6, still no significant win (see 88-dim entry).
- [ ] Replace simplified TimesNet baseline before any final EXP-2.2 fair-baseline claim.
- [ ] Author confirmations (now the gating unknowns): which corpus/split the paper's headline
      used (DAIC-WOZ 107/82 vs E-DAIC official; single test vs CV vs dev), exact 88-dim windowing
      /openSMILE config, test-eval authorization. Any of these could still change the picture but
      must be pre-specified, not chosen post-hoc.
- [ ] (Optional, legitimate) Equal-budget hyperparameter search for ALL 6 models on dev before
      any final claim — never tune D-MSTCN alone.
- [ ] Rent 8-GPU server for Phase 6 scaling (only after quality results locked, ~3 days).
