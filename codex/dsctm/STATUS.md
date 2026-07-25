# D-MSTCN Campaign Status

Last updated: 2026-07-20 04:00 IST

Operational continuation instructions for Claude are maintained in `instruction.md`.

## Active checkpoint

- Isolated checkout: `codex/dsctm`, branch `experimentation2`, based on local commit
  `03cc9ec` from the candidate `experimentation1` implementation.
- Shared environment verified: PyTorch 2.6.0+cu124 sees the RTX 4060 Ti.
- Gate 0 independently rerun: 11/11 tests pass and EXP-0.1–0.4 execute successfully.
- Gate 1 reproduced StudentLife and both DAIC variants against live `/mnt` paths.
- A mask-aware five-seed DAIC-WOZ rerun completed; D-MSTCN ranked 3/6 and no
  paired participant-bootstrap interval excluded zero.
- The confirmatory rerun was repeated from committed code `07a78a1` with identical
  values; its JSON SHA-256 is `b0f0427d...a74f` and reliability artifacts exist.
- EXP-6.1 single-device FP32 inference profiling completed with 30 synchronized
  repetitions per representative input; raw timings are preserved.
- Gate 0 now includes per-branch perturbation RF checks and finite AMP loss/gradient
  checks; 14/14 tests pass.
- User approved continuation of the approximately 17 GPU-hour campaign. Corrected
  StudentLife headline evaluation is the active first job.
- A faithful TimesNet classification baseline is implemented from the official THUML
  source pinned at `4e938a1`; its CPU shape test passes. It will replace the historical
  simplified placeholder in subsequent fairness runs.
- Phase-5 ablation execution now preserves a partial artifact after every completed
  variant and reports paired Hodges–Lehmann shift, rank-biserial effect, exact Wilcoxon
  reachability, and Holm/BH adjusted p-values for the prespecified comparison family.
- The first corrected StudentLife launch was cancelled at 16 minutes with no reported
  metric after audit found `_ffill` backward-filled a leading missing prefix from a
  future observation. That run is invalid and excluded for a prespecified correctness
  reason, not model quality. Strictly causal forward-fill and a versioned v2 cache are
  now being validated before restart.
- EXP-1.3 preprocessing robustness is implemented for causal forward fill, train-fold
  mean, zero, and mask-aware zero conditions on identical grouped folds. It preserves
  per-condition checkpoints and data hashes; targeted tests pass.
- Immutable per-fold/per-seed run artifacts are integrated for upcoming preprocessing
  and ablation phases: run identity, resolved config, environment, metrics, curve,
  de-identified predictions, logs, and checkpoint-retention disclosure. Registry tests pass.
- EXP-3.3 controlled delay code is ready with short/medium/long XOR dependencies,
  grouped evaluation, branch controls, immutable fit artifacts, and success/falsification
  criteria recorded before execution. Generator tests pass.
- EXP-2.2/2.3 fair tuning is ready: six models receive eight model-specific dev trials,
  test is inaccessible to the search routine, configurations are frozen before five-seed
  confirmation, and official split membership is represented only by a SHA-256 hash.
- Participant-only DAIC-WOZ eGeMAPS extraction is implemented from released transcript
  intervals. It excludes Ellie, concatenates participant clips chronologically without
  synthetic gaps, emits no IDs in its aggregate manifest, and has a parser regression test.
- The immutable registry is now restart-safe for an already completed identical fit and
  records failed fair-tuning trials as `model_failed` directories with configuration,
  environment, failure class, and stderr instead of silently dropping them.
- Phase 5 now implements all single-branch, pair-branch, and full-branch combinations;
  dynamic, fixed-mean, and learned-static CSAG; half/double attention temperature; and
  no/global/subject/equal-parameter-global FiLM controls. Targeted tests pass. This expands
  the phase from 105 to 210 fits, so the expanded phase requires revised compute approval.
- Future EXP-4.1 executions now preserve every completed model/seed/fold fit in the
  immutable registry, in addition to the per-model partial summary. The already-running
  corrected job began at 22:19:20 IST from launch commit `10b6c48` and continues under
  that revision; later durability/Phase-5 commits must not be attributed to its result.
- The mathematical record now explicitly distinguishes historical fixed-protocol results
  from pending equal-budget tuning, marks the old StudentLife row as leakage-quarantined,
  and does not imply that single-process TCP utilities prove distributed optimization or
  multi-GPU scaling.
- The complete current regression suite passes 27/27 on CPU at handoff revision
  `d8a7b93`. This validates the post-launch implementation; the live EXP-4.1 process
  remains correctly attributed to its separate launch revision `10b6c48`.
- Direct corrected-cache audit confirms 2160 finite 60x8 windows, 46 participants,
  class counts 578/973/609, zero subject overlap in all folds, content hash
  `a9cbaa3a22c2bf4e`, and split hash `6208d08f0b8db52b`. Backward-compatible loading now
  recovers the unambiguous semantic version `studentlife-v2-causal_ffill`; future headline
  outputs embed semantic version and content hash.
- A fail-closed EXP-4.1 result auditor is ready. It requires the exact six-model family,
  seeds 0/1/2, five finite in-range fold values per model, recomputed fold means, valid
  confidence intervals, complete paired comparisons, expected split/data hashes, and emits
  a SHA-256 receipt. It will run only after the final corrected JSON exists.
- Corrected EXP-4.1 reached its first durable seed boundary at 23:30 IST: D-MSTCN
  seed 0 pooled macro-F1 is 0.3376 after all five grouped folds. This is explicitly
  provisional (1/3 D-MSTCN seeds), preserved in a live log, and not used for ranking.
- D-MSTCN seed 1 completed all five folds at 01:15 IST with pooled macro-F1 0.3524.
- Corrected EXP-4.1 **completed** at 03:58 IST (PID 59422 exited normally) and **passed
  the fail-closed audit** at 03:59 IST. All six models, seeds 0/1/2, five grouped folds.
  Final JSON `studentlife_headline_corrected.json` SHA-256
  `abf7079fe189cd7b53239aebbbd3bcd4a7608a8412010ecefd5589c3734f8a3a`; audit receipt
  `studentlife_headline_corrected_audit.json` (`checks_passed: true`, no errors);
  split hash `6208d08f0b8db52b`; independently audited cache hash `a9cbaa3a22c2bf4e`.
  Pooled macro-F1 (mean ± std over seeds; fold-level 95% CI):
  - transformer: 0.3675 ± 0.0047 [0.3528, 0.3733]
  - itransformer: 0.3612 ± 0.0098 [0.3447, 0.3704]
  - timesnet: 0.3493 ± 0.0073 [0.3321, 0.3546]
  - **D-MSTCN: 0.3428 ± 0.0067 [0.3142, 0.3539] — ranks 4th of 6**
  - temporal-cnn: 0.3243 ± 0.0077 [0.3040, 0.3342]
  - lstm: 0.2970 ± 0.0041 [0.2664, 0.3188]
  Paired D-MSTCN-vs-baseline family (5 grouped folds, two-sided exact Wilcoxon, Hodges–
  Lehmann shift): D-MSTCN is numerically below transformer (HL −0.036, p=0.0625),
  itransformer (HL −0.020, p=0.0625) and timesnet (HL −0.018, p=0.625), and above
  temporal-cnn (HL +0.023, p=0.3125) and lstm (HL +0.040, p=0.0625). **No comparison is
  statistically resolvable**: with 5 non-zero paired folds the minimum achievable two-sided
  exact Wilcoxon p is 0.0625 (`significance_reachable: false` for every pair), so p<0.05 is
  unreachable in either direction. This corrected, leakage-safe result confirms the
  campaign's standing finding: D-MSTCN shows **no reproducible headline advantage** on
  StudentLife; plain transformer-family baselines numerically outperform it. Preserved
  as-is per the honesty rules. Limitation: the launch revision `10b6c48` predates embedding
  the data hash in output, so `embedded_data_hash` is null and the run is tied to its cache
  by the independently audited hash, not an in-file field (auditor flags this and still
  passes on all other checks).
  Seed-2 individual pooled value for D-MSTCN was not durably captured (stdout→pts); only the
  runner-emitted model-level aggregates are recorded — no per-seed value was inferred.
- The mathematical record now covers the complete implemented pipeline: biased Linear
  maps and exact FiLM initialization, masking/normalization, PyTorch weighted-CE reduction,
  optimizer/selection, TimesNet fusion, all Phase-5 controls, metrics, bootstrap/paired
  inference, multiplicity, splits, determinism, registry identity, and result admission.
  It explicitly records that held-out participants share trained FiLM row 0 (no test-time
  individualized adaptation) and that TCP/distributed/FedAvg performance is unavailable.
- Existing headline results are preserved as negative evidence and remain marked
  imported until their artifacts and evaluation path are fully audited.

## Phase dashboard

| Phase | Status | Evidence / limitation |
|---|---|---|
| Gate P | needs refresh | Original preflight predates dataset discovery |
| Gate 0 correctness | complete for single-device model behavior | RF gradient+perturbation, causality, masks, checkpoint, determinism, AMP finite; distributed optimizer sync remains blocked/incomplete |
| Gate 1 provenance/leakage | complete for split overlap/provenance | StudentLife split hash `6208d08f0b8db52b`; DAIC official splits have zero participant overlap |
| Phase 2 fair baselines | rerun pending | Faithful TimesNet is implemented; historical simplified-TimesNet comparisons remain quarantined |
| Phase 3 synchronization | partial | single-process controller only; optimizer-state and failure semantics incomplete |
| Phase 4 headline | confirmatory DAIC-WOZ rerun complete; corrected StudentLife rerun **complete + audit-passed** | padding-aware DAIC-WOZ: D-MSTCN 3/6, no resolved pairwise advantage; corrected StudentLife: D-MSTCN 4/6 (0.3428), beaten numerically by transformer/itransformer/timesnet, no pairwise difference statistically resolvable (5-fold Wilcoxon min p=0.0625). No reproducible headline advantage. |
| Phase 5 ablations | runner ready | recovery checkpoints and paired multiplicity-corrected statistics implemented; results pending |
| Phase 6 systems | EXP-6.1 complete; 6.2–6.5 blocked | one-GPU profile complete; only one physical GPU locally |
| Manuscript/response | blocked in part | no editable manuscript source or original decision letter |

## Newly identified correctness gaps

1. TCP optimizer-state/partial-failure semantics remain incomplete and true distributed
   behavior cannot be tested on one GPU.
2. Model checkpoints are not retained on disk; immutable run artifacts explicitly disclose
   this and preserve selected predictions, curves, metrics, config, and environment.
3. Transformer training emitted a nondeterministic memory-efficient-attention warning;
   exact reproducibility is not established for that baseline kernel.

## Next actions

Corrected StudentLife headline evaluation is **complete and audit-passed** (see above).
Remaining approved, locally feasible queue (run one GPU experiment at a time, confirm exact
script/config/output before each launch — see `instruction.md`):

1. `scripts/run_exp13_preprocessing.py` — EXP-1.3 four leakage-safe preprocessing conditions.
2. `scripts/run_exp33_delay.py` — EXP-3.3 controlled delay/XOR task.
3. `scripts/run_exp22_fair_tuning.py` — EXP-2.2/2.3 equal 8-trial dev search, then frozen
   5-seed confirmation (DAIC-WOZ; test inaccessible during search).
4. `scripts/build_daicwoz_participant_egemaps88.py` — participant-only representation, then
   fair all-model refinement after verifying cache/split provenance.

Not yet launched — pending user go-ahead (each is a multi-step GPU job). The expanded
210-fit Phase 5 still requires separate revised compute approval. No git commit/push has
been performed for this monitoring pass.
