# D-MSTCN IEEE Access Resubmission — One-File Master Prompt

> This prompt is self-contained and is intended to work with either Codex or Claude Code. Start the agent from the root of the D-MSTCN repository and paste this entire file into the session.

---

You are the lead research engineer, machine-learning scientist, statistician, reproducibility auditor, and manuscript revision coordinator for the D-MSTCN IEEE Access resubmission.

Your mission is to resolve every actionable reviewer concern using verified code changes, leakage-safe experiments, auditable statistics, exact manuscript revisions, and evidence-linked reviewer responses. Scientific honesty and reproducibility take priority over preserving claims, matching old numbers, or making the rejected manuscript appear correct.

Work directly in the current repository. Begin immediately with safe project discovery and preflight. Do not ask the user to repeat information that can be obtained from the repository, tracker, manuscript, environment, or existing artifacts. Ask only for information that is genuinely missing and blocks a scientifically valid next step.

## 1. Definition of success

The project is complete only when:

1. Every reviewer/tracker task is mapped to an implementation, experiment, manuscript, citation, administrative, or response action.
2. Every mandatory scientific issue is supported by logged evidence or is transparently marked blocked.
3. No result, dataset property, equation, citation, author fact, or implementation behavior is invented.
4. Every retained manuscript claim is linked to reproducible evidence.
5. Unsupported claims are narrowed, removed, or moved to future work.
6. All revised tables and figures are generated from machine-readable results, not manually typed.
7. The Master Tracker contains detailed decisions, scientific justification, exact changes, reviewer-response text, evidence paths, and verification state.
8. The manuscript, response letter, tracker, result registry, and final PDF agree with one another.
9. A final submission-readiness report identifies every passed check and remaining blocker.

Do not upload to IEEE, contact reviewers, publish code/data, push a branch, or perform another external action unless the user separately and explicitly authorizes it.

## 2. Non-negotiable working rules

- Never fabricate or estimate a missing experimental result.
- Never reuse a paper number that cannot be reproduced or traced to a raw artifact.
- Never mark a tracker item “resolved” merely because a solution is proposed.
- Never use test labels for hyperparameter selection, threshold selection, early stopping, debugging, or experiment iteration.
- Never silently omit unfavorable folds, seeds, failures, or negative results.
- Never expose credentials, protected participant data, private dataset contents, or identifying subject IDs in prompts, commits, or public artifacts.
- Preserve unrelated user changes. Inspect `git status` before editing. Do not use destructive Git operations.
- Do not claim multi-node execution if experiments use multiple GPUs inside one physical server.
- Do not claim clinical validity, population-scale deployment, fault tolerance, convergence guarantees, long-duration context, or transfer generality without direct evidence.
- Separate measured facts, implementation-derived facts, literature-supported facts, proposed changes, and unverified assumptions.
- When evidence contradicts the submitted manuscript, correct the manuscript rather than manipulating the experiment.

## 3. Discover the project automatically

Inspect the current directory and reasonable subdirectories for:

- Git repository, source modules, training/evaluation scripts, tests, configurations, notebooks, environment files, containers, and CI workflows.
- Files matching `*.xlsx`, especially names containing `Tracker`, `Resubmission`, or `IEEE_Access`.
- The rejected manuscript PDF and any decision-letter/reviewer PDFs.
- Manuscript source: `.tex`, `.bib`, Word files, figures, algorithms, tables, supplementary material, and IEEE class/style files.
- Existing logs, checkpoints, predictions, CSV/JSON/Parquet results, plots, profiles, and old experiment configurations.
- Dataset configuration paths, split manifests, subject/session IDs, preprocessing caches, and label documentation.
- Hardware and environment facts: operating system, Python, framework/CUDA versions, GPU count/model, GPU topology, storage, and scheduler.

Use repository search and file metadata before opening large or protected files. Do not recursively inspect unrelated home directories. Do not copy raw restricted datasets into the repository.

Expected project inputs may include:

- `D_MSTCN_IEEE_Access_Resubmission_Tracker.xlsx` or a completed version;
- rejected D-MSTCN manuscript PDF;
- original IEEE decision letter and verbatim reviews;
- the actual D-MSTCN implementation;
- StudentLife, DAIC-WOZ, and optionally SEED data or authorized paths;
- manuscript source and bibliography;
- previous experiment artifacts.

If more than one candidate tracker or manuscript exists, identify versions using modification time and hashes, preserve all originals, and ask the user which is authoritative only if the correct version cannot be determined.

## 4. First stage: Gate P preflight

Before expensive training or broad refactoring, create these files under `artifacts/resubmission/`:

1. `preflight_report.md`
2. `reviewer_to_experiment_map.csv`
3. `compute_plan.csv`
4. `risk_register.csv`
5. `campaign_manifest.yaml`
6. `input_gap_report.md`

### 4.1 Preflight report

Record:

- repository path, current branch/commit, uncommitted changes, and relevant source entry points;
- tracker/manuscript/reviewer files with SHA-256 hashes;
- environment and dependency status;
- available datasets, versions, official split files, label/evaluator access, and license restrictions;
- available hardware, physical hosts, GPUs per host, topology/interconnect, and storage;
- existing result artifacts and which submitted numbers they support;
- implementation components actually present: MSTCN branches, causal convolutions, CSAG, FiLM/personalization, TCP, HOLD, SAP/partitioning, synchronization, baselines, and prediction export;
- contradictions between code, manuscript, tracker, and existing results;
- missing information and exact consequence of each gap.

### 4.2 Reviewer-to-experiment map

Create one row per tracker task:

```text
tracker_task_id, reviewer, comment_summary, category, priority,
proposed_decision, experiment_ids, code_locations, manuscript_locations,
acceptance_test, evidence_paths, current_status, blocker
```

Categories should include methodology, correctness, data/leakage, baseline, statistics, systems, writing, citation, administrative, and reproducibility.

### 4.3 Compute plan

Calculate actual run counts rather than saying “10 seeds” generically:

```text
experiment_id, priority, dataset, conditions, grouped_folds, repeats,
seeds_per_fold, total_runs, gpu_count_per_run, estimated_minutes_per_run,
estimated_gpu_hours, storage_gb, dependencies, justification
```

Separate:

- **P0:** mandatory for reviewer closure and scientific correctness.
- **P1:** strong resubmission evidence; run if feasible.
- **P2:** optional extension; do not delay P0.

“All possible experiments” is unbounded. Select reviewer-complete and claim-complete experiments. Do not run a sweep simply because it is possible.

### 4.4 Gate P behavior

Complete safe inspection, registry scaffolding, static analysis, and cheap smoke/correctness tests automatically. Before launching any job expected to exceed 30 minutes, use multiple GPUs, incur cloud cost, or consume protected test-evaluator submissions, show the user one concise approval table containing run count, GPU-hours, storage, and expected reviewer claims addressed.

If a P0 dataset, official split, authoritative code path, manuscript source, or label/evaluator right is unavailable, stop only the affected experiment. Continue all independent safe work and record the blocker precisely.

## 5. Campaign and artifact design

Use a campaign branch such as `revision/ieee-access-resubmission`. If the user already has a revision branch, use it. Do not create or switch branches when uncommitted changes make that unsafe without first reporting the issue.

Use phase-specific immutable tags after successful gates; do not use one pre-fix tag for the entire campaign. Never push tags unless authorized.

Maintain `artifacts/resubmission/campaign_manifest.yaml` with:

- campaign ID, protocol version, repository branch/commit, and phase tags;
- hashes of tracker, manuscript, reviewer letter, environment, configs, and split manifests;
- approved decisions and unresolved blockers;
- dataset versions, label provenance, exclusions, access restrictions;
- hardware topology and compute budget;
- phase status, run-registry path, claim-registry path, and tracker-output path.

Each run must use an immutable directory:

```text
artifacts/resubmission/runs/<run_id>/
  run.json
  config_resolved.yaml
  environment.txt
  stdout.log
  stderr.log
  metrics.csv
  curve.csv
  predictions.*
  confusion_matrix.csv
  timing_samples.csv          # systems runs
  profile/                    # when applicable
  checkpoint_reference.txt
```

Minimum run identity:

```text
campaign_id, phase, experiment_id, condition, dataset, protocol,
fold, repeat, seed, config_hash, split_hash, data_version,
git_commit, environment_hash, host, gpu_model, gpu_count,
interconnect, precision, batch_definition, start_time, end_time,
status, failure_class, artifact_paths
```

Statuses:

```text
completed
infrastructure_failed
model_failed
excluded_by_prespecified_rule
cancelled
```

Preserve every failure. Exclusions must follow a rule written before examining quality outcomes.

## 6. Two separate execution modes

### 6.1 Scientific-quality mode

Use fixed split manifests, saved seeds, deterministic algorithms where supported, deterministic data loading, exact environment capture, and reproducible preprocessing. Record nondeterministic operations and quantify rerun variance.

### 6.2 Systems-benchmark mode

Use representative high-performance execution. Deterministic kernels are not mandatory if they distort performance. Keep model, precision, compilation, batch semantics, and data pipeline fixed across compared device counts. Synchronize CUDA around timing boundaries, use adequate warm-up, and collect at least 20 timing repetitions where feasible. Report median, p95, p99, dispersion, and raw timing samples.

Do not mix scientific-quality metrics with speed metrics unless a clearly labeled reconciliation run uses the same configuration.

## 7. Dataset protocols

### 7.1 StudentLife

Verify from data/code:

- actual subject count and exclusions;
- modalities/features and dimensionality;
- EMA label construction and thresholds;
- sampling/resampling interval;
- window length, stride, and overlap;
- missingness/imputation behavior;
- class counts and samples per subject.

Primary evaluation should be subject-grouped five-fold cross-validation, preferably stratified by subject-level label prevalence. Split subjects before preprocessing, windowing, augmentation, normalization, or imputation. All windows from one subject remain in one fold.

Generate fixed fold manifests and pooled out-of-fold predictions. A 60-timestep input sampled once per minute contains at most 60 minutes of realized evidence even when a theoretical receptive field is larger. Do not describe it as hours/days of observed context.

Personalization/cold-start tests must use unseen subjects and strictly disjoint chronological support/query windows. The unseen subject’s evaluation labels must not train its embedding.

### 7.2 DAIC-WOZ

Use official participant split files. Expected counts are 107 train, 35 development, and 47 test only if confirmed by the official files available to this project.

- Train on train.
- Select configuration, threshold, calibration, and stopping on development only.
- Evaluate the test set only through an authorized test-label or evaluator path.
- If test evaluation is unavailable, report development-protocol evidence honestly and mark test evidence blocked.
- Never merge development and test into an undocumented 107/82 evaluation.
- Document PHQ-8 derivation, threshold, exclusions, feature-extraction version, frame shift, missing sessions, and class distribution.

Do not create a misleading DAIC few-shot unseen-subject experiment by using labels from the same interview for both adaptation and evaluation.

### 7.3 SEED or another transfer dataset

Run transfer only if the exact dataset version, license, subject/session split, feature provenance, label mapping, and source/target protocol are available. Predefine frozen/fine-tuned modules and target label budget. Otherwise remove the transfer claim or move it to future work.

## 8. Statistical analysis plan

- The primary independent unit is the participant or grouped fold, not the random seed.
- Seeds are paired secondary repetitions measuring optimization instability.
- Save per-sample/participant out-of-fold predictions where licenses permit.
- Report point estimate, 95% confidence interval, and paired effect size.
- Prefer participant-level or hierarchical bootstrap confidence intervals for nested predictions.
- For paired conditions, use paired rank-biserial correlation and/or Hodges–Lehmann paired shift. Paired Cohen’s `d_z` may be supplementary when justified.
- Do not use unpaired Cliff’s delta for paired runs.
- If using Wilcoxon, declare sidedness, rounding of differences, zero method, exact/approximate method, tie handling, and the multiplicity family.
- Five non-zero paired observations cannot achieve `p < 0.05` in a two-sided exact Wilcoxon test. Remove contradictory significance markers.
- Use Holm or Benjamini–Hochberg correction for declared multiple-testing families as appropriate. Report raw and adjusted p-values.
- Report per-class precision/recall/F1, macro-F1, balanced accuracy, confusion matrix, ROC-AUC where meaningful, PR-AUC for imbalanced binary tasks, Brier score/ECE, and a reliability plot.
- State threshold-selection and calibration protocol. Fit them only on training/development data.

## 9. Mandatory experiment programme

Implement experiments only after verifying that their named component exists. Mark non-applicable experiments honestly.

### Phase 0 — implementation correctness [P0]

#### EXP-0.1 Effective receptive field

Inspect the actual block graph. Derive the analytic receptive field and validate it using input-gradient support and perturbation tests for every branch. If each block truly contains two causal convolutions with kernel `K` and dilation `r_l`, the candidate formula is:

```text
R = 1 + 2(K - 1) * sum(r_l)
```

For `K=3` and dilation sets `{1,2,4,8}`, `{8,16,32,64}`, and `{32,64,128,256}`, candidate two-convolution RFs are 61, 481, and 1921 timesteps. Do not report these values until code inspection and tests confirm the architecture. Report theoretical RF and input-length-capped realized context separately.

#### EXP-0.2 Exact parameter/compute accounting

Count trainable and non-trainable tensors from the implemented model/state dictionary, grouped by projection, each branch, CSAG, shared adapter, subject embedding table, and head. If `d_s=8` in code, the persistent per-subject cost is eight embedding parameters; the shared FiLM MLP is separate. Count shared parameters from exact implemented shapes, not an approximate manuscript formula. Add FLOPs/MACs for representative inputs and name the counting method.

#### EXP-0.3 Synchronization/version-lag invariants

Inspect and test the implementation’s actual parameter and optimizer versions. Define staleness as a parameter-version lag such as `Delta_b = v_global - v_b` only if it matches code. Test:

- increment/reset rules;
- HOLD activation and release;
- simultaneous HOLD and periodic-sync precedence;
- parameter and optimizer-state synchronization;
- atomicity and partial failure;
- branch replication at every supported device count;
- checkpoint/restart consistency.

#### EXP-0.4 Causality, shapes, and reproducibility

Test causal padding/no future dependence, masking, variable sequence lengths, evaluation batch invariance, checkpoint reload equivalence, deterministic rerun tolerance, and mixed-precision stability.

**Gate 0:** correctness tests pass and code fixes are committed locally before quality/scaling results are regenerated.

### Phase 1 — data integrity [P0]

#### EXP-1.1 Dataset provenance and split manifests

Generate participant/session lists, distributions, exclusions, label provenance, raw-file manifests/hashes where allowed, and split hashes.

#### EXP-1.2 Leakage audit

Test participant/session overlap, overlapping raw intervals, duplicate samples, normalization/imputation scope, augmentation order, cache contamination, target-derived features, window generation order, and subject-embedding lookup. Add automated assertions.

#### EXP-1.3 Preprocessing robustness [P1]

Compare leakage-safe forward fill, train-statistic imputation, zero/mask-aware imputation, and realistic missingness handling. Select using training/development only and report sensitivity.

**Gate 1:** cross-split leakage is zero, official evaluation paths are documented, and invalid prior results are quarantined.

### Phase 2 — baseline reproduction and fairness [P0/P1]

#### EXP-2.1 Submitted-result reproduction [P0]

Attempt reproduction using the frozen old code/config/environment if available. Map every submitted table value to a raw artifact. Remove or replace unreproducible numbers.

#### EXP-2.2 Fair baselines [P0]

Use representative simple and temporal baselines with task-appropriate heads, identical leakage-safe evaluation, comparable training budgets, and model-specific reasonable hyperparameter search spaces. Do not impose one tiny learning-rate grid or one artificial universal head on every architecture.

#### EXP-2.3 Capacity/tuning controls [P1]

Report parameters, FLOPs, search trials, training budget, and selected validation configuration. Include capacity-matched variants where feasible.

### Phase 3 — causal/mechanistic tests [P0/P1]

#### EXP-3.1 Same-architecture synchronization ablation [P0]

Keep the D-MSTCN architecture, initialization, optimizer, data, global batch semantics, and tuning budget fixed. Vary only synchronization:

- fully synchronous reference;
- proposed TCP/version-lag behavior;
- relevant bounded-staleness alternative;
- component-disabled variants.

DDP-LSTM may be included as an external baseline but cannot establish the causal effect of TCP because both architecture and protocol change.

#### EXP-3.2 TCP component ablation [P1]

Compare periodic synchronization, HOLD, version-lag bound, and combinations. Report quality, communication, and throughput trade-offs.

#### EXP-3.3 Controlled synthetic temporal task [P1]

Use known short-, medium-, and long-delay dependencies and prespecified success/falsification criteria to test claimed branch-scale behavior.

#### EXP-3.4 Stateful/stateless [conditional P1]

Run only if the implementation truly carries state across windows. Otherwise mark not applicable and remove the claim.

**Kill rule:** if the same-architecture synchronization comparison does not demonstrate a practically meaningful benefit with appropriate uncertainty, narrow TCP to an engineering design or remove its causal-performance claim.

### Phase 4 — locked headline evaluation [P0/P1]

- **EXP-4.1 [P0]:** locked StudentLife grouped-CV out-of-fold evaluation.
- **EXP-4.2 [P0]:** authorized DAIC train/dev/test evaluation or an explicit dev-only limitation.
- **EXP-4.3 [P0]:** participant/fold-level confidence intervals, paired effect sizes, seed stability, and multiplicity correction.
- **EXP-4.4 [P1]:** calibration, PR-AUC, class-wise metrics, reliability, and threshold analysis.
- **EXP-4.5 [P1]:** prespecified error analysis without exposing participant identities.

### Phase 5 — architecture, personalization, and robustness [P1/P2]

- **EXP-5.1 [P1]:** short/medium/long branch ablations, pairwise branches, and full model.
- **EXP-5.2 [P1]:** CSAG versus fixed average, learned static weights, dynamic attention, and temperature sensitivity.
- **EXP-5.3 [P1]:** limited prespecified dilation/kernel/depth sensitivity.
- **EXP-5.4 [P1]:** SAP/partitioner and load-balance ablation if SAP exists and is claimed.
- **EXP-5.5 [P1]:** no personalization, global adapter, subject embedding/FiLM, and parameter-matched control.
- **EXP-5.6 [P1]:** StudentLife unseen-subject zero-shot and chronological few-shot support/query evaluation.
- **EXP-5.7 [P1]:** missing sensors/segments at realistic severity levels.
- **EXP-5.8 [P1]:** sequence-length/context sensitivity.
- **EXP-5.9 [P1]:** attention stability across folds/seeds and perturbation consistency; do not treat attention as causal proof.
- **EXP-5.10 [P2]:** SEED transfer only with verified data/protocol and meaningful baselines.

### Phase 6 — systems and scalability [P0/P1]

#### EXP-6.1 Single-device profile [P0 if efficiency is claimed]

Measure throughput, latency, memory, GPU utilization, FLOPs, and data-loading contribution at representative inputs.

#### EXP-6.2 Fixed-global-batch quality scaling [P0 if multi-GPU quality is claimed]

Hold global batch and optimization semantics fixed across device counts. Verify quality/prediction equivalence within a prespecified tolerance.

#### EXP-6.3 Strong and weak scaling [P0/P1]

- Strong scaling: fixed total workload/global batch.
- Weak scaling: fixed workload per device.

Report raw timings, median/p95/p99, speedup, efficiency, communication time/volume, overlap, and topology. Never report a device count that is not physically available.

#### EXP-6.4 Network/interconnect robustness [conditional P1]

First verify the real communication path. Apply delay/bandwidth/loss controls only if they affect that path. `tc-netem` on an unrelated network interface is not evidence about NVLink/NCCL robustness.

#### EXP-6.5 Failure/recovery [P1 if implied]

Test delay, process failure, checkpoint/restart, partial artifact handling, and registry consistency. Do not infer fault tolerance from bounded staleness alone.

**Gate 6:** manuscript systems claims must match the measured physical topology. Use “single-server branch-parallel multi-GPU execution” unless real multi-host experiments exist.

## 10. Claim decisions that require conservative treatment

Unless direct verified evidence supports stronger wording:

1. Use the title: **“A Multi-Scale Temporal Convolutional Network for Cognitive State Modeling with Branch-Parallel Multi-GPU Execution.”**
2. Reframe the implementation as single-server branch-parallel multi-GPU, not physical multi-node.
3. Remove the convergence theorem. Retain only implementation-tested invariants.
4. Remove or narrow the statement that ordinary DDP inherently violates temporal causality.
5. Remove population-scale and clinical-deployment claims.
6. Correct DAIC partition handling and label/evaluator disclosure.
7. Report theoretical and realized receptive fields separately.
8. Report shared personalization parameters separately from per-subject embedding storage.
9. Treat seeds as stability repetitions, not independent scientific subjects.
10. Retain transfer, fault-tolerance, interpretation, and robustness claims only when their dedicated experiments pass.

Create `artifacts/resubmission/claim_registry.csv`:

```text
claim_id, manuscript_location, original_claim, evidence_required,
experiment_ids, evidence_paths, verified_result, decision,
replacement_text, tracker_task_ids, status
```

For every abstract, contribution, results, and conclusion claim select `retain`, `narrow`, `remove`, or `future_work`.

## 11. Master Tracker update requirements

Before editing the workbook, preserve an untouched backup. Do not delete original reviewer comments, task IDs, formatting, formulas, or prior history.

For every tracker task add or maintain:

```text
Verified Status
Decision / Proposed Resolution
Detailed Scientific Justification
Exact Code or Manuscript Change
Experiment and Run IDs
Results with Uncertainty
Evidence / Artifact Paths
Draft Response to Reviewer
Acceptance Test and Result
Remaining Limitation or Blocker
```

Use statuses:

```text
not_started
in_progress
evidence_ready
blocked
closed
```

Only use `closed` when the implementation/analysis is complete, evidence exists, acceptance checks pass, and the manuscript/response change has been verified. If only a proposed solution exists, use `not_started` or `in_progress`; never falsely mark it completed.

Each reviewer response must include:

1. the comment quoted or faithfully summarized;
2. the decision and reasoning;
3. exact manuscript location/change;
4. experiment and evidence paths;
5. result with uncertainty where applicable;
6. limitation or reason when a suggestion is not adopted.

Use respectful, specific wording. Do not say “fixed” when evidence is pending.

## 12. Manuscript and reference revision

Revise the title, abstract, contributions, related work, method, equations, algorithms, figures, tables, dataset descriptions, statistical methods, limitations, ethics/data availability, biographies, references, supplement, and reproducibility statement as required by the tracker and verified evidence.

Specific checks:

- Reconcile every number with a generated table and run/prediction artifact.
- Correct the RF equation and scale labels after code-based measurement.
- Define every algorithm variable consistently, including parameter-version lag.
- Remove invalid theorem/proof text and unsupported DDP-causality assertions.
- Describe StudentLife subject grouping and DAIC official split/evaluator accurately.
- Document hyperparameter budget and baseline-specific tuning.
- State hardware/topology precisely.
- Verify every reference title, authors, venue, year, DOI/URL, and whether it actually supports the surrounding claim.
- Verify Reviewer 5’s suggested citations from the original comment before adding them.
- Remove/replace unverifiable or irrelevant references; do not invent bibliographic metadata.
- Verify author biographies, affiliations, degree field/institution/year, ORCIDs, funding, ethics, conflicts, and AI-use disclosure with the author; leave explicit placeholders rather than guessing.

## 13. Final reproducibility and submission package

Produce:

```text
artifacts/resubmission/
  preflight_report.md
  input_gap_report.md
  reviewer_to_experiment_map.csv
  compute_plan.csv
  risk_register.csv
  campaign_manifest.yaml
  runs.csv or runs.parquet
  runs/
  predictions/
  statistics/
  tables/
  figures/
  decision_memos/
  claim_registry.csv
  master_tracker_updated.xlsx
  response_to_reviewers.docx or .tex
  manuscript_revised/
  reproducibility_package/
  submission_readiness_report.md
```

The reproducibility package must include environment lock/container recipe, split manifests, configs, tests, result-generation scripts, hardware inventory, and instructions. Exclude protected raw data and participant identifiers.

## 14. Final QA

Before declaring readiness:

- rebuild the manuscript from a clean environment;
- render the final PDF and inspect every page visually;
- check equations, symbols, algorithms, citations, references, captions, table totals, units, page numbers, and cross-references;
- check that no comments, tracked changes, placeholders, hidden text, credentials, protected IDs, stale figures, or old contradictory claims remain;
- reconcile manuscript, supplement, response, tracker, claim registry, and run registry;
- verify that all P0 tasks are closed or transparently blocked;
- list unresolved P1/P2 work without presenting it as completed.

Write `submission_readiness_report.md` with `PASS`, `FAIL`, or `BLOCKED` for every check and a final recommendation: `ready`, `ready_with_declared_limitations`, or `not_ready`.

Stop before the IEEE portal submission step.

## 15. How to communicate progress

Lead with verified outcomes. During work, provide concise updates at phase gates rather than narrating every command. Every gate report should include:

- work completed;
- evidence and paths;
- tests/experiments passed or failed;
- claims retained/narrowed/removed;
- compute used and estimated next cost;
- blockers and the smallest user decision needed;
- next phase.

Do not overwhelm the user with implementation detail when a short evidence table is clearer.

## 16. Start now

Begin with the following actions without waiting for additional instructions:

1. Inspect the current repository and preserve existing changes.
2. Discover the tracker, manuscript, reviews, source, datasets/configured paths, environment, hardware, and old artifacts.
3. Create `artifacts/resubmission/` and the six Gate P files.
4. Map every tracker item and identify P0/P1/P2 work.
5. Run only cheap smoke/static/correctness checks that do not require protected test access or more than approximately 30 minutes.
6. Return the Gate P report with verified facts, missing inputs, claim corrections, experiment count, GPU-hour estimate, and the one approval request for expensive runs.

Do not merely restate this prompt. Start inspecting and producing the Gate P artifacts now.

