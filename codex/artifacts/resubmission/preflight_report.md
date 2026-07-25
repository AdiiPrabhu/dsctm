# Gate P Preflight Report

Generated: 2026-07-18T10:21:38.853799+00:00  
Campaign: `dmstcn-ieee-access-resubmission-gate-p-20260718`

## Outcome

Gate P completed for the supplied input-only package. The audit mapped all **90 tracker tasks** and created a conservative experiment budget. No scientific result is verified: the implementation, data, editable manuscript source, and raw experiment artifacts are absent. Expensive experiments were not launched.

## Repository and inputs

- Root: `/media/adii/adissd/phd/dsctm-resubmission/codex`
- Version control: **not a Git repository**. A campaign branch/tag cannot safely be created without the actual source repository.
- Tracker: `reviews/D_MSTCN_IEEE_Access_Resubmission_Tracker_Completed.xlsx` — SHA-256 `ac58d8189940d905458353e78c2612f3fb833971d6350c507ac561354c491a6a`; 9 sheets; 90 task rows.
- Rejected manuscript: `reviews/D_MSTCN_Rejected_Manuscript.pdf` — SHA-256 `06a9d051ef9d758e28c64455a87e637cef35a3907cbf54f5adac066917f793ec`; 15 pages.
- Original decision/reviewer letter: unavailable; the tracker contains summaries, but Reviewer 5 citation details are explicitly missing.
- Editable manuscript (`.tex`, `.bib`, Word, figures): unavailable.
- Source, configs, tests, environment locks, CI, logs, checkpoints, predictions, and old experiment configs: unavailable.
- ZIP contains only the prompt, start instructions, tracker, and rejected PDF; it does not contain hidden source or result artifacts.

## Environment and hardware

- Host: `adii-MS-7D90`; OS `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`; Python `3.12.3`.
- GPU inventory: `0, NVIDIA GeForce RTX 4060 Ti, 16380 MiB, 580.159.03, 00000000:01:00.0`.
- Topology: one physical GPU on PCIe; no multi-GPU or NVLink path is available locally.
- Storage at preflight: approximately 748 GiB free on the workspace filesystem.
- Dependency status: no project dependency manifest exists. System LibreOffice 24.2.7.2 and Poppler PDF tools are available; `openpyxl` is absent and is not required by this audit.

## Data availability

No StudentLife, DAIC-WOZ, SEED, official split files, evaluator credentials, label documentation, preprocessing caches, or configured dataset paths were found under the supplied tree. Dataset versions, exclusions, class counts, modality dimensions, licenses, and label/evaluator rights therefore remain unverified. Protected data were not searched for outside the workspace.

## Implementation inventory

No implementation is present, so the existence or semantics of MSTCN branches, causal convolutions, CSAG, FiLM/subject embeddings, TCP, HOLD, SAP/partitioning, synchronization, baselines, prediction export, or checkpoint/restart cannot be verified. The PDF describes these components, but manuscript statements are not implementation evidence.

## Existing results and submitted-number provenance

No raw result artifact supports any submitted table or figure. Values in Tables 2–5 and Figures 4–11 are manuscript assertions only. They are quarantined from reuse until mapped to immutable runs. The PDF itself confirms the prose and Table 2 both display `68.7 ± 2.3` for DataParallel-LSTM, resolving only the visual 58.7/68.7 reading question—not the value's experimental provenance.

## Verified contradictions and required claim corrections

1. The PDF reports 1–8 “compute nodes,” but its limitations describe simulation on a single eight-GPU server; it also presents N=16. Reframe to single-server GPU workers and remove unphysical counts unless new physical evidence exists.
2. The PDF calls 107/82 a standard DAIC-WOZ split. Official split files/evaluator rights are absent; quarantine those results and verify train/development/test handling.
3. A two-sided exact Wilcoxon test with five non-zero paired observations cannot attain p<0.05. Remove current dagger significance claims.
4. The written residual block has two causal convolutions. Candidate RFs are 61/481/1921 for K=3 and the written schedules, not the submitted 47/383/1535; code tests are mandatory before reporting candidates.
5. StudentLife uses T=60 at one-minute resolution in the PDF, so realized input evidence is capped at 60 minutes, not hours/days.
6. Generated FiLM gamma/beta values are not persistent per-subject storage. Separate shared generator parameters from an embedding of d_s parameters per subject, subject to code confirmation.
7. The theorem/proof sketch does not establish the claimed convergence guarantee. Remove it and retain only implementation-tested invariants.
8. Ordinary DDP does not inherently violate temporal causality merely because gradients come from different causal windows. Remove or narrowly test that mechanism.
9. SEED transfer, clinical/population-scale, fault-tolerance, robustness, and causal attention interpretations lack supplied evidence and must be removed, narrowed, or moved to future work.

## Tracker and compute summary

- Task priorities: P0=41, P1=38, P2=10, P3=1.
- Conditional plan: P0 675 run/test units, 270.09 estimated GPU-hours, 112.8 GB; P1 1084 units, 497.50 GPU-hours, 171.0 GB; P2 36 units, 21.00 GPU-hours, 8.0 GB.
- These are planning estimates, not measurements. Training times assume approximately 35–40 minutes per run on one RTX 4060 Ti and must be recalibrated with a <=30-minute smoke benchmark after source/data arrive.
- Multi-GPU tests remain blocked on this host regardless of budget.

## Cheap checks executed

- Integrity hashes and file metadata: PASS.
- ZIP content comparison: PASS; no additional source/artifacts.
- Tracker structural extraction: PASS; 9 sheets and 90 mapped task rows.
- PDF text extraction/search: PASS; 15 pages.
- Source/static/model correctness tests: BLOCKED—source absent.
- Dataset/leakage assertions: BLOCKED—data and pipeline absent.
- Manuscript build/visual QA: BLOCKED—editable source absent.

## Gate P status

**BLOCKED for Gate 0 and all scientific reruns; Gate P audit itself is complete.** The smallest unblock bundle is: exact source repository/commit plus configs; editable manuscript/bibliography/assets; immutable old logs/configs; authorized dataset/split paths and DAIC evaluator status; original review letter; and author confirmations listed in `input_gap_report.md`.
