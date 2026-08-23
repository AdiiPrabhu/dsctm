# Gate P — Preflight Report
**D-MSTCN IEEE Access Resubmission**
Generated: 2026-07-18 · Protocol: D_MSTCN_ONE_FILE_MASTER_PROMPT.md · Phase: Gate P (preflight only)

---

## 0. Headline finding (read first)

**The working directory is a documentation/planning package, not the D-MSTCN
repository.** It contains only (a) the master prompt, (b) the completed reviewer
tracker (`.xlsx`), (c) the compiled rejected manuscript (`.pdf`), and (d) a zip that
re-bundles the same three files. There is **no source code, no datasets, no manuscript
LaTeX/Word source, no raw logs, no checkpoints, and no configs.** START_HERE.md itself
lists all of these under "Materials still needed for full execution."

Consequence: the entire empirical programme (Phases 0–6: receptive-field/parameter/sync
verification, leakage audits, baseline reproduction, causal ablations, headline
evaluation, systems scaling) **cannot be executed or verified from this package** and is
marked **blocked**, not failed. What *can* be done now — and has been started — is
scientific planning, PDF-verifiable corrections, analytic (pure-math) checks, the claim
registry, and reviewer-response drafting. 67 / 90 tracker tasks are blocked on missing
inputs; 21 are in progress (decisions + audits determinable from the PDF/tracker); 2 are
evidence-ready (analytic statistics).

A second, independent blocker: even with code + data + authorization, **the systems
experiments cannot run on this host** — see §5.

---

## 1. Repository / working directory

| Item | Value |
|------|-------|
| Path | `/media/adii/adissd/phd/dsctm-resubmission/cold` |
| Git repository | **No** (`git status` → "not a git repository"). Campaign branch/tag actions from master-prompt §5 are not applicable until a repo exists. |
| Uncommitted changes | n/a (no VCS) |
| Source entry points | **None found** — no `.py`, `.tex`, `.bib`, `.ipynb`, Dockerfile, `requirements.txt`, or config files. |
| Artifacts created this gate | `artifacts/resubmission/` (Gate P files only) |

## 2. Input files (SHA-256)

| File | SHA-256 | Role |
|------|---------|------|
| `reviews/D_MSTCN_IEEE_Access_Resubmission_Tracker_Completed.xlsx` | `ac58d8189940d905458353e78c2612f3fb833971d6350c507ac561354c491a6a` | Master tracker (90 tasks, 9 sheets) — **authoritative, only version present** |
| `reviews/D_MSTCN_Rejected_Manuscript.pdf` | `06a9d051ef9d758e28c64455a87e637cef35a3907cbf54f5adac066917f793ec` | Rejected manuscript, 15 pp (paper body pp. 4–15) — **compiled PDF only, no source** |
| `D_MSTCN_ONE_FILE_MASTER_PROMPT.md` | `1c9517ef3a09c469e8b2e98d69d996ca650a65ee0969610257a5301a2ef0990d` | Protocol |
| `START_HERE.md` | `210c3ed338f2f7344356ebd33356f790c257df329c1f330e3d1f7fd975d2bb67` | Package readme |
| `D_MSTCN_Resubmission_Ready_Package.zip` | `bbd988412fba9b02287872c9c53ae2b1ff681985fd726ed2e090e9893aa53d95` | Re-bundle of the four files above (no new content) |

Originals are preserved untouched; all outputs are written under `artifacts/resubmission/`.
No separate IEEE decision letter / verbatim review PDFs are present — reviewer content is
represented only as summarized comments inside the tracker (R1–R7 + Editor).

## 3. Environment & dependencies

| Item | Value |
|------|-------|
| OS | Ubuntu, Linux 6.8.0-101-generic, x86_64 |
| Python | 3.12.3 (system) |
| Scientific stack | **Absent** system-wide: no numpy, scipy, pandas, torch, sklearn, matplotlib, openpyxl, pdf libs. System pip is PEP-668 externally-managed. |
| Tooling bootstrapped for Gate P | venv in scratchpad with `openpyxl` 3.1.5 (to read the tracker) + Python stdlib for analytic checks. No project environment lock exists to reproduce. |
| Deep-learning runtime | **Not installed** (no PyTorch/CUDA toolkit verified). Would need full setup before any experiment. |

## 4. Datasets, splits, labels, evaluators

| Dataset | Status |
|---------|--------|
| StudentLife (48 subjects claimed) | **Not present.** No raw files, features, EMA labels, split manifests, or windowed caches. |
| DAIC-WOZ (189 sessions claimed) | **Not present.** No official participant split files (107/35/47), no eGeMAPS features, no PHQ-8 labels. Test-evaluator access status **unknown**. |
| SEED (EEG transfer) | **Not present.** No data, no license confirmation, no subject/session split. |

No dataset can be provenance-audited, leakage-tested, or evaluated. DAIC test evaluation
right is undocumented. Per master-prompt §7, all three dataset protocols are blocked at
step 1 (provenance/manifests).

## 5. Hardware vs. manuscript claims — a hard reproduction blocker

| | This host (measured) | Manuscript claim (pp. 8, 10) |
|--|----------------------|------------------------------|
| GPUs | **1 × NVIDIA RTX 4060 Ti, 16 GB** | 8 × NVIDIA A100 SXM4, 80 GB each |
| Interconnect | single consumer GPU (PCIe) | NVLink 3.0, NCCL backend |
| CPU / RAM / disk | 24 cores / 46 GB / 748 GB free | (server-class) |

**Even if code, data, and authorization were provided, the systems programme (EXP-6.1–6.5:
single-device profile, fixed-global-batch scaling, strong/weak scaling to N=8, interconnect
robustness, failure/recovery) cannot be reproduced on this machine.** A single 16 GB
consumer GPU cannot instantiate the 8-GPU NVLink topology the paper measures, and it cannot
physically produce N=2/4/8 branch-parallel scaling. Any multi-GPU/scaling number would
require the original 8-GPU server (or an equivalent) — an external resource not available
here. This independently reinforces tracker decision **DEC-01 / D1-01** (reframe to
single-server branch-parallel multi-GPU) and makes clear that the scaling *evidence* itself
must come from the authors' original hardware, not from this environment.

## 6. Implementation-component inventory (master-prompt §4.1)

Every named component is described **only in the manuscript prose/figures**; none is present
as code, so none can be inspected or tested here.

| Component | In manuscript? | Code present? | Verifiable now? |
|-----------|:--:|:--:|--|
| 3 MSTCN branches (SSB/MSB/LSB), dilations {1,2,4,8}/{8,16,32,64}/{32,64,128,256} | ✔ | ✘ | Analytic RF only (see EXP-0.1) |
| Causal dilated conv blocks (Eq. 2, two convs/block, K=3, L_b=4) | ✔ | ✘ | No (needs code for padding/causality tests) |
| CSAG cross-scale attention gate (Eqs. 3–6) | ✔ | ✘ | No |
| FiLM personalization adapter, e_s∈R⁸ (Eqs. 7–9) | ✔ | ✘ | Analytic parameter check only (EXP-0.2) |
| TCP staleness protocol, Δ_b, HOLD, periodic AllReduce (Alg. 1) | ✔ | ✘ | No (needs code for invariants/atomicity) |
| SAP scale-aware partitioner (Eq. 11) | ✔ | ✘ | No |
| Theorem 1 gradient-error bound (Eq. 13) | ✔ | n/a | Reviewed as text → recommend removal (see claim registry) |
| 7 baselines (LSTM, T-CNN, Transformer, TimesNet, iTransformer, DataParallel-LSTM, FedAvg-LSTM) | ✔ | ✘ | No |
| Prediction export / logs | — | ✘ | No |

## 7. Contradictions & correctness issues found from the PDF + tracker

These are established from the compiled PDF and analytic math alone (evidence:
`statistics/gateP_analytic_checks.md`), independent of the missing code:

1. **Receptive field (EXP-0.1 / T2-02).** Block Eq. (2) has two convs/block; the two-conv
   formula `R=1+2(K−1)Σr` gives 61/481/1921, but the manuscript prints 47/383/1535 — which
   match **neither** a clean one-conv nor two-conv derivation. RF value + Fig 1/2 labels need
   code-based correction; no RF number should be asserted until code confirms the block graph.
2. **Scale labels vs. sampling (T2-08).** At StudentLife's 1-min sampling with T=60, all
   branches realize ≤60 steps = 60 min. The "seconds" (SSB) and "hours/circadian" (LSB) labels
   are physically impossible for that dataset. Report theoretical vs. realized context separately.
3. **Statistical impossibility (E4-13/E4-15).** "Wilcoxon signed-rank, α=0.05, 5 seeds": the
   minimum two-sided exact p at n=5 is **0.0625 > 0.05**. Every "† p<0.05 (5 seeds)" marker in
   Table 2 and figure captions is unreachable as printed.
4. **N=16 on an 8-GPU server (T2-07).** Table 3 reports N=16 scalability on hardware with 8
   GPUs; N=16 cannot represent 16 physical resources → remove/relabel.
5. **"Nodes" = single-server GPUs (D1-01/T2-07).** All "compute node" language describes GPU
   workers on one NVLink server; "multi-node/cluster/population-scale" claims are unsupported.
6. **Parameter accounting (T2-03).** "Adapter adds 2D parameters per subject — negligible"
   conflates generated γ/β with stored params; the per-subject cost is d_s=8, the FiLM
   generator is shared. Needs exact code counts.
7. **Theorem 1 (D1-04/T2-09/T2-10).** The proof sketch does not establish convergence of a
   biased, causally-masked gradient estimator; recommend removal, keep only an operational
   version-lag invariant.
8. **η overloaded (T2-01).** η is used for both learning rate and scaling efficiency.
9. **References [25],[32],[37] (G0-01…G0-03).** Flagged for removal/replacement in the tracker's
   Reference Audit; web re-verification pending (not doable offline-certain here).
10. **Reviewer 1's "58.7 vs 68.7" (G0-07).** The submitted Table 2 DataParallel-LSTM row shows
    **68.7 ± 2.3** (confirmed by reading the PDF), so this is a reviewer misread, to be rebutted
    politely — but the value still needs log-based reproduction, which is blocked.

## 8. What was completed in Gate P

- Full inventory + SHA-256 of all inputs; environment & hardware capture.
- Read of the complete tracker (90 tasks × 26 cols, 9 sheets) and the full 15-page manuscript.
- Six Gate P artifacts + `claim_registry.csv` authored under `artifacts/resubmission/`.
- `reviewer_to_experiment_map.csv`: all 90 tasks mapped to experiments, categories, status, blockers.
- Two analytic correctness checks executed and saved (RF derivation; Wilcoxon n=5 impossibility).

## 9. The single decision that unblocks real work

Provide the **D-MSTCN implementation + configs, the manuscript LaTeX/`.bib` source, and
authorized StudentLife/DAIC (and SEED) data paths**, plus **access to the original 8-GPU
server (or equivalent) and DAIC test-evaluation status**. Without the code+data, no
experiment or number can be reproduced; without the source, no manuscript edit can be
applied; without the 8-GPU host, no scaling evidence can be regenerated. See
`input_gap_report.md` for the itemized gaps and their exact consequences.
