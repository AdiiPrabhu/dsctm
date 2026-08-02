# Gate P Input Gap Report

## Blocking inputs

| Missing input | Affected work | Exact consequence | Minimum safe resolution |
|---|---|---|---|
| Actual D-MSTCN Git repository, commit, configs, tests | EXP-0.*, all training/systems experiments | Architecture, RF, counts, causality, TCP/HOLD/SAP, baselines, and prediction export cannot be inspected or tested | Provide the local path or a preserved checkout and identify the authoritative revision |
| Editable manuscript source, bibliography, figures/tables | Manuscript revision and final QA | Exact text, equations, citations, vector assets, and PDF cannot be updated or rebuilt | Provide LaTeX/Word source and every referenced asset |
| Old raw logs, checkpoints, configs, predictions, plotting scripts | EXP-2.1 and all submitted numbers | No value in Tables 2–5/Figures 4–11 has provenance | Provide an immutable artifact bundle; otherwise authorize removal/rerun |
| StudentLife authorized path and provenance | EXP-1.*, 2.*, 4.1, 5.* | Subject counts, labels, windows, leakage, folds, and metrics cannot be verified | Provide local path, version, terms, exclusions, and label documentation |
| DAIC-WOZ authorized path, official split files, feature provenance | EXP-1.*, 2.*, 4.2 | The submitted 107/82 protocol cannot be validated or corrected | Provide paths and versioned split/feature manifests |
| DAIC test-evaluator/test-label authorization status | EXP-4.2 | A test claim cannot be produced honestly | State the authorized evaluator route; otherwise accept dev-only reporting |
| Original decision letter and verbatim reviews | Response completeness; Reviewer 5 suggestions | Tracker summaries cannot prove verbatim coverage; three suggested citations are unknown | Provide PDF/email export of the decision and all reviews |
| 2–8 GPU single-server compute access (if systems claims retained) | EXP-0.3, 3.*, 6.2–6.5 | Current one-GPU host cannot test branch replication, scaling, interconnect, or failures | Provide authorized machine/topology and budget; otherwise remove multi-GPU result claims |
| Author confirmations: scope, title, AI use, biography, ethics, funding, conflicts, data terms | Governance and submission metadata | These facts cannot be inferred | Obtain dated confirmations from both authors/institution as applicable |

## Non-blocking but material

- Deadline and compute budget: needed to trim P1/P2 after P0 calibration.
- Container/environment lock: can be rebuilt from source manifests if supplied, but old-result reproduction needs the old environment.
- SEED dataset/license/protocol: omit or future-work the transfer claim if unavailable.

No credentials, participant identifiers, or protected raw records should be copied into this repository. Local paths and access constraints are sufficient for the next audit step.
