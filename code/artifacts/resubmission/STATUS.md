# D-MSTCN Experiment Status

Last updated: 2026-07-19 21:15 IST

## Current state

- Campaign resumed after discovery of the complete datasets under
  `/mnt/adissd/phd/dsctm-resubmission/dataset`.
- A candidate authoritative implementation was discovered at
  `/mnt/adissd/phd/dsctm-resubmission/cold/dsctm` on branch `experimentation1`.
- The candidate contains Gate 0/1 evidence and completed matched-budget headline
  runs. Its latest results are negative for the manuscript's headline accuracy
  claim; those results will be preserved, audited, and never rewritten.
- Current action: create an isolated Codex checkout, audit code/results, and define
  the remaining reviewer-complete experiment campaign.

## Status conventions

- `complete`: verified by an artifact or test named in this file.
- `in_progress`: actively being implemented or checked.
- `blocked`: cannot be completed without a named external input or capability.
- `not_started`: no claim of completion.

## Phase dashboard

| Phase | State | Evidence / next action |
|---|---|---|
| Gate P | complete, needs dataset refresh | Existing files in `artifacts/resubmission/`; update stale dataset findings |
| Gate 0 correctness | candidate evidence exists | Re-run from isolated checkout and audit assertions |
| Gate 1 data integrity | candidate evidence exists | Re-run manifests/leakage checks against current dataset paths |
| Phase 2 baselines | candidate matched-budget runs exist | Audit baseline fidelity; simplified TimesNet is not claim-ready |
| Phase 3 synchronization | not_started | Audit scaffold and identify single-GPU vs multi-GPU scope |
| Phase 4 headline | candidate runs complete | Preserve negative findings and validate raw artifacts |
| Phase 5 ablations | not_started | Implement reviewer-complete ablations after audit |
| Phase 6 systems | blocked locally | One physical RTX 4060 Ti cannot establish 2–8 GPU scaling |
| Manuscript/response | blocked in part | Editable manuscript and original decision letter are not present |

## Integrity note

No value from the candidate implementation is accepted as verified until its code,
configuration, dataset split, and raw result artifact have been checked in the
isolated Codex checkout.
