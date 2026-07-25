# Gate P — Input Gap Report
**D-MSTCN IEEE Access Resubmission** · Generated 2026-07-18

Each gap lists: what is missing, which tracker tasks / experiments it blocks, and the
**exact consequence** if it is not supplied. Gaps are ordered by how much they block.
Nothing below is fabricated or worked around — a missing input yields a `blocked` status,
never an invented result.

Legend for "unblocks": tracker task IDs (see `reviewer_to_experiment_map.csv`) and
master-prompt EXP IDs.

---

## GAP-1 — D-MSTCN implementation, configs, environment lock  ⛔ **P0, blocks the most**
- **Missing:** model/training/eval source, `config_resolved` files, seeds, dependency lock,
  container recipe. No `.py`/notebook of any kind is in the package.
- **Blocks:** EXP-0.1 (RF code confirmation), EXP-0.2 (parameter/FLOP counts), EXP-0.3 (TCP/HOLD/
  sync invariants), EXP-0.4 (causality/shape/determinism tests); T2-02, T2-03, T2-04, T2-05, T2-06,
  T2-11, T2-12; all of Phase 2–6 experiments; G0-09 (freeze/archive).
- **Consequence:** No component behavior can be inspected or tested. The RF equation, parameter
  accounting, staleness-counter semantics, replication rule, complexity, and every "verify from
  code" tracker instruction remain unverifiable. Correctness Gate 0 cannot be passed.

## GAP-2 — Authorized StudentLife / DAIC-WOZ (and SEED) data + official splits  ⛔ **P0**
- **Missing:** raw or feature data, EMA/PHQ-8 labels, official DAIC 107/35/47 split files,
  StudentLife subject IDs, SEED subject/session split + license.
- **Blocks:** EXP-1.1/1.2/1.3 (provenance, leakage, preprocessing), EXP-2.1 (reproduction),
  EXP-4.1/4.2 (headline eval), EXP-5.6 (cold-start), EXP-5.10 (SEED transfer); V3-01…V3-09; E4-12.
- **Consequence:** No leakage audit, no split manifest, no reproduction of any Table 2/3/4/5 value,
  no valid headline evaluation. Data-integrity Gate 1 cannot be passed. SEED transfer claim cannot
  be substantiated → must move to future work unless data+protocol are provided (master-prompt §7.3).

## GAP-3 — Raw experiment logs / predictions / checkpoints  ⛔ **P0**
- **Missing:** per-run metrics, curves, predictions, confusion matrices, timing samples, checkpoints.
- **Blocks:** G0-06 (provenance map), G0-07 (reproduce 68.7), E4-06/E4-17 (timing/comm), EXP-2.1.
- **Consequence:** No submitted number can be traced to a raw artifact. Per master-prompt §2, any
  number not traceable to a raw artifact must be rerun or removed — but rerun is itself blocked by
  GAP-1/GAP-2. Reviewer 1's 58.7/68.7 point can be *rebutted from the PDF* (Table 2 shows 68.7±2.3)
  but not *reproduced from logs*.

## GAP-4 — Manuscript LaTeX/Word source + `.bib` + figure sources  ⛔ **P0 for text application**
- **Missing:** `.tex`/Word, `references.bib`, figure/algorithm source (only the compiled PDF exists;
  the PDF cover lists a "D-MSTCN_COMPILED_from_LaTeX.pdf" main document, but no source is shipped).
- **Blocks:** applying every manuscript edit — T2-01 (notation), title/abstract (W5-01), problem
  statement (W5-03), contributions (W5-02), limitations (W5-06), conclusion (W5-09), F6-01…F6-08
  (formatting, vector re-export, headers), S7-03/S7-04 (highlighted + clean PDFs).
- **Consequence:** Corrections can be *specified exactly* (and are, in the tracker/claim registry) but
  **cannot be applied or recompiled**. No revised PDF, no highlighted-changes PDF, no clean source can
  be produced. Final QA (master-prompt §14, rebuild from clean environment) is not reachable.

## GAP-5 — Original 8-GPU A100 NVLink server (or equivalent)  ⛔ **P0 for systems evidence**
- **Missing:** the multi-GPU host the paper measured. This machine has **1 × RTX 4060 Ti (16 GB)**.
- **Blocks:** EXP-6.1–6.5 (single-device profile, fixed-global-batch scaling, strong/weak scaling to
  N=8, interconnect robustness, failure/recovery); E4-03/E4-04/E4-05/E4-06/E4-17.
- **Consequence:** Even with code+data, no scaling/throughput/efficiency number (S, η, C_vol, 57%
  reduction) can be regenerated here. Systems evidence must come from the authors' original hardware.
  This is independent of GAP-1/2 and cannot be substituted by emulation on a single GPU.

## GAP-6 — DAIC-WOZ test-evaluation right / access status  ⚠ **P0 scope-limiting**
- **Missing:** confirmation of whether the team can evaluate on the DAIC **test** partition (labels or
  authorized evaluator), or only train/dev.
- **Blocks:** EXP-4.2 (authorized test evaluation) and the correct framing of V3-02 (107/82 split).
- **Consequence:** If test access is unavailable, DAIC results must be reported as **development-protocol
  only** and test evidence marked blocked (master-prompt §7.2). The current 107/82 evaluation must not be
  presented as a test result.

## GAP-7 — Author/coauthor factual inputs  ⚠ **P0 governance**
- **Missing:** actual generative-AI tool usage (G0-08), coauthor integrity sign-off (G0-10, D1-08,
  S7-07), and complete author biographies/affiliations/degrees/ORCIDs (F6-05; Hiremath's bio has no
  degree field/year/institution; no second ORCID).
- **Consequence:** Disclosure and biography cannot be self-certified. These stay `blocked` pending
  author confirmation; placeholders are used rather than guessed facts (master-prompt §12).

## GAP-8 — Separate IEEE decision letter & verbatim reviews  ℹ **Low — mitigated**
- **Missing:** the original decision letter / reviewer PDFs. Reviewer content exists only as tracker
  summaries (R1–R7 + Editor) in the "Reviewer Matrix" and per-task rows.
- **Consequence:** Response drafting can proceed against the tracker's summaries, but any *verbatim*
  quotation of a reviewer (master-prompt §11.1) cannot be guaranteed. Confirm exact wording before
  finalizing the response letter. W5-12 (Reviewer 5's suggested citations) needs the original comment
  text to verify each suggested citation.

---

## Summary

| Gap | Severity | If never supplied |
|-----|----------|-------------------|
| GAP-1 code | P0 | Correctness Gate 0 unreachable; ~40 experimental/verify tasks blocked |
| GAP-2 data+splits | P0 | Data Gate 1 unreachable; no reproduction/headline eval; SEED→future work |
| GAP-3 logs | P0 | No result provenance; numbers untraceable |
| GAP-4 source | P0 | No manuscript can be edited/recompiled; no revised/clean/highlighted PDF |
| GAP-5 8-GPU host | P0 | No systems/scaling evidence reproducible here |
| GAP-6 DAIC test | P0 | DAIC limited to dev-only; test evidence blocked |
| GAP-7 author facts | P0 | Disclosure/bio/sign-off cannot be certified |
| GAP-8 verbatim reviews | Low | Response uses summaries; verbatim quotes unverified |

**Bottom line:** GAP-1 + GAP-2 + GAP-4 together mean the resubmission cannot be *executed* from
this package — only *planned*. All planning, analytic, and decision work that does **not** require
those inputs has been done or is in progress (23 of 90 tasks). The remaining 67 tasks are blocked
strictly on the inputs above, and are recorded as `blocked` with the specific missing input, never
as complete.
