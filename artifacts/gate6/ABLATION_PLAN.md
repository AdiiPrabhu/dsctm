# Gate 6 — Fresh Ablation Campaign

Status: **PLANNED AND TESTED, NOT EXECUTED.** 78 tasks, `--array=0-77%4`.
Evidence: `receptive_fields.json`, `../gate12/full_suite.xml`.

## Families (78 tasks = 26 variants × 3 seeds)

| Family | Variants | Tasks | Tracker |
|---|---:|---:|---|
| Branch combinations (all 7) | SSB, MSB, LSB, SSB+MSB, SSB+LSB, MSB+LSB, all | 21 | E4-09 |
| Dilation schedules | original, compressed, expanded, uniform, duration_aligned | 15 | **E4-08** |
| Fusion | mean, static, `linear_csag`, `nonlinear_csag`, temp×0.5, temp×2 | 18 | E4-10 |
| Personalization | no-FiLM, subject, global, parameter-matched global | 12 | E4-11, E4-12 |
| Preprocessing | causal_ffill, zero, train_mean, mask_aware_zero | 12 | V3-04, V3-05 |

E4-08 was unaddressed by both original candidates. It is now covered.

## Receptive fields — derived, never typed

Every value from `Branch.theoretical_rf_two_conv()`, `R = 1 + 2(K−1)·Σr`.

| Schedule | SSB | MSB | LSB |
|---|---:|---:|---:|
| original | 61 | 481 | 1921 |
| compressed | 41 | 161 | 641 |
| expanded | 341 | 2721 | 10881 |
| uniform | 17 | 129 | 513 |
| duration_aligned | 61 | 961 | 3841 |

### Finding G6-1 — the manuscript's printed values follow `6·r_max − 1`

| Branch | Derived | Printed | `6·r_max − 1` |
|---|---:|---:|---:|
| SSB | 61 | 47 | **47** |
| MSB | 481 | 383 | **383** |
| LSB | 1921 | 1535 | **1535** |

All three match exactly. This is a specific, identifiable formula error, not a typo or a
different block design — worth saying plainly in the response letter.

### Finding G6-2 — on StudentLife, MSB and LSB exceed the input window

StudentLife windows are **T = 60** (60 one-minute bins). Against that:

| Branch | RF | Span at 1 min/step | Fits in T=60? |
|---|---:|---|---|
| SSB | 61 | 61 min | marginally (61 > 60) |
| MSB | 481 | 8.0 h | **no — 8× the window** |
| LSB | 1921 | 32.0 h | **no — 32× the window** |

The medium and long branches cannot observe minutes-to-hours structure on StudentLife
because **no such structure is present in a 60-step input**. Their effective context is
capped at 60 steps regardless of dilation. The manuscript's §III-A claim that LSB captures
"circadian rhythms over hours to days" is unattainable on this dataset as windowed.

This has three consequences:

1. It offers a mechanism for the observed result that branch ablations barely move
   StudentLife macro-F1 — on that corpus the three branches see nearly the same context.
2. The `duration_aligned` schedule in this plan is the honest fix: schedules matched to the
   corpus's actual window length. Its value is now an empirical question.
3. Either the StudentLife windowing must lengthen, or the multi-scale claim must be made on
   DAIC-WOZ (T = 2000 at 0.5 s → SSB 30 s, MSB 4 min, LSB 16 min — all inside the window).

**This is the most substantive scientific finding of the audit so far.** It was not
reachable from either original codebase because neither derived receptive fields against
the actual sequence lengths.

## Blockers

| ID | Blocker |
|---|---|
| B-015 | G6-2 needs an author decision: re-window StudentLife, restrict the multi-scale claim to DAIC-WOZ, or report the limitation. Changes what Gate 6 runs. |
| B-016 | `expanded` LSB has RF 10,881 > DAIC-WOZ T = 2000. It will be padding-dominated; kept deliberately as the upper bracket, and its degeneracy must be reported, not hidden. |
