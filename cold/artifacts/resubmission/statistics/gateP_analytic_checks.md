# Gate P — Analytic Correctness Checks (no code/data required)

These two checks use only the manuscript's *stated* architecture and evaluation
protocol. They are pure mathematics and require neither the implementation nor the
datasets, so they are valid Gate P evidence. Both are **decision-supporting**, not
final: the receptive-field verdict still requires code confirmation (EXP-0.1).

Generated: 2026-07-18. Source: reviews/D_MSTCN_Rejected_Manuscript.pdf (SHA-256
06a9d051…f793ec), pp. 7–11.

## 1. Receptive field (relates to EXP-0.1, tracker T2-02, T2-08)

Manuscript Eq. (2) block: `H^l = H^(l-1) + Conv_r(GELU(Conv_r(LN(H^(l-1)))))`
→ **two** causal convolutions per residual block, K=3, L_b=4 blocks/branch.

Candidate formula (tracker T2-02): `R = 1 + 2(K-1)·Σ_l r_l`.

| Branch | dilations | Σr | 1-conv RF | 2-conv RF (T2-02) | Manuscript (Fig 1/2) |
|--------|-----------|----|-----------|-------------------|----------------------|
| SSB (short)  | {1,2,4,8}      |  15 |  31 |   61 |   47 |
| MSB (medium) | {8,16,32,64}   | 120 | 241 |  481 |  383 |
| LSB (long)   | {32,64,128,256}| 480 | 961 | 1921 | 1535 |

**Finding.** The submitted 47/383/1535 match neither a clean one-conv nor a clean
two-conv derivation. The two-conv formula gives 61/481/1921. → The RF equation,
Figure 2, and the "seconds/minutes/hours" scale labels must be corrected from a
code-based impulse/gradient measurement before any value is reported. **Do not adopt
61/481/1921 in the manuscript until code confirms the block graph** (per master-prompt
EXP-0.1 and the tracker's "subject to code confirmation" caveat).

### Theoretical vs realized context (input-length cap)

| Dataset | sampling | T | branch | theoretical | realized (cap T) | realized duration |
|---------|----------|---|--------|-------------|------------------|-------------------|
| StudentLife | 1 min | 60 | SSB/MSB/LSB | 61/481/1921 | **60/60/60** | 60 / 60 / 60 min |
| DAIC-WOZ | 0.5 s | 2000 | SSB | 61 | 61 | 30.5 s |
| DAIC-WOZ | 0.5 s | 2000 | MSB | 481 | 481 | 240.5 s |
| DAIC-WOZ | 0.5 s | 2000 | LSB | 1921 | 1921 | 960.5 s |

At StudentLife's 1-minute sampling with T=60, **all three branches are capped at 60
timesteps = 60 minutes of realized evidence**. The Figure 1/2 labels "seconds" (SSB)
and "hours/circadian" (LSB) are physically impossible for this dataset. Report
theoretical and realized context separately (master-prompt §7.1, EXP-0.1).

## 2. Wilcoxon signed-rank feasibility with n=5 (relates to EXP-4.3, E4-13/E4-15)

Manuscript §IV-C: "All significance uses the Wilcoxon signed-rank test at α=0.05";
§IV-D: "5 independent seeds". Table 2 / figures mark † = "p<0.05 ... (5 seeds)".

Smallest achievable **two-sided exact** p for n paired non-zero differences = 2/2^n:

| n (seeds) | smallest two-sided exact p | p<0.05 reachable? |
|-----------|----------------------------|-------------------|
| 4  | 0.1250 | no |
| **5**  | **0.0625** | **NO** |
| 6  | 0.0312 | yes |
| 8  | 0.0078 | yes |
| 10 | 0.0020 | yes |

**Finding.** With 5 paired observations the minimum two-sided exact Wilcoxon p is
0.0625 > 0.05 — significance is **unreachable regardless of the data**. Every
"† p<0.05 (5 seeds)" marker in Table 2 and the figure captions is therefore invalid
as printed. Resolution (E4-13): make the participant/grouped fold the primary unit,
increase seeds only as secondary stability repetitions, declare sidedness/zero-method/
ties/multiplicity, and report CIs + paired effect sizes. Remove the contradictory
markers until a valid test supports them.
