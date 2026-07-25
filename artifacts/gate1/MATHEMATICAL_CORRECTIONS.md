# Gate 1 — Mathematical Corrections Required in the Manuscript

Every correction below is derived from the implementation at `codex/dsctm/` and pinned by a
test in `codex/dsctm/tests/`. None is a matter of opinion; each is recomputable on demand.

Evidence: `artifacts/gate1/gate1_tests.{xml,log}` — 59 passed.

---

## C-1 · Receptive fields are 61 / 481 / 1921, not 47 / 383 / 1535

**Manuscript:** §III-B-2, Fig. 1 branch labels, Fig. 2 caption, and the §III-F complexity
constant `O(1536·T)`.

**Correct values**, from Eq. (2)'s two-convolution residual block with `K = 3`:

    R = 1 + 2(K-1) · Σ r_l

| Branch | Dilations | Σr | Correct RF | Manuscript | One-conv RF (for reference) |
|---|---|---:|---:|---:|---:|
| SSB | 1, 2, 4, 8 | 15 | **61** | 47 | 31 |
| MSB | 8, 16, 32, 64 | 120 | **481** | 383 | 241 |
| LSB | 32, 64, 128, 256 | 480 | **1921** | 1535 | 961 |

**Where the printed numbers came from.** They satisfy `6·r_max − 1` exactly
(6·8−1 = 47, 6·64−1 = 383, 6·256−1 = 1535). No standard dilated-TCN derivation produces
`6·r_max − 1`. This is a formula error, not evidence of a different block design — the
implementation matches Eq. (2) as printed.

**Consequential edits:** the complexity constant `O(1536·T)` derives from 1535+1 and must become
`O(1922·T)` (or be restated from the corrected RF). Fig. 2's shaded support regions must be
redrawn. The seconds/minutes/hours framing in §III-A-(i) should be restated against the corrected
spans and the *actual* sampling rate of each corpus.

**Pinned by:** `test_receptive_fields_are_61_481_1921_derived_from_implementation`,
`test_measured_rf_matches_two_conv_formula` (empirical gradient support),
`test_manuscript_printed_rf_is_wrong`.

---

## C-2 · Per-subject adapter cost is `d_s` = 8, not `2D` = 256

**Manuscript:** §III-B-4 — *"The adapter adds 2D parameters per subject — a negligible overhead."*

**Correct statement.** Adding one subject grows the model by exactly `d_s` parameters: one row of
the embedding table. The vectors γ, β ∈ R^D are **generated** by a shared MLP
(Eqs. 7–8: `γ(e_s) = W_γ ReLU(W₁ e_s + b₁) + b_γ`) and are activations, not per-subject storage.

The manuscript overstates per-subject cost by 32× at the default `D = 128`, `d_s = 8`.

Measured directly: `FiLMAdapter(D=128, n_subjects=51)` minus `FiLMAdapter(D=128, n_subjects=50)`
= **8** parameters, for `d_s ∈ {4, 8, 16}`. Same result at whole-model level.

**Pinned by:** `test_per_subject_adapter_cost_is_d_s_not_2D`,
`test_full_model_per_subject_growth_is_d_s`.

---

## C-3 · Eqs. (3)–(4) collapse to a single affine map

**Manuscript:** §III-B-3, the Cross-Scale Attention Gate.

    Z = W_z·[H_s;H_m;H_l] + b_z      (3)
    A = W_α·Z + b_α                  (4)

There is no nonlinearity between them, so `A = (W_α W_z)·[H_s;H_m;H_l] + (W_α b_z + b_α)` — one
affine map R^{3D} → R^3. The "learned gate" has the expressive power of a single linear layer
followed by a softmax; the intermediate 3D→3D projection adds 3D×3D + 3D parameters (147,840 at
D = 128) that buy no representational capacity.

**This is a property of the published equations**, and the implementation is faithful to them.
Reviewer R3 attacked the novelty of the fusion mechanism; this will be found if it is not
pre-empted.

**Recommended handling.** State the collapse explicitly, then let Gate 6 decide empirically:
`linear_csag` (faithful, default) and `nonlinear_csag` (declared activation between the
projections) are both implemented and will be ablated on identical folds. Report what the
nonlinearity is actually worth rather than claiming capability the printed equations do not have.

**Pinned by:** `test_default_csag_is_manuscript_faithful_and_unchanged`,
`test_linear_csag_alias_is_numerically_identical_to_default` (atol=0, rtol=0),
`test_nonlinear_csag_is_a_distinct_declared_variant`.

---

## C-4 · Significance markers in Table 2 are unreachable as printed

**Manuscript:** Table 2 and §V — `† p < 0.05 (Wilcoxon signed-rank, 5 seeds)`.

With n = 5 paired non-zero differences, the **minimum attainable** two-sided exact Wilcoxon
signed-rank p-value is 2/2⁵ = **0.0625**. No configuration of the data can produce p < 0.05.
Every `†` in Table 2 is unattainable regardless of the underlying numbers. The same holds for the
5-fold StudentLife protocol.

**Required change.** Remove the significance markers. Report bootstrap confidence intervals and
effect sizes (Hodges–Lehmann shift, rank-biserial correlation) with the multiplicity family
declared in advance and Holm/BH correction applied. Both are already implemented in
`eval/statistics.py`; Gate 5 makes them the primary inference.

**Pinned by:** `test_wilcoxon_n5_two_sided_cannot_reach_significance`,
`test_significance_reachable_at_n6`.

---

## C-5 · The split shipped with the reviewer package is E-DAIC, not DAIC-WOZ

**Manuscript:** §IV-A — *"DAIC-WOZ [2]: 189 semi-structured clinical interview sessions ...
We use the standard 107/82 train/test split [23]."*

`reviewer-package/data/` contains **163 / 56 / 56 = 275** participants with zero pairwise overlap
— the **E-DAIC (AVEC-2019)** partition. Its own `PROVENANCE.md` states the files "derive from the
USC E-DAIC distribution."

Two separate problems in the manuscript sentence:

1. **Corpus identity.** 189 sessions is DAIC-WOZ; 275 is E-DAIC. The paper must say which was
   used, and the reproducibility package must ship the matching split.
2. **107/82 is not a train/test split.** DAIC-WOZ's official AVEC-2017 partition is
   107 train / 35 dev / 47 test. "107/82" is train versus dev+test **merged** — which is what
   reviewer R6 objected to. Whichever corpus is used, report the three-way official partition and
   state that dev drives selection while test is evaluated once.

**Pinned by:** `test_official_edaic_split_files_are_disjoint_and_correctly_sized`,
`test_shipped_split_is_not_the_manuscript_107_82_partition`.

**Still open (needs the author):** which corpus produced the manuscript's reported numbers. This
determines what Gate 5 runs and cannot be resolved from code.

---

## C-6 · Claims that cannot be made until Gates 7–9 execute

Not corrections yet — flagged so they are not carried forward unexamined.

| Manuscript claim | Location | Status |
|---|---|---|
| Scalability efficiency η ≥ 0.81 at 8 nodes | Abstract, Table 3, Fig. 6 | No distributed code has ever run. Unsupported. |
| 57 % lower per-epoch time at 8 nodes | Abstract, §V-B | Unsupported. |
| Communication volume 187 MB/epoch, 2.66× lower than DataParallel-LSTM | Table 2, Fig. 5 | Estimated, never instrumented. |
| N = 16 on an eight-GPU server | Table 3 | Node/GPU/rank counts conflated (tracker T2-07). |
| Theorem 1 bounded-staleness convergence | §III-D | Proof sketch only; TCP is a single-process counter simulation. Gate 11 decides retain-or-replace. |
| "Standard DDP violates causal temporal ordering" | Abstract, §I-B P2 | Never demonstrated. Gate 3/7 provides the control that would make this testable. |

---

## Summary for the response letter

| ID | Correction | Certainty |
|---|---|---|
| C-1 | RF 47/383/1535 → **61/481/1921**; complexity constant follows | Proven, recomputable |
| C-2 | Adapter cost `2D` → **`d_s`** (32× overstatement) | Proven, recomputable |
| C-3 | CSAG Eqs. (3)–(4) collapse to one affine map | Proven algebraically; empirical ablation queued |
| C-4 | Table 2 `†p<0.05` unattainable at n=5 | Proven arithmetically |
| C-5 | Corpus/split description inconsistent with shipped data | Proven for the shipped files; corpus choice needs the author |
| C-6 | All distributed claims unsupported to date | Pending Gates 7–10 |
