# D-MSTCN — Algorithm, Math, and Results Expectations

Reference for the D-MSTCN model **as implemented in this repo** (`src/dsctm/models/dmstcn.py`,
`blocks.py`, `train/tcp.py`). Equations are numbered to match the manuscript (pp. 7–8). Where the
code deviates from the printed manuscript, the deviation is a **correction** and is flagged inline
(these are the Gate-0 findings T2-02 / T2-03). Nothing here is aspirational — every number is
computed from the code (see `scripts/run_gate0.py`, `pytest`).

---

## 1. Notation

| symbol | meaning | value used here |
|---|---|---|
| $X \in \mathbb{R}^{B\times T\times F}$ | input window: batch, time, features | StudentLife $F{=}8,T{=}60$; DAIC-WOZ $F{=}88$; SEED $F{=}310$ |
| $D$ | shared model width | 128 |
| $K$ | causal-conv kernel size | 3 |
| $r_\ell$ | dilation of block $\ell$ | per-branch schedule (below) |
| $b \in \{s,m,l\}$ | branch index (short/medium/long scale) | 3 branches |
| $L_b$ | # dilated residual blocks in branch $b$ | 4 each |
| $d_s$ | per-subject embedding dim (the **only** per-subject stored cost) | 8 |
| $C$ | # classes | StudentLife 3, DAIC 2 |

Tensors flow as $(B,T,D)$ at module boundaries and $(B,D,T)$ inside convolutions.

---

## 2. Forward pass (Eqs. 1–10)

Pipeline: `X → input proj (1) → {SSB,MSB,LSB} branches (2) → CSAG fusion (3–6) → FiLM (7–9) → pool+MLP head (10) → logits`.

### 2.1 Input projection — Eq. 1
A single shared linear lift into width $D$:
$$H^{(0)} = X\,W_{\text{in}} + b_{\text{in}},\qquad W_{\text{in}}\in\mathbb{R}^{F\times D}.$$
`nn.Linear(F, D)` — shared across branches (code: `input_proj`).

### 2.2 Dilated causal residual block — Eq. 2
Each block $\ell$ (at dilation $r_\ell$) is
$$H^{(\ell)} = H^{(\ell-1)} + \mathrm{Conv}_{r_\ell}\!\Big(\mathrm{GELU}\big(\mathrm{Conv}_{r_\ell}(\mathrm{LN}(H^{(\ell-1)}))\big)\Big).$$
- **LayerNorm** over the $D$ channel dim at block input.
- **Two** causal convs per block at the same dilation, GELU between them.
- Identity **residual** add. Dropout (default 0) after the second conv.
- Each $\mathrm{Conv}_{r}$ is a `CausalConv1d`: left-pad by $(K-1)\,r$, so output $t$ depends only on inputs $\le t$ (strict causality; asserted by EXP-0.4, 0.0 future leakage).

### 2.3 Three temporal branches (multi-scale)
Every branch is $L_b{=}4$ stacked blocks over its own **dilation schedule**:

| branch | dilations $\{r_\ell\}$ | $\sum_\ell r_\ell$ |
|---|---|---|
| SSB (short) | 1, 2, 4, 8 | 15 |
| MSB (medium) | 8, 16, 32, 64 | 120 |
| LSB (long) | 32, 64, 128, 256 | 480 |

Branch outputs: $H_s, H_m, H_l \in \mathbb{R}^{B\times T\times D}$.

### 2.4 Cross-Scale Attention Gate (CSAG) — Eqs. 3–6
Per-timestep soft fusion of the three branches (not a fixed average):
$$
Z = W_z\,[H_s;H_m;H_l] + b_z \quad(3),\qquad
A = W_\alpha Z + b_\alpha \quad(4),
$$
$$
\alpha = \mathrm{softmax}\!\big(A/\sqrt{D}\big)\quad(5),\qquad
H = \sum_{b} \alpha_{:,b}\odot H_b \quad(6).
$$
- $[\,\cdot\,]$ = channel concat ($3D$); $W_z:\,3D\!\to\!3D$, $W_\alpha:\,3D\!\to\!3$.
- Temperature $\sqrt{D}$ by default; $\alpha\in\mathbb{R}^{B\times T\times 3}$ are per-timestep weights over branches.
- **Ablation** `csag_mode="mean"` replaces (3–6) with a fixed mean (EXP-5.2 noCSAG).

### 2.5 FiLM subject personalization — Eqs. 7–9
Per-subject embedding $e_s\in\mathbb{R}^{d_s}$ → **shared** generator MLP → per-channel scale/shift:
$$
h = \mathrm{ReLU}(W_1 e_s),\quad
\gamma = W_\gamma h,\quad \beta = W_\beta h \quad(7\text{–}8),\qquad
H' = \gamma \odot H + \beta \quad(9).
$$
- Initialized to identity ($\gamma\!\approx\!1,\beta\!\approx\!0$) for a stable start.
- **The $(\gamma,\beta)\in\mathbb{R}^{2D}$ are _generated_, not stored per subject.** Only $e_s$ ($d_s{=}8$ floats) is stored per subject — this is the **T2-03 correction** to the manuscript's "$2D$ parameters per subject" (see §4).
- **Ablation** `use_film=False` drops personalization (EXP-5.5).

### 2.6 Head — Eq. 10
Temporal mean-pool → 2-layer MLP → logits (softmax lives in the loss):
$$
z = W_2\,\mathrm{ReLU}\!\big(W_1\,\tfrac1T\!\textstyle\sum_t H'_{:,t,:}\big),\qquad z\in\mathbb{R}^{B\times C}.
$$

---

## 3. Causality & receptive field

### 3.1 Strict causality (Eq. 12)
Left-padding by $(K-1)r$ and no right context ⇒ no future leakage. The distributed variant also
masks future-time gradient contributions:
$$\tilde g_{:,t} = g_{:,t}\cdot \mathbb{1}[t \le t_{\text{current}}]\quad(12)$$
(`tcp.causal_gradient_mask`). Verified: 0.0 leakage, batch-invariance $\sim10^{-7}$, bit-exact determinism.

### 3.2 Receptive field — measured, not assumed (T2-02)
With **two** convs per block, one block at dilation $r$ adds $2(K-1)r$ to the RF, so per branch
$$\boxed{R_b = 1 + 2(K-1)\sum_\ell r_\ell.}$$

| branch | $\sum r_\ell$ | **RF (this code, 2-conv)** | RF (1-conv, ref.) | manuscript printed |
|---|---:|---:|---:|---:|
| SSB | 15 | **61** | 31 | 47 |
| MSB | 120 | **481** | 241 | 383 |
| LSB | 480 | **1921** | 961 | 1535 |

The manuscript's printed **47 / 383 / 1535** match **neither** the 1-conv nor the 2-conv formula →
reported here **as measured** (reviewer T2-02). Note: RF is a *theoretical* upper bound; the
*realized* context is capped by the window length $T$ (StudentLife $T{=}60$; DAIC $T$ capped 2000).
So for StudentLife the LSB's 1921-step RF is never exercised — a point that matters for §8.

---

## 4. Parameter accounting (DAIC-WOZ, $F{=}88$, $C{=}2$, 189 subjects)

Computed from the code (total **1,373,197 ≈ 1.37 M**):

| component | params | note |
|---|---:|---|
| input projection | 11,392 | $88\!\to\!128$ |
| SSB / MSB / LSB (each) | 395,264 | 4 blocks × (2 convs 49,280 + LN 256) |
| CSAG | 148,995 | $W_z\,147{,}840 + W_\alpha\,1{,}155$ |
| FiLM adapter | 10,248 | embed 1,512 + shared gen 8,736 |
| head | 16,770 | |
| **total** | **1,373,197** | |

**Per-subject marginal cost = $d_s = 8$ floats** (one embedding row), *not* $2D=256$. The FiLM
generator (8,736 params) is **amortized/shared** across all subjects. This is the exact T2-03 fix:
adding subject $N{+}1$ costs 8 floats, not 256.

---

## 5. TCP — Temporal Coordination Protocol (Algorithm 1)

A distributed **training** protocol; this repo simulates its bookkeeping in one process
(`train/tcp.StalenessController`) so the *invariants* are testable without the 8-GPU cluster (the
systems *performance* claims still need real hardware — GAP-5 / Phase 6).

Staleness = parameter-version lag $\Delta_b = v_{\text{global}} - v_b$. Per step, per branch:
- **update**: local step applied, $\Delta_b \mathrel{+}= 1$;
- **hold**: if $\Delta_b \ge \delta_{\max}$ → suspend the branch, trigger AllReduce, reset all $\Delta{=}0$;
- **skip**: branch not updating this step.

Precedence: **HOLD takes priority** over the periodic ($t \bmod t_{\text{sync}}=0$) AllReduce; the
bound $\Delta_b \le \delta_{\max}$ is invariant. All four invariants verified in EXP-0.3.
**Kill rule (master-prompt §7):** if a same-architecture sync comparison shows no practically
meaningful benefit, TCP must be narrowed to an *engineering* design and its causal-performance
claim removed.

---

## 6. Training objective

Softmax cross-entropy on the pooled logits. For the imbalanced DAIC corpora, **class-balanced CE**
with weights from **train labels only** (leakage-safe):
$$w_c = \frac{N}{C\cdot n_c},\qquad \mathcal{L} = -\frac1B\sum_i w_{y_i}\log \mathrm{softmax}(z_i)_{y_i}.$$
Opt-in via `cfg["class_weight"]="balanced"` (`trainer._build_loss`). This removed the majority-class
collapse that plain CE caused on E-DAIC (5/6 models); it is a genuine methodological fix, applied
identically to all 6 models.

---

## 7. Expected vs. observed results

### 7.1 What the manuscript expects
A **headline accuracy win**: D-MSTCN ranked 1st, attributed to (i) multi-scale dilated branches,
(ii) CSAG cross-scale fusion, (iii) FiLM personalization, and (iv) efficient TCP multi-GPU training.
For a *positive* result we would need D-MSTCN 1st with a paired uncertainty interval that **excludes
0** against **credible** baselines.

### 7.2 What we observe (same fair matched-budget protocol; 6 models; identical loss/seeds/eval)

| setting | corpus | features | D-MSTCN rank | test macro-F1 | credible significant win? |
|---|---|---|---:|---:|:--|
| EXP-4.1 | StudentLife | 8 sensors | 4/6 | 0.3233 | no (5 folds can't reach $p<.05$) |
| EXP-4.2 | E-DAIC | 23-dim LLD | 2/6 | 0.5529 | no (all paired CIs span 0) |
| EXP-4.2b | E-DAIC | 88-dim func | 3/6 | 0.5222 | no |
| EXP-4.2c | **DAIC-WOZ** | **88-dim func** | **2/6** | **0.4854** | **no** |

On DAIC-WOZ (the corpus **and** features the paper cites), D-MSTCN is **2nd**, behind temporal-cnn.
The **only** paired participant-bootstrap CI above 0 in the entire study is vs **TimesNet**
(Δ +0.079, 95% CI [+0.005, +0.153]) — but TimesNet is the **weakest** model here and a **simplified
placeholder baseline flagged for replacement**, so it supports **no** headline. D-MSTCN also has the
**widest seed spread** of any model (init-sensitive).

**Conclusion:** the negative headline is robust across **3 corpora × 2 feature sets**. D-MSTCN is
1st in none. Its architectural *properties* (strict causality, bounded per-subject cost, multi-scale
RF, branch-parallel training) are correct and verified; its *accuracy superiority* is not supported.

---

## 8. How the result could legitimately become positive / better

Two categories. Everything in §8.1 is honest science (pre-specify it, apply it to **all 6 models**);
everything in §8.2 would be fabrication and is ruled out by the project's no-fabrication rule.

### 8.1 Legitimate levers (worth doing; each needs author input or a fair, pre-registered sweep)
1. **Exact protocol/feature fidelity with the author.** Which corpus + split the headline used
   (DAIC-WOZ 107/82 vs E-DAIC official; single-test vs CV vs dev-only), the exact 88-dim openSMILE
   window/config, and test-eval authorization. We reproduced the *stated* 88-dim setup; an exact
   match could still move ranks. **This is the single highest-value unknown.**
2. **Equal-budget hyperparameter search for ALL six models** on dev (LR, depth $L_b$, width $D$,
   dropout, weight decay, epochs). Every model here currently runs one fixed config. D-MSTCN may be
   under-tuned — *but so may the baselines*; the only honest version tunes all models with the same
   budget and picks by dev. This is the most likely lever to change a rank without cheating.
3. **Stabilize D-MSTCN's variance.** It has the widest seed spread → its *mean* is dragged by bad
   inits. Legit fixes (applied to all models): more seeds, dropout > 0, better init, LR warmup/SWA,
   longer patience. Lowers variance → can raise the mean rank honestly.
4. **Participant-only audio.** DAIC-WOZ `*_AUDIO.wav` includes the Ellie interviewer; restricting
   eGeMAPS to participant turns via the transcript could sharpen the depression signal. Apply to all
   models; it's a genuine feature-quality improvement, not a D-MSTCN-specific tweak.
5. **Match the task to the inductive bias.** D-MSTCN's edge is *long-range multi-scale* temporal
   structure (RF up to 1921). StudentLife realizes ≤ 60 steps and DAIC windows are short/pooled, so
   the LSB is largely wasted — the benchmark may simply not need what D-MSTCN adds. A dataset/task
   with genuine long-range dependencies (pre-specified success/falsification criteria, EXP-4.x style)
   is the honest place the architecture could actually win.
6. **Decision-threshold / calibration tuning on dev** (all models): pick the operating threshold that
   maximizes dev macro-F1, then freeze it for test. Can lift macro-F1 without touching test.
7. **Report a stronger, fair metric picture.** PR-AUC / balanced accuracy / calibration (ECE) — if
   D-MSTCN is better-calibrated or stronger on the minority class, that is a legitimate, reportable
   strength even absent a macro-F1 win.

### 8.2 Do **NOT** do (these would be fabrication — explicitly ruled out)
- Seed-picking / reporting the lucky seed (the retracted "1st/6 +0.195" was exactly this 2-seed
  artifact).
- Tuning **only** D-MSTCN while baselines stay at defaults.
- Dev-as-test, or merging dev+test (the 107/82 the reviewer already flagged).
- Metric/split shopping after seeing results; post-hoc narrative around the vs-TimesNet CI as if it
  were a headline.
- Weakening baselines (e.g., leaving TimesNet simplified) to manufacture a gap.

### 8.3 Strategic recommendation (the honest, publishable path)
Given four settings all landing the same way, **reframe the contribution away from a headline
accuracy win** toward what *is* verified and defensible:
- **Efficiency / systems:** bounded per-subject cost ($d_s{=}8$ vs $2D$), branch-parallel training,
  measured single-server profile (Phase 6).
- **Causality & determinism:** strict no-future-leakage, bit-exact reproducibility.
- **Personalization:** FiLM as a cheap, principled subject-adaptation mechanism (EXP-5.5 ablation).
- **Methodology / provenance honesty:** the imbalance fix, the RF/param corrections, the corpus and
  feature-version discrepancies — these are real, citable contributions.

That is the framing most likely to survive review, because it is the framing the evidence supports.

---

*Provenance:* all equations verified against `src/dsctm/models/*.py`; RF and parameter counts from
`scripts/run_gate0.py` and a direct instantiation (§4 numbers are exact); results from
`artifacts/resubmission/phase4/{studentlife,daic}_headline*.json` and
`daicwoz_headline_egemaps88.json`. See `HANDOFF.md` for the running log and `SUMMARY.md` for the
full result tables.
