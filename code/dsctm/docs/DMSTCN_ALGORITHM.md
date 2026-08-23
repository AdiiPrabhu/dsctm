# D-MSTCN — Algorithm, Math, and Results Expectations

Reference for the D-MSTCN model **as implemented in this repo** (`src/dsctm/models/dmstcn.py`,
`blocks.py`, `train/trainer.py`). The single-process TCP utilities are documented separately in
§5 because they are not part of the optimizer used by the reported model fits. Equations 1--12
retain the manuscript numbering where applicable. Where the
code deviates from the printed manuscript, the deviation is a **correction** and is flagged inline
(these are the Gate-0 findings T2-02 / T2-03). Implemented behavior is stated separately from
prespecified but pending experiments and historical evidence; no planned distributed behavior is
presented as executed.

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
h = \mathrm{ReLU}(W_1 e_s+b_1),\quad
\gamma = W_\gamma h+b_\gamma,\quad \beta = W_\beta h+b_\beta \quad(7\text{–}8),\qquad
H' = \gamma \odot H + \beta \quad(9).
$$
- Initialized exactly to identity ($\gamma=1,\beta=0$) at construction: both output
  weight matrices are zero, $b_\gamma=1$, and $b_\beta=0$.
- **The $(\gamma,\beta)\in\mathbb{R}^{2D}$ are _generated_, not stored per subject.** Only $e_s$ ($d_s{=}8$ floats) is stored per subject — this is the **T2-03 correction** to the manuscript's "$2D$ parameters per subject" (see §4).
- **Ablation** `use_film=False` drops personalization (EXP-5.5).
- Under cross-subject evaluation, held-out participants have no fitted individual row:
  all map to the shared, trained unknown row zero. Thus the evaluation does not perform
  test-time participant adaptation. It measures training-subject conditioning plus the
  learned unknown-subject mapping, not individualized FiLM for a previously unseen person.

### 2.6 Head — Eq. 10
Length-masked temporal mean-pool → 2-layer MLP → logits (softmax lives in the loss):
$$
\bar H_i = \frac{\sum_{t=1}^{T}m_{it}H'_{it}}{\max(1,\sum_{t=1}^{T}m_{it})},\qquad
z_i = W_2\,\mathrm{ReLU}(W_1\bar H_i+b_1)+b_2,\qquad z\in\mathbb{R}^{B\times C}.
$$

Here $m_{it}\in\{0,1\}$ is one only for observed timesteps. The data contract carries
the true sequence length; normalization statistics use valid training timesteps only,
right-padded values are reset to zero after normalization, and padding is excluded from
pooling. This is required because DAIC sequences are right-padded to a common $T=2000$.

---

## 3. Causality & receptive field

### 3.1 Strict causality (Eq. 12)
Left-padding by $(K-1)r$ and no right context implies no future leakage. The repository also
contains and tests a single-process causal-gradient masking utility:
$$\tilde g_{:,t} = g_{:,t}\cdot \mathbb{1}[t \le t_{\text{current}}]\quad(12)$$
(`tcp.causal_gradient_mask`). This is not an implemented or measured distributed optimizer.
The model forward pass is verified at 0.0 future leakage, batch-invariance $\sim10^{-7}$,
and bit-exact deterministic replay under the tested single-device setup.

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

## 4. Parameter accounting (reference configuration, $F{=}88$, $C{=}2$, 189 embedding rows)

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

The manuscript proposes a distributed training protocol, but this repository implements only a
single-process **bookkeeping simulation** (`train/tcp.StalenessController`). It is not called by
`train_model`, `headline_cv`, or `train_select_evaluate`; all reported quality fits use ordinary
single-device Adam. Therefore no distributed-training or scaling result follows from this code.

The simulator describes its integer state as parameter-version lag
$\Delta_b=v_{\mathrm{global}}-v_b$, but it does not instantiate branch parameters or $v_b$.
Operationally, the code performs the following state-machine transitions per branch and step:

- **update**: local step applied, $\Delta_b \mathrel{+}= 1$;
- **hold**: if $\Delta_b \ge \delta_{\max}$ → suspend the branch, trigger AllReduce, reset all $\Delta{=}0$;
- **skip**: branch not updating this step.

Precedence: **HOLD takes priority** over the periodic ($t \bmod t_{\text{sync}}=0$) AllReduce; the
bound $\Delta_b \le \delta_{\max}$ is invariant. All four invariants verified in EXP-0.3.
The named `allreduce` action only increments a simulated global-version counter and resets the
three lag counters. It performs no collective communication. The causal-gradient-mask function in
§3.1 is likewise a tested tensor transformation, not integrated distributed optimization.

---

## 6. Training objective

Softmax cross-entropy on the pooled logits. For the imbalanced DAIC corpora, **class-balanced CE**
uses weights from **train labels only** (leakage-safe). Let
$\widetilde n_c=\max(n_c,1)$ and $\widetilde N=\sum_c\widetilde n_c$; the code computes

$$w_c = \frac{\widetilde N}{C\widetilde n_c},\qquad
\mathcal{L}_{\mathcal B} =
-\frac{\sum_{i\in\mathcal B}w_{y_i}\log \mathrm{softmax}(z_i)_{y_i}}
       {\sum_{i\in\mathcal B}w_{y_i}}.$$

The denominator is PyTorch's weighted-mean reduction. With no class weights, it reduces
to the ordinary batch mean.
Opt-in via `cfg["class_weight"]="balanced"` (`trainer._build_loss`). This removed the majority-class
collapse that plain CE caused on E-DAIC (5/6 models); it is a genuine methodological fix, applied
identically to all 6 models.

Unless a configuration overrides it, fitting uses Adam with $\beta_1=0.9$, $\beta_2=0.999$ and
the configured learning rate and weight decay. After every epoch, `CosineAnnealingLR` updates the
learning rate over `max_epochs` toward `lr_min`. Validation macro-F1 is evaluated after each epoch;
an epoch is considered better only under a strict $>$ comparison. Training stops after
`early_stop_patience` consecutive non-improving epochs. In grouped CV, the stored validation
probabilities are those observed at the best epoch. In official-split evaluation, the complete
best-development state is copied to CPU, restored after training, and the test loader is evaluated
exactly once.

For personalized D-MSTCN fits, training subjects receive indices $1,\ldots,S$ and index zero is
reserved for an unseen subject. Independently for each training minibatch element, the index is
replaced by zero with probability 0.1 (the default `emb_dropout`), so the unknown row is trained.
Every validation or test participant absent from the training split maps to row zero. Baselines do
not receive subject indices.

---

## 7. Expected vs. observed results

### 7.1 What the manuscript expects
A **headline accuracy win**: D-MSTCN ranked 1st, attributed to (i) multi-scale dilated branches,
(ii) CSAG cross-scale fusion, (iii) FiLM personalization, and (iv) efficient TCP multi-GPU training.
For a *positive* result we would need D-MSTCN 1st with a paired uncertainty interval that **excludes
0** against **credible** baselines.

### 7.2 What the completed historical runs observe

These are fixed-training-protocol comparisons across six architectures, not the pending
equal-budget model-specific search in EXP-2.2/2.3. Loss, seeds, split, and evaluation are
shared within each row, but architecture capacity is not parameter-matched. Historical
TimesNet used a simplified placeholder and therefore cannot establish baseline fairness.

| setting | corpus | features | D-MSTCN rank | test macro-F1 | credible significant win? |
|---|---|---|---:|---:|:--|
| EXP-4.1 (quarantined) | StudentLife | 8 sensors | 4/6 | 0.3233 | invalid: leading-prefix backward-fill leakage |
| EXP-4.2 | E-DAIC | 23-dim LLD | 2/6 | 0.5529 | no (all paired CIs span 0) |
| EXP-4.2b | E-DAIC | 88-dim func | 3/6 | 0.5222 | no |
| EXP-4.2c | **DAIC-WOZ** | **88-dim func** | **2/6** | **0.4854** | **no** |

The table records what those immutable historical artifacts contain; it is not final
fair-baseline evidence. On the historical DAIC-WOZ run (the corpus **and** features the
paper cites), D-MSTCN is **2nd**, behind temporal-cnn.
The **only** paired participant-bootstrap CI above 0 in the entire study is vs **TimesNet**
(Δ +0.079, 95% CI [+0.005, +0.153]) — but TimesNet is the **weakest** model here and a **simplified
placeholder baseline flagged for replacement**, so it supports **no** headline. D-MSTCN also has the
**widest seed spread** of any model (init-sensitive).

**Current conclusion:** no completed valid setting supports accuracy superiority. A
padding-aware DAIC-WOZ rerun likewise ranks D-MSTCN 3rd/6 with every paired participant
bootstrap interval spanning zero (see `METRICS.md`). The corrected StudentLife run and
equal-budget fair tuning remain pending, so no stronger final cross-dataset claim is made.
Strict causality, bounded per-subject storage, and multi-scale receptive fields are verified;
branch-parallel or multi-GPU performance is not verified on this one-GPU host.

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
Given the valid completed settings and the padding-aware confirmatory DAIC-WOZ rerun,
the present evidence favors reframing the contribution away from a headline accuracy win
toward what is verified and defensible, subject to the pending corrected/tuned experiments:
- **Efficiency / systems:** bounded per-subject cost ($d_s{=}8$ vs $2D$) and measured
  single-device inference profile. Branch-parallel multi-GPU scaling remains unverified.
- **Causality & determinism:** strict no-future-leakage, bit-exact reproducibility.
- **Personalization:** FiLM as a cheap, principled subject-adaptation mechanism (EXP-5.5 ablation).
- **Methodology / provenance honesty:** the imbalance fix, the RF/param corrections, the corpus and
  feature-version discrepancies — these are real, citable contributions.

That is the framing most likely to survive review, because it is the framing the evidence supports.

---

*Provenance:* all equations verified against `src/dsctm/models/*.py`; RF and parameter counts from
`scripts/run_gate0.py` and a direct instantiation (§4 numbers are exact for the stated reference
configuration); historical results from
`artifacts/resubmission/phase4/{studentlife,daic}_headline*.json` and
`daicwoz_headline_egemaps88.json`. See `HANDOFF.md` for the running log and `SUMMARY.md` for the
full result tables.

## 9. Code-alignment amendments from the Codex audit

The canonical data object stores a valid length $L_i$ for each sample and defines
$m_{it}=\mathbb{1}[t<L_i]$. Let $o_{itf}$ indicate that feature $f$ is not NaN before
normalization. For a training set $\mathcal T$, NumPy `nanmean`/`nanstd` computes

$$
\mu_f=\frac{\sum_{i\in\mathcal T}\sum_t m_{it}o_{itf}X_{itf}}
              {\sum_{i\in\mathcal T}\sum_t m_{it}o_{itf}},\qquad
\sigma_f=\sqrt{\frac{\sum_{i\in\mathcal T}\sum_t
              m_{it}o_{itf}(X_{itf}-\mu_f)^2}
              {\sum_{i\in\mathcal T}\sum_t m_{it}o_{itf}}}+10^{-6}.
$$

Only training samples and valid timesteps enter these statistics. Normalized padding is
set back to zero. The mask-aware head equation in §2.6 excludes padding from the temporal
mean. These operations now match `train/trainer.py`, `data/contract.py`, and
`models/blocks.py`; `tests/test_causality.py` verifies invariance to arbitrary values in a
masked future tail. The $10^{-6}$ is added **after** `nanstd`, exactly as in
`fit_normalizer`; it is not added to the variance inside the square root. If a training
feature has no non-NaN valid value, NumPy yields non-finite statistics and `_make_loader`
maps the resulting non-finite standardized values to zero via `nan_to_num`.

StudentLife grouped CV uses `StratifiedGroupKFold` over window labels with participant IDs
as indivisible groups. This preserves the participant as the independent split unit while
balancing the observed multiclass distribution and fold sample sizes; it does not reduce a
participant's multiclass label history to a rounded mean.

For the StudentLife causal-forward-fill condition, a missing feature at time $t$ is
imputed only from its most recent observed past value:

$$
\widetilde X_{itf}=\begin{cases}
X_{itf}, & X_{itf}\text{ observed},\\
\widetilde X_{i,t-1,f}, & X_{itf}\text{ missing and a value has been observed by }t-1,\\
0, & \text{otherwise}.
\end{cases}
$$

The zero in the last case is subsequently transformed using training-fold normalization.
It is not replaced with the first future observation. The former implementation did that
for a leading missing prefix and was therefore backward-fill leakage; all StudentLife
evidence using data hash `62de62987570bc40` is quarantined. Corrected v2 data hash:
`a9cbaa3a22c2bf4e`.

The corrected cache's semantic version is `studentlife-v2-causal_ffill`. Early v2 NPZs
omitted the `version` field; the loader recovers it only for the unambiguous prespecified
v2 filename and otherwise retains the `studentlife-v1-legacy` label. This metadata recovery
does not alter $X$, labels, groups, timestamps, or the content hash.

EXP-1.3 compares four prespecified treatments of missing point-sensor bins on the same
participant folds: (i) the causal forward fill above; (ii) training-fold mean imputation,
implemented by computing $\mu_f,\sigma_f$ with `nanmean`/`nanstd` over observed valid
training timesteps and mapping a missing standardized value to zero; (iii) literal zero
imputation before training-fold normalization; and (iv) zero imputation concatenated with
eight binary observed-value indicators. Interval-derived occurrence features retain zero
as the defined no-coverage value. No validation/test value contributes to an imputation
statistic.

### 9.1 TimesNet fairness control

The confirmatory `timesnet` baseline follows the official THUML classification pathway
pinned at upstream commit `4e938a1767106324dd753b2a44832bf870a0252e`. For an embedded
sequence $E\in\mathbb R^{B\times T\times D}$, the top-$k$ nonzero Fourier amplitudes
select periods

$$p_j=\left\lfloor\frac{T}{f_j}\right\rfloor,\qquad
(f_1,\ldots,f_k)=\operatorname{TopK}_{f>0}\;\mathbb E_{b,d}|\operatorname{RFFT}(E)_{bfd}|.$$

For each $p_j$, the sequence is right-padded to a multiple of $p_j$, reshaped into a
two-dimensional temporal grid, passed through two inception convolution blocks with a
GELU between them, and restored to one dimension. If
$a_{ij}=\mathbb E_d|\operatorname{RFFT}(E_i)_{f_jd}|$, period outputs $U_{ij}$ are fused
using sample-specific spectral weights and a residual:

$$U_i=E_i+\sum_{j=1}^{k}\operatorname{softmax}(a_i)_{j}U_{ij}.$$

After two TimesBlocks and layer normalization, the implemented classification path applies
GELU, dropout, zeros masked timesteps, flattens the fixed-length representation, and uses
a linear class projection. `d_model=d_ff=32`, $k=2$, six odd-kernel inception branches,
and two TimesBlocks are explicit local capacity choices subject to the equal-budget search;
they are not attributed to the rejected manuscript. Historical runs using
`TimesNetBaseline` are simplified-placeholder evidence and cannot establish fairness.

### 9.2 Controlled delay task (EXP-3.3)

Before observing results, three delays are fixed at $d\in\{4,64,192\}$ in sequences of
$T=256$. Each sample draws signed bits $a,b\in\{-1,+1\}$, balanced over their four
combinations within every subject, and sets

$$X_{T-1-d,0}=3a,\qquad X_{T-1,0}=3b,\qquad X_{T-1,1}=1,$$

with low-amplitude Gaussian nuisance channels and subject offsets. The target is the XOR
$y=\mathbb 1[a\ne b]$. Channel 1 identifies the query timestep but contains no label.
The compared models are full D-MSTCN and otherwise identical SSB-only, MSB-only, and
LSB-only variants ($D=64$), using grouped five-fold evaluation and three seeds.

Prespecified support requires (a) full-model pooled macro-F1 at least 0.65 at every delay,
and (b) on $d=192$, full or LSB performance at least 0.10 macro-F1 above SSB. Failure of
either criterion falsifies, rather than confirms, the proposed scale-behavior evidence.

### 9.3 Equal-budget model tuning (EXP-2.2/2.3)

Each of six architectures receives exactly eight development-search fits with seed zero.
The model-specific Cartesian spaces vary two capacity choices, two structural/dropout
choices where applicable, and learning rate $\{10^{-4},3\times10^{-4}\}$. D-MSTCN and
TCN vary width and dropout; LSTM varies hidden width and layer count; Transformer and
iTransformer vary embedding width and layer count; TimesNet varies width and layer count.
All use train-only class-balanced cross-entropy. For each model,

$$h_m^*=\arg\max_{h\in\mathcal H_m}\operatorname{MacroF1}_{\mathrm{dev}}(m,h,s=0),$$

with ties resolved by the lower prespecified trial index. The test loader is never created
by the search call. After selection, $h_m^*$ is frozen and evaluated for seeds 0–4; each
fit touches test exactly once after loading its best-development state. Failed trials remain
in the search record and their budget is not reassigned.

### 9.4 Participant-only DAIC-WOZ representation

For the prespecified feature-quality refinement, transcript rows whose normalized speaker
field equals `Participant` define intervals $[s_j,e_j)$. Each interval is clipped to the
audio bounds, extracted in chronological transcript order, and concatenated directly:

$$x^{(P)}=x[s_1:e_1]\,\Vert\,x[s_2:e_2]\,\Vert\cdots\Vert\,x[s_J:e_J].$$

No interviewer samples and no synthetic gaps are inserted. The concatenated waveform is
partitioned into non-overlapping 0.5-second windows; eGeMAPSv02 Functionals produces 88
features per window. At most 2000 windows are retained, shorter sequences are right-padded,
and their true length drives the model mask. Non-finite openSMILE outputs are mapped to zero
and their fraction is recorded. This changes only the input representation; all compared
models must use the same cache and configurations selected without test access.

### 9.5 CSAG and personalization controls

The full dynamic CSAG remains exactly as defined in Equations (3)--(6). The fixed-mean
control replaces its sample-dependent weights by

$$H=\frac{1}{J}\sum_{b=1}^{J}H_b,$$

where $J=3$ is the number of temporal branches (distinct from batch size $B$). The
learned-static control instead has three trainable logits $q_b$ and computes

$$\pi_b=\frac{\exp(q_b)}{\sum_{j=1}^{J}\exp(q_j)},\qquad
H=\sum_{b=1}^{J}\pi_bH_b.$$

These weights have no sample or timestep dependence. The low- and high-temperature
controls use $0.5\sqrt D$ and $2\sqrt D$, respectively, in the CSAG attention denominator.
For the experiment configuration $D=128$, these values are 5.656854249 and 22.627416998.

The subject FiLM condition uses one learned embedding row per training subject. The
`global` control stores one row and indexes it for every sample. The `global_matched`
control allocates the same number of rows as subject FiLM but always indexes row zero;
the unused rows match parameter count without adding subject information. The
`noAdapter` control removes both the embedding table and FiLM parameter generator.
Consequently, the full-versus-`global_matched` comparison is the parameter-count-matched
personalization test; full-versus-`global` is not interpreted as parameter-count-only.

### 9.6 Complete implemented Phase-5 family

`experiments/ablation.py` fixes one common split and evaluates 14 configurations over
three seeds and five folds: full; `noSSB`, `noMSB`, `noLSB`; `1scale_SSB`,
`1scale_MSB`, `1scale_LSB`; fixed-mean `noCSAG`; learned `staticCSAG`; `tempLow` and
`tempHigh`; `noAdapter`; `globalAdapter`; and parameter-matched `matchedGlobal`.
For a one-branch condition, `DMSTCN.forward` directly uses that branch output, so neither
dynamic nor static fusion parameters are instantiated. For each condition $v$, seed-level
fold scores are first averaged within each fold,

$$\bar F_{v,k}=\frac{1}{S}\sum_{s=1}^{S}F_{v,s,k},\qquad S=3, K=5,$$

and the reported point estimate is $K^{-1}\sum_k\bar F_{v,k}$. The reported delta is
$\Delta_v=\bar F_v-\bar F_{\mathrm{full}}$. All 13 full-versus-control Wilcoxon
p-values form one multiplicity family and receive both Holm and Benjamini--Hochberg
adjustments. Implementation completeness does not imply that this expanded 210-fit
experiment has been executed; run status and results belong in `STATUS.md` and
`METRICS.md`.

## 10. Evaluation metrics as computed

For class $c$, with $TP_c$, $FP_c$, and $FN_c$ computed from hard predictions,

$$P_c=\frac{TP_c}{TP_c+FP_c},\qquad
R_c=\frac{TP_c}{TP_c+FN_c},\qquad
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c},$$

where scikit-learn's `zero_division=0` makes an undefined precision, recall, or F1 zero.
In `classification_metrics`, macro-F1 averages the labels in the union observed in
`y_true` and `y_pred`, following scikit-learn's default:
$\operatorname{MacroF1}=|\mathcal L|^{-1}\sum_{c\in\mathcal L}F1_c$. This is the
primary metric. The specialized DAIC bootstrap helper instead loops over all configured
classes $c=0,\ldots,C-1$, including a class absent from a resample with contribution zero.
Accuracy is
$N^{-1}\sum_i\mathbb 1[\hat y_i=y_i]$, and balanced accuracy is the unweighted mean
of class recalls. The main `classification_metrics` function also stores per-class
precision/recall/F1 and the confusion matrix.

For probabilities $p_{ic}$ and one-hot targets $q_{ic}$, the multiclass Brier score is

$$\operatorname{Brier}=\frac1N\sum_{i=1}^{N}\sum_{c=1}^{C}(p_{ic}-q_{ic})^2.$$

The 15-bin expected calibration error uses confidence
$\hat p_i=\max_c p_{ic}$, correctness $a_i=\mathbb 1[\arg\max_c p_{ic}=y_i]$, and
equal-width bins $I_j=(j/15,(j+1)/15]$:

$$\operatorname{ECE}=\sum_{j:n_j>0}\frac{n_j}{N}
 \left|\frac1{n_j}\sum_{i\in I_j}a_i-
             \frac1{n_j}\sum_{i\in I_j}\hat p_i\right|.$$

Binary ROC-AUC and average precision use the class-1 probability. Multiclass ROC-AUC
uses macro one-vs-rest and multiclass average precision uses a macro average over one-hot
classes. If scikit-learn cannot define either quantity for the observed labels, the stored
value is `null`; it is not imputed. These probability metrics are descriptive unless an
experiment explicitly prespecifies them as an endpoint.

## 11. Uncertainty and paired inference as implemented

### 11.1 Grouped-CV summaries

For StudentLife headline, preprocessing, and ablation experiments, the code averages
macro-F1 across seeds within each of the five fixed folds and then treats those five fold
values as the resampling and pairing units. `bootstrap_ci` draws five fold scores with
replacement 10,000 times and reports the 2.5th and 97.5th percentiles of their means.
This is a fold-level percentile interval, not a participant-level bootstrap and not a
seed-level confidence interval.

Given paired fold vectors $x,y$, the implementation defines $d_k=x_k-y_k$. Its function
named `hodges_lehmann_paired` returns $\operatorname{median}_k(d_k)$; this is explicitly
the median paired difference used by this code, not the Walsh-average form sometimes
called the one-sample Hodges--Lehmann estimator. After discarding exact zero differences,
the matched-pairs rank-biserial statistic is

$$r_{rb}=\frac{R_+-R_-}{n(n+1)/2},$$

where $R_+$ and $R_-$ sum average ranks of $|d_k|$ for positive and negative differences.
The Wilcoxon wrapper calls SciPy with `zero_method="wilcox"`; it additionally reports
$p_{\min}=2/2^{n_{\ne0}}$ for a two-sided test and whether $p_{\min}\le0.05$. Thus five
nonzero folds have minimum attainable two-sided exact $p=0.0625$. SciPy may fall back to
its documented approximation in edge cases such as zeros; the returned mode is not
overstated as exact. Paired Cohen $d_z=\bar d/s_d$ exists as a supplementary utility but
is not written by the current headline or ablation runners.

For a family of $M$ raw p-values, Holm adjustment sorts ascending and applies the
monotone running maximum of $(M-i+1)p_{(i)}$; Benjamini--Hochberg applies the reverse
monotone minimum of $Mp_{(i)}/i$. Current Phase 5 applies both to the 13 prespecified
full-versus-control tests. Merely having these utilities does not mean every historical
artifact received a multiplicity correction.

### 11.2 Official-split DAIC participant bootstrap

For each model and each bootstrap replicate, the official-split runner samples $N$ test
sessions with replacement using one shared index vector for every model. It computes
macro-F1 separately for each trained seed on that resample and averages those seed
macro-F1 values. If $F^{*(r)}_m$ denotes replicate $r$ for model $m$, its point estimate is
the corresponding mean across seeds on the unresampled test set, and its interval is the
2.5th--97.5th percentile range of $F^{*(r)}_m$. The paired D-MSTCN contrast is

$$D_b^{*(r)}=F_{\mathrm{D\text{-}MSTCN}}^{*(r)}-F_b^{*(r)},$$

using the same resampled sessions. The artifact stores the observed point difference,
the percentile interval of $D_b^*$, and $R^{-1}\sum_r\mathbb 1[D_b^{*(r)}>0]$ for
$R=10{,}000$. This last number is an empirical bootstrap proportion, not a Bayesian
posterior probability. Seed-paired statistics are stored separately and describe
optimization variation, not independent participant evidence.

`cluster_bootstrap_ci` is a separate general utility: it samples cluster labels with
replacement and then samples observations within each chosen cluster with replacement.
The official DAIC headline uses the specialized fixed-test-session routine above, not
that general utility.

## 12. Split, selection, and reproducibility invariants

The canonical sample mask is $m_{it}=\mathbb 1[t<L_i]$ with $1\le L_i\le T$.
StudentLife grouped CV uses five `StratifiedGroupKFold` folds with shuffle seed zero;
all windows of a participant stay together. The manifest hash is the first 16 hexadecimal
characters of SHA-256 over the sorted JSON participant-fold mapping. Split audits assert
both disjoint sample indices and an empty train/validation participant intersection.

`set_seed(s,"scientific")` seeds Python, NumPy, CPU Torch, and every CUDA device; selects
deterministic cuDNN behavior; disables cuDNN benchmarking; and requests deterministic
Torch algorithms with `warn_only=True`. Consequently, an unsupported nondeterministic
operation can warn rather than abort, so reproducibility is asserted only for the tested
paths and environment, not universally across devices or library versions.

Headline StudentLife uses three seeds, five grouped folds, plain cross-entropy, batch size
64, learning rate $3\times10^{-4}$, minimum learning rate $10^{-6}$, weight decay
$10^{-4}$, at most 100 epochs, and patience 15. The fixed DAIC headline uses five seeds,
the official train/dev/test assignment, class-balanced cross-entropy, batch size 8, at
most 40 epochs, and patience 8. Fair tuning instead uses batch size 32 and five frozen
confirmation seeds after the eight-trial search in §9.3. These are distinct protocols;
their outputs must not be pooled as if they were one experiment.

## 13. Artifact identity, failure preservation, and result admission

For each completed fold/seed fit, `write_completed_fit` derives a run identifier from the
experiment, condition, dataset, fold, repeat, seed, and a 16-hex SHA-256 config hash. The
run directory records resolved configuration, environment, metrics, curve, deidentified
ordered labels/probabilities when available, and run identity. The data-version field is
a 16-hex SHA-256 digest over contiguous bytes of $X$, $y$, and participant IDs; the digest
is stored, not those IDs. An existing identical completed directory is reused rather than
overwritten. A failed fit receives status `model_failed`, its exception class and message
are preserved, and fair-tuning search budget is not reassigned. These behaviors provide
resume safety for the implemented writers, not transactional guarantees for arbitrary
process or filesystem failure.

Before a final corrected StudentLife headline summary is admitted, the mechanical audit
requires the experiment, dataset, protocol, split hash, ordered seed list, exact six-model
set, five finite in-range seed-averaged fold values per model, recomputable fold means,
finite in-range pooled means, ordered finite confidence bounds, and the complete five-way
D-MSTCN comparison family. It validates an embedded data hash when present and otherwise
requires the caller-supplied independently audited expected hash; it emits SHA-256 of the
source JSON. Because the summary contains fold values already averaged over seeds, this
audit cannot by itself prove that all 90 underlying fits ran. Registry rows and per-fit
artifacts are the separate evidence for fit completeness.

## 14. Evidence boundary

The equations above describe code behavior, not empirical success. Implemented and tested
single-device properties include tensor shapes, receptive fields, strict causal forward
dependence, mask-aware pooling, parameter accounting, deterministic replay on the tested
environment, leakage-safe grouped splitting, and artifact validation. The repository does
**not** implement a distributed optimizer, branch workers, real AllReduce, FedAvg training,
or multi-GPU throughput measurement. Therefore it supplies no evidence for branch-parallel
speedup, communication reduction, scalability, or a causal accuracy benefit from TCP.

Historical StudentLife artifacts built from hash `62de62987570bc40` remain invalid because
of leading-prefix backward fill. Historical simplified-TimesNet comparisons remain
non-confirmatory for baseline fairness. Negative and null findings are evidence and must
remain in `METRICS.md`; implementation completion, provisional seeds, and synthetic-task
success must never be rewritten as a real-data headline win.
