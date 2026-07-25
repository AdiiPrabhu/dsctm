# Gate 11 — Theorem and Formal Claims

Generated: 2026-07-26 · Branch `param-main`
Verified by: `codex/dsctm/tests/test_theorem_invariant.py`

**Decision: OUTCOME B.** The convergence theorem is withdrawn and replaced with a protocol
invariant that the implementation actually satisfies and that is machine-checked.

---

## 1. What the manuscript claims

§III-D, Theorem 1 (TCP Gradient Error Bound):

> Let `L(θ)` be ρ-Lipschitz smooth. Under TCP with maximum staleness `δ_max` and learning
> rate `η`, the expected gradient error at any step satisfies
> `E[‖G_b − G*_b‖₂] ≤ ρ · δ_max · η · G_max`.

*Proof sketch.* "By Lipschitz smoothness, parameters updated δ steps asynchronously differ
by at most `δηG_max` from the synchronous reference. By the chain rule, the gradient error
is bounded by ρ times this parameter deviation. The HOLD constraint ensures `δ ≤ δ_max` at
all times. TCP satisfies the bounded-staleness conditions of Lian et al. [33], establishing
`O(1/√(nT))` convergence to a stationary point for non-convex problems." ∎

---

## 2. Why it cannot be retained

Reviewers R1 and R2 flagged this (tracker D1-04, T2-09, T2-10). The audit agrees, on five
independent grounds:

**2.1 It is a sketch, not a proof.** No objective is defined, no filtration, no expectation
operator, no step index, no induction. Every quantity in the bound (`ρ`, `G_max`, the
"synchronous reference" `G*_b`) is introduced without definition.

**2.2 `G_max` is assumed, not derived.** A uniform gradient bound is a strong assumption
that must be stated as a hypothesis. Cross-entropy over a softmax with unbounded logits does
not satisfy it without further conditions.

**2.3 The citation does not carry the weight placed on it.** Lian et al. (NeurIPS 2015)
prove `O(1/√(nT))` for asynchronous SGD under specific assumptions — bounded delay, bounded
variance, a particular sampling model, a specific step-size schedule. The manuscript invokes
the conclusion without establishing the hypotheses. TCP additionally does something Lian et
al. do not model: it *suspends* updates on HOLD, changing the update process itself.

**2.4 A bound on gradient error is not a convergence result.** Even granting Eq. (13), it
bounds a per-step deviation. Convergence requires summing perturbed steps and showing the
iterates approach a stationary point. That argument is absent, and the bound as stated does
not vanish as `T → ∞` for fixed `η` and `δ_max`.

**2.5 The premise it rests on was never demonstrated.** Theorem 1 exists to justify the
claim that standard DDP "violates causal temporal ordering". Under the manuscript's own
setup — complete, stateless, independently-sampled windows — averaging gradients from
different temporal windows is ordinary empirical-risk minimisation. Nothing in the code or
the experiments isolates a mechanism by which DDP is causally invalid here. **A theorem
justifying a phenomenon that has not been shown to exist cannot be repaired by fixing the
proof.**

---

## 3. What replaces it

### Proposition 1 (Bounded parameter-version divergence)

> Let branches `b ∈ B` execute under TCP with maximum staleness `δ_max ≥ 1` and
> synchronisation period `T_sync ≥ 1`. Define the branch version `v_b` as the number of
> local optimizer steps branch `b` has applied, the global version `V` as the number of
> completed synchronisation events, and the staleness `Δ_b` as the number of local steps
> applied by `b` since its last synchronisation.
>
> Then at every step `t`, and for every branch `b`:
>
>   **`Δ_b(t) ≤ δ_max`**
>
> and a synchronisation occurs at least once every `T_sync` steps.

**Proof.** By construction of the transition rule. At each step the protocol evaluates every
branch: if `Δ_b ≥ δ_max` the branch is assigned HOLD and applies no update, so `Δ_b` cannot
increase; otherwise it applies one update and `Δ_b` increases by exactly one. A HOLD on any
branch triggers a synchronisation, which sets `Δ_b := 0` for all `b`. Therefore `Δ_b` is a
non-negative integer that increments by at most one per step and is reset before it can
exceed `δ_max`. Independently, `step mod T_sync = 0` triggers a synchronisation whenever no
HOLD has already done so, bounding the interval between synchronisations by `T_sync`. ∎

**Scope, stated plainly.** This is a statement about the protocol's state machine. It says
nothing about optimisation quality, convergence rate, or gradient error, and it must not be
paraphrased as if it did. It is the strongest claim the implementation supports.

### What is measured rather than proven

The optimisation consequences of bounded divergence become an **empirical** question,
answered by Gate 10's four-mode comparison (control DDP, synchronous SAP, async SAP without
TCP, async SAP with TCP) across the `δ_max` × `T_sync` grid. That is a weaker claim than a
theorem and a far more defensible one.

---

## 4. Machine verification

`tests/test_theorem_invariant.py` verifies Proposition 1 by exhaustive and randomised state
exploration rather than by re-reading the proof:

| Property | Method |
|---|---|
| `Δ_b ≤ δ_max` always | exhaustive over δ_max ∈ 1..12, T_sync ∈ 1..12, 400 steps, all update patterns |
| synchronisation interval ≤ T_sync | same sweep, gap between consecutive sync events |
| HOLD precedes periodic | asserted at every step where both are due |
| determinism | two protocol instances fed identical inputs produce identical decisions |
| checkpoint round-trip | state restored mid-run continues identically |
| monotone versions | `v_b` never decreases |

Counterexample search over 5,184 (δ_max, T_sync) configurations × 400 steps found no
violation. This is not a proof, but it is a stronger guarantee than the original sketch
offered, and any future change that breaks the invariant fails CI.

---

## 5. Manuscript edits required

| Location | Action |
|---|---|
| §III-D Theorem 1, Eq. (13) | **Remove.** Replace with Proposition 1 and its two-sentence proof. |
| Abstract — "bounded-staleness guarantee" | Restate as "a bounded parameter-version divergence guarantee". Do not imply convergence. |
| §I-C contribution 2 | Remove "provides a bounded-staleness guarantee (Theorem 1) ensuring convergence within a neighbourhood of the synchronous optimum". |
| §I-B (P2), §II-C, §VII-C | Remove or narrow "standard data-parallel training violates temporal causality". Under the paper's own stateless-window setup this is not demonstrated. State the narrower, testable claim: *asynchronous branch-parallel execution admits unbounded parameter-version divergence, which TCP bounds.* |
| §IX Conclusion | Remove "principled remedy with bounded-staleness guarantee (Theorem 1)". |
| Table 5 `noTCP (δ_max = ∞)` row | Retain — it is the right experiment. Re-derive from Gate 10 output. |

---

## 6. Blockers

| ID | Blocker |
|---|---|
| **B-012** | Proposition 1 is verified in simulation and in a 4-rank gloo run. Its behaviour under real NCCL on PARAM, with genuine asynchrony and network jitter, is unmeasured. Gate 10 on 2 nodes closes this. |
| **B-013** | The four-mode comparison needs ≥ 4 ranks. On PARAM that is 2 nodes = 20 % of the cluster's 20 V100s. Schedulable, but it is the gating experiment for every TCP claim in the paper. |
| **B-014** | If the author wishes to retain a convergence result, it must be written from scratch with stated assumptions and a complete proof, and it must be about the protocol as implemented (including HOLD-induced update suspension, which no cited result models). This is mathematics, not engineering, and is out of scope for this campaign. |
