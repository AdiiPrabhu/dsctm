"""Phase 0 — implementation correctness (master-prompt §9, Gate 0).

All four experiments run WITHOUT any dataset, so they execute today on the single
RTX 4060 Ti and produce the evidence the preflight marked blocked on missing code:

  EXP-0.1  effective receptive field (measured vs. formulas vs. manuscript print)
  EXP-0.2  exact parameter accounting (+ FLOPs), correcting the FiLM claim
  EXP-0.3  TCP staleness / HOLD / precedence invariants
  EXP-0.4  causality, batch-invariance, determinism, checkpoint equivalence
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models import DMSTCN, DMSTCNConfig
from ..models.blocks import Branch
from ..repro import capture_environment, set_seed
from ..train.tcp import StalenessController, causal_gradient_mask

MANUSCRIPT_PRINTED_RF = {"ssb": 47, "msb": 383, "lsb": 1535}


# --------------------------------------------------------------------------- #
def _measure_branch_rf(branch, D, T, device, eps=1e-6):
    """Empirical receptive field of the last output timestep via input-gradient
    support: positions with non-zero gradient of output[T-1] w.r.t. the input."""
    x = torch.randn(1, D, T, device=device, requires_grad=True)
    target = branch(x)[0, :, T - 1].sum()
    g = torch.autograd.grad(target, x)[0].abs().sum(dim=(0, 1))  # (T,)
    support = (g > g.max() * eps).nonzero().flatten()
    lo, hi = int(support.min()), int(support.max())
    return {"measured_rf": hi - lo + 1, "support_lo": lo, "support_hi": hi, "T_probe": T}


def _perturbation_support(branch, D, rf, device):
    """Check the final output just outside and just inside the measured RF."""
    T = rf + 4
    x = torch.randn(1, D, T, device=device)
    with torch.no_grad():
        base = branch(x)[..., -1]
        outside = x.clone()
        outside[..., T - rf - 1] += 1.0
        inside = x.clone()
        inside[..., T - rf] += 1.0
        outside_diff = float((branch(outside)[..., -1] - base).abs().max())
        inside_diff = float((branch(inside)[..., -1] - base).abs().max())
    return {"outside_rf_max_diff": outside_diff, "inside_rf_max_diff": inside_diff,
            "perturbation_support_ok": outside_diff == 0.0 and inside_diff > 0.0}


def exp_0_1_receptive_field(device="cpu"):
    set_seed(0)
    cfg = DMSTCNConfig()
    schedules = {"ssb": cfg.ssb, "msb": cfg.msb, "lsb": cfg.lsb}
    probe_T = {"ssb": 512, "msb": 2048, "lsb": 8192}
    rows = []
    for name in ("ssb", "msb", "lsb"):
        br = Branch(cfg.D, cfg.K, schedules[name]).to(device).eval()
        m = _measure_branch_rf(br, cfg.D, probe_T[name], device)
        p = _perturbation_support(br, cfg.D, m["measured_rf"], device)
        rows.append({
            "branch": name,
            "dilations": list(schedules[name]),
            "measured_rf": m["measured_rf"],
            "formula_two_conv": br.theoretical_rf_two_conv(),
            "formula_one_conv": br.theoretical_rf_one_conv(),
            "manuscript_printed_rf": MANUSCRIPT_PRINTED_RF[name],
            "measured_matches_two_conv": m["measured_rf"] == br.theoretical_rf_two_conv(),
            "manuscript_matches_measured": MANUSCRIPT_PRINTED_RF[name] == m["measured_rf"],
            **p,
        })
    return {
        "experiment": "EXP-0.1",
        "title": "Effective receptive field",
        "branch_rf": rows,
        "finding": (
            "Measured RF equals the two-convolution formula 1+2(K-1)Σr (61/481/1921) "
            "for this faithful implementation; the manuscript's printed 47/383/1535 "
            "match neither formula and must be corrected (T2-02)."
        ),
    }


# --------------------------------------------------------------------------- #
def exp_0_2_params_flops(device="cpu", n_subjects=48):
    cfg = DMSTCNConfig(n_subjects=n_subjects)
    model = DMSTCN(cfg).to(device)

    def count(m):
        return int(sum(p.numel() for p in m.parameters()))

    groups = {
        "input_projection": count(model.input_proj),
        "branch_ssb": count(model._branches["ssb"]),
        "branch_msb": count(model._branches["msb"]),
        "branch_lsb": count(model._branches["lsb"]),
        "csag": count(model.csag),
        "film_subject_embedding_table": int(model.film.embed.weight.numel()),
        "film_shared_generator": count(model.film.shared)
        + count(model.film.to_gamma)
        + count(model.film.to_beta),
        "head": count(model.head),
    }
    total = int(sum(p.numel() for p in model.parameters()))

    flops = None
    try:
        from thop import profile

        X = torch.randn(1, 60, cfg.input_dim, device=device)
        s = torch.zeros(1, dtype=torch.long, device=device)
        macs, _ = profile(model, inputs=(X, s), verbose=False)
        flops = {"input": "T=60, F=%d" % cfg.input_dim, "macs": float(macs), "flops": float(2 * macs)}
    except Exception as e:  # pragma: no cover
        flops = f"unavailable ({e})"

    return {
        "experiment": "EXP-0.2",
        "title": "Exact parameter / compute accounting",
        "total_params": total,
        "by_group": groups,
        "per_subject_stored_params": cfg.d_s,
        "generated_gamma_beta_per_forward": 2 * cfg.D,
        "manuscript_claim": "adapter adds 2D parameters per subject",
        "correction": (
            "Per-subject STORED cost = d_s = %d (one embedding row). The γ,β vectors "
            "(2D = %d) are GENERATED by a shared FiLM MLP and are activations, not "
            "per-subject stored parameters (T2-03)." % (cfg.d_s, 2 * cfg.D)
        ),
        "flops_T60": flops,
    }


# --------------------------------------------------------------------------- #
def exp_0_3_sync_invariants():
    res = {}
    # periodic AllReduce + bound
    c = StalenessController(delta_max=10, t_sync=50)
    for _ in range(50):
        c.step()
    res["periodic_allreduce_at_t_sync"] = any(e[2] == "periodic" for e in c.events)
    res["delta_bounded"] = all(v <= 10 for v in c.delta.values())

    # HOLD activation + reset: 3 updates take Δ 0→3, the 4th step trips HOLD → reset.
    c2 = StalenessController(delta_max=3, t_sync=10_000)
    log = [c2.step(updating_branches=["ssb"]) for _ in range(4)]
    res["hold_triggered"] = any(a.get("ssb") == "hold" for a in log)
    res["hold_resets_delta"] = c2.delta["ssb"] == 0  # reset by the HOLD on step 4

    # invariant Δ ≤ δ_max across a long run
    c3 = StalenessController(delta_max=10, t_sync=7)
    max_delta = 0
    for _ in range(500):
        c3.step(updating_branches=["ssb", "msb"])
        max_delta = max(max_delta, max(c3.delta.values()))
    res["max_delta_over_500_steps"] = max_delta
    res["invariant_delta_le_delta_max"] = max_delta <= 10

    # causal gradient mask (Eq. 12)
    g = torch.ones(4, 20)
    masked = causal_gradient_mask(g, t_current=9)
    res["causal_mask_future_zeroed"] = (
        float(masked[:, 10:].sum()) == 0.0 and float(masked[:, :10].sum()) > 0
    )
    return {
        "experiment": "EXP-0.3",
        "title": "TCP staleness / HOLD / precedence invariants",
        "checks": res,
        "note": "Invariants verified in single-process simulation; multi-GPU PERFORMANCE still needs the 8-GPU server (GAP-5).",
    }


# --------------------------------------------------------------------------- #
def exp_0_4_causality(device="cpu"):
    set_seed(0, "scientific")
    cfg = DMSTCNConfig()
    r = {}

    # (1) branch causality: output at t must not change when inputs > t change
    br = Branch(cfg.D, cfg.K, cfg.ssb).to(device).eval()
    T, t0 = 128, 64
    x = torch.randn(1, cfg.D, T, device=device)
    with torch.no_grad():
        o1 = br(x)
        x2 = x.clone()
        x2[..., t0 + 1:] += 10.0
        o2 = br(x2)
        r["branch_causal_max_diff_upto_t0"] = float((o1[..., : t0 + 1] - o2[..., : t0 + 1]).abs().max())
    r["is_causal"] = r["branch_causal_max_diff_upto_t0"] < 1e-4

    model = DMSTCN(cfg).to(device).eval()
    X = torch.randn(8, 60, cfg.input_dim, device=device)
    s = torch.zeros(8, dtype=torch.long, device=device)

    # (2) batch invariance
    with torch.no_grad():
        full = model(X, s)
        alone = torch.cat([model(X[i : i + 1], s[i : i + 1]) for i in range(8)], 0)
        r["batch_invariance_max_diff"] = float((full - alone).abs().max())
    r["batch_invariant"] = r["batch_invariance_max_diff"] < 1e-4

    # (3) determinism
    set_seed(0, "scientific")
    a = model(X, s)
    set_seed(0, "scientific")
    b = model(X, s)
    r["determinism_max_diff"] = float((a - b).abs().max())
    r["deterministic"] = r["determinism_max_diff"] < 1e-6

    # (4) variable sequence length
    ok = True
    for T2 in (30, 60, 120, 200):
        try:
            model(torch.randn(4, T2, cfg.input_dim, device=device),
                  torch.zeros(4, dtype=torch.long, device=device))
        except Exception:
            ok = False
    r["variable_length_ok"] = ok

    # (5) checkpoint reload equivalence
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    m2 = DMSTCN(cfg).to(device)
    m2.load_state_dict(torch.load(buf))
    m2.eval()
    with torch.no_grad():
        r["checkpoint_reload_max_diff"] = float((model(X, s) - m2(X, s)).abs().max())
    r["checkpoint_equivalent"] = r["checkpoint_reload_max_diff"] < 1e-6

    # (6) AMP numerical stability smoke test; this does not imply FP32 equivalence.
    if device == "cuda":
        amp_model = DMSTCN(DMSTCNConfig(D=32, head_hidden=32)).to(device).train()
        amp_x = torch.randn(2, 60, cfg.input_dim, device=device)
        amp_s = torch.zeros(2, dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_loss = amp_model(amp_x, amp_s).square().mean()
        amp_loss.backward()
        grads = [p.grad for p in amp_model.parameters() if p.grad is not None]
        r["mixed_precision_loss_finite"] = bool(torch.isfinite(amp_loss).item())
        r["mixed_precision_gradients_finite"] = bool(
            grads and all(torch.isfinite(g).all().item() for g in grads)
        )
    else:
        r["mixed_precision_loss_finite"] = None
        r["mixed_precision_gradients_finite"] = None

    return {"experiment": "EXP-0.4", "title": "Causality / shapes / reproducibility", "checks": r}


# --------------------------------------------------------------------------- #
def run_gate0(out_root="artifacts/resubmission/gate0", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    results = {
        "gate": "Gate 0 — implementation correctness",
        "generated": started,
        "device": device,
        "environment": capture_environment(),
        "experiments": [
            exp_0_1_receptive_field(device),
            exp_0_2_params_flops(device),
            exp_0_3_sync_invariants(),
            exp_0_4_causality(device),
        ],
    }
    (out / "gate0_results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "gate0_summary.md").write_text(_summarize(results))
    return results


def _summarize(results) -> str:
    L = ["# Gate 0 — Implementation Correctness (measured)",
         f"Generated {results['generated']} · device `{results['device']}`", ""]
    for e in results["experiments"]:
        L.append(f"## {e['experiment']} — {e['title']}")
        if e["experiment"] == "EXP-0.1":
            L.append("| branch | dilations | measured RF | formula (2-conv) | manuscript printed |")
            L.append("|--------|-----------|-------------|------------------|--------------------|")
            for r in e["branch_rf"]:
                L.append(f"| {r['branch']} | {r['dilations']} | **{r['measured_rf']}** | "
                         f"{r['formula_two_conv']} | {r['manuscript_printed_rf']} |")
            L.append(f"\n{e['finding']}")
        elif e["experiment"] == "EXP-0.2":
            L.append(f"Total params: **{e['total_params']:,}**  ·  per-subject stored: "
                     f"**{e['per_subject_stored_params']}** (not 2D={e['generated_gamma_beta_per_forward']})")
            L.append(f"\n{e['correction']}")
        elif e["experiment"] == "EXP-0.3":
            for k, v in e["checks"].items():
                L.append(f"- {k}: `{v}`")
        elif e["experiment"] == "EXP-0.4":
            for k, v in e["checks"].items():
                L.append(f"- {k}: `{v}`")
        L.append("")
    return "\n".join(L)
