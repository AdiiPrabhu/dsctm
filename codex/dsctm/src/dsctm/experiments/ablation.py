"""Phase 5 — architecture / personalization ablations (master-prompt §9).

Reproduces the manuscript's Table 5 component ablation under the CORRECTED protocol
(subject-grouped 5-fold, leakage-safe), on this GPU:

  branch controls: full, each of three two-branch removals, and each single branch;
  fusion controls: dynamic CSAG, fixed mean, learned-static weights, and half/double
  attention temperature;
  personalization controls: subject FiLM, no FiLM, one-row global FiLM, and a
  parameter-count-matched global FiLM that always indexes row zero.

Each of the 14 variants is evaluated with fold-level uncertainty; delta F1 versus full
and multiplicity-corrected paired comparisons are reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.splits import subject_grouped_kfold
from ..eval import statistics as st
from ..models import DMSTCN, DMSTCNConfig
from ..registry import write_completed_fit
from ..train.trainer import headline_cv

ABLATIONS = {
    "full":          dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True, film_mode="subject", csag_mode="attention"),
    "noSSB":         dict(enabled_branches=("msb", "lsb"), use_film=True, film_mode="subject", csag_mode="attention"),
    "noMSB":         dict(enabled_branches=("ssb", "lsb"), use_film=True, film_mode="subject", csag_mode="attention"),
    "noLSB":         dict(enabled_branches=("ssb", "msb"), use_film=True, film_mode="subject", csag_mode="attention"),
    "1scale_SSB":    dict(enabled_branches=("ssb",), use_film=True, film_mode="subject", csag_mode="attention"),
    "1scale_MSB":    dict(enabled_branches=("msb",), use_film=True, film_mode="subject", csag_mode="attention"),
    "1scale_LSB":    dict(enabled_branches=("lsb",), use_film=True, film_mode="subject", csag_mode="attention"),
    "noCSAG":        dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True, film_mode="subject", csag_mode="mean"),
    "staticCSAG":    dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True, film_mode="subject", csag_mode="static"),
    "tempLow":       dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True, film_mode="subject", csag_mode="attention", temperature=5.656854249),
    "tempHigh":      dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True, film_mode="subject", csag_mode="attention", temperature=22.627416998),
    "noAdapter":     dict(enabled_branches=("ssb", "msb", "lsb"), use_film=False, film_mode="subject", csag_mode="attention"),
    "globalAdapter": dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True, film_mode="global", csag_mode="attention"),
    "matchedGlobal": dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True, film_mode="global_matched", csag_mode="attention"),
}
CFG = {"batch_size": 64, "lr": 3e-4, "lr_min": 1e-6, "weight_decay": 1e-4,
       "max_epochs": 100, "early_stop_patience": 15}


def run_studentlife_ablation(ds, seeds=(0, 1, 2), cfg=None, n_splits=5,
                             out_root="artifacts/resubmission/phase5", log=print):
    import torch
    cfg = cfg or CFG
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    folds, manifest = subject_grouped_kfold(ds.subject_id, ds.y, n_splits, seed=0)
    perfold, results = {}, {}
    for name, over in ABLATIONS.items():
        def build(n, over=over):
            return DMSTCN(DMSTCNConfig(input_dim=ds.F, n_classes=ds.n_classes, n_subjects=n, **over))
        personalize = over["use_film"]
        seed_perfold = []
        for s in seeds:
            run_cfg = {**cfg, "model": "dmstcn", "variant": name, **over}
            r = headline_cv(
                build, ds, folds, cfg, dev, seed=s, personalize=personalize,
                on_fold_complete=lambda fold, fit, s=s, run_cfg=run_cfg: write_completed_fit(
                    experiment_id="EXP-5.1-5.2-5.5", condition=name, dataset=ds,
                    protocol="subject_grouped_5fold", fold=fold, seed=s,
                    split_hash=manifest["split_hash"], config=run_cfg, result=fit,
                ),
            )
            seed_perfold.append(r["per_fold_macro_f1"])
        mean_perfold = np.array(seed_perfold).mean(0)
        perfold[name] = mean_perfold
        point, lo, hi = st.bootstrap_ci(mean_perfold)
        results[name] = {"macro_f1": float(mean_perfold.mean()),
                         "fold_ci95": [float(lo), float(hi)],
                         "per_fold": mean_perfold.tolist()}
        log(f"[ablate] {name:11s} macroF1={mean_perfold.mean():.4f}")
        Path(out_root).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / "studentlife_ablation_partial.json").write_text(
            json.dumps({"completed_variants": list(results), "split_hash": manifest["split_hash"],
                        "seeds": list(seeds), "results": results}, indent=2, default=str)
        )
    full = results["full"]["macro_f1"]
    for name in results:
        results[name]["delta_vs_full"] = results[name]["macro_f1"] - full
    comparisons = {}
    raw_p = []
    names = []
    for name in results:
        if name == "full":
            continue
        test = st.wilcoxon_paired(perfold["full"], perfold[name])
        comparisons[name] = {
            "hodges_lehmann_full_minus_variant": st.hodges_lehmann_paired(
                perfold["full"], perfold[name]
            ),
            "rank_biserial_full_minus_variant": st.paired_rank_biserial(
                perfold["full"], perfold[name]
            ),
            "wilcoxon": test,
        }
        raw_p.append(test["p_value"])
        names.append(name)
    holm = st.holm_bonferroni(raw_p)
    bh = st.benjamini_hochberg(raw_p)
    for i, name in enumerate(names):
        comparisons[name]["wilcoxon"]["holm_adjusted_p"] = float(holm[i])
        comparisons[name]["wilcoxon"]["bh_adjusted_p"] = float(bh[i])
    out = {"experiment": "EXP-5.1/5.2/5.5", "dataset": "studentlife",
           "protocol": "subject_grouped_5fold", "split_hash": manifest["split_hash"],
           "seeds": list(seeds), "results": results,
           "full_vs_variants": comparisons,
           "multiplicity_family": "all prespecified full-versus-component comparisons"}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "studentlife_ablation.json").write_text(json.dumps(out, indent=2, default=str))
    return out
