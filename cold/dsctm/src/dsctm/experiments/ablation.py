"""Phase 5 — architecture / personalization ablations (master-prompt §9).

Reproduces the manuscript's Table 5 component ablation under the CORRECTED protocol
(subject-grouped 5-fold, leakage-safe), on this GPU:

  full        SSB+MSB+LSB + CSAG + FiLM
  noSSB       drop short branch
  noMSB       drop medium branch
  noLSB       drop long branch
  noCSAG      fixed-average fusion instead of attention (EXP-5.2)
  noAdapter   no FiLM personalization (EXP-5.5)
  1scale_SSB  single short branch only

Each variant is evaluated with fold-level uncertainty; ΔF1 vs full is reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.splits import subject_grouped_kfold
from ..eval import statistics as st
from ..models import DMSTCN, DMSTCNConfig
from ..train.trainer import headline_cv

ABLATIONS = {
    "full":       dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True,  csag_mode="attention"),
    "noSSB":      dict(enabled_branches=("msb", "lsb"),        use_film=True,  csag_mode="attention"),
    "noMSB":      dict(enabled_branches=("ssb", "lsb"),        use_film=True,  csag_mode="attention"),
    "noLSB":      dict(enabled_branches=("ssb", "msb"),        use_film=True,  csag_mode="attention"),
    "noCSAG":     dict(enabled_branches=("ssb", "msb", "lsb"), use_film=True,  csag_mode="mean"),
    "noAdapter":  dict(enabled_branches=("ssb", "msb", "lsb"), use_film=False, csag_mode="attention"),
    "1scale_SSB": dict(enabled_branches=("ssb",),              use_film=True,  csag_mode="attention"),
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
            r = headline_cv(build, ds, folds, cfg, dev, seed=s, personalize=personalize)
            seed_perfold.append(r["per_fold_macro_f1"])
        mean_perfold = np.array(seed_perfold).mean(0)
        perfold[name] = mean_perfold
        point, lo, hi = st.bootstrap_ci(mean_perfold)
        results[name] = {"macro_f1": float(mean_perfold.mean()),
                         "fold_ci95": [float(lo), float(hi)],
                         "per_fold": mean_perfold.tolist()}
        log(f"[ablate] {name:11s} macroF1={mean_perfold.mean():.4f}")
    full = results["full"]["macro_f1"]
    for name in results:
        results[name]["delta_vs_full"] = results[name]["macro_f1"] - full
    out = {"experiment": "EXP-5.1/5.2/5.5", "dataset": "studentlife",
           "protocol": "subject_grouped_5fold", "split_hash": manifest["split_hash"],
           "seeds": list(seeds), "results": results}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "studentlife_ablation.json").write_text(json.dumps(out, indent=2, default=str))
    return out
