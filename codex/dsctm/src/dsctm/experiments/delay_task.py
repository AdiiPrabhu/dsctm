"""EXP-3.3 controlled short/medium/long delay experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.splits import subject_grouped_kfold
from ..data.synthetic import make_delay_dependency
from ..models import DMSTCN, DMSTCNConfig
from ..registry import write_completed_fit
from ..train.trainer import headline_cv

DELAYS = {"short": 4, "medium": 64, "long": 192}
VARIANTS = {"full": ("ssb", "msb", "lsb"), "ssb": ("ssb",),
            "msb": ("msb",), "lsb": ("lsb",)}
CFG = {"batch_size": 64, "lr": 3e-4, "lr_min": 1e-6, "weight_decay": 1e-4,
       "max_epochs": 60, "early_stop_patience": 10}
SUCCESS_CRITERIA = {
    "task_learned": "full-model pooled macro-F1 >= 0.65 for each delay",
    "scale_specific": "on long delay, full or LSB exceeds SSB by >= 0.10 macro-F1",
    "falsification": "criteria failure rejects the claimed scale-behavior evidence",
}


def run_delay_task(seeds=(0, 1, 2), out_root="artifacts/resubmission/phase3", log=print):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for scale, delay in DELAYS.items():
        ds = make_delay_dependency(delay=delay)
        folds, manifest = subject_grouped_kfold(ds.subject_id, ds.y, 5, seed=0)
        for variant, branches in VARIANTS.items():
            def build(n, branches=branches):
                return DMSTCN(DMSTCNConfig(input_dim=ds.F, n_classes=2, n_subjects=n,
                                           D=64, head_hidden=64,
                                           enabled_branches=branches))
            pooled, fold_values = [], []
            condition = f"{scale}_{variant}"
            config = {**CFG, "delay": delay, "branches": list(branches), "D": 64}
            for seed in seeds:
                run = headline_cv(
                    build, ds, folds, CFG, device, seed=seed, personalize=True,
                    on_fold_complete=lambda fold, fit, seed=seed: write_completed_fit(
                        experiment_id="EXP-3.3", condition=condition, dataset=ds,
                        protocol="subject_grouped_5fold_delay_xor", fold=fold, seed=seed,
                        split_hash=manifest["split_hash"], config=config, result=fit),
                )
                pooled.append(run["pooled"]["macro_f1"])
                fold_values.append(run["per_fold_macro_f1"])
                log(f"[delay] {condition} seed{seed}: pooled={pooled[-1]:.4f}")
            results[condition] = {
                "delay": delay, "branches": list(branches),
                "pooled_macro_f1_mean": float(np.mean(pooled)),
                "pooled_macro_f1_std": float(np.std(pooled)),
                "per_fold_mean_over_seeds": np.asarray(fold_values).mean(0).tolist(),
            }
            out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
            (out / "delay_task_partial.json").write_text(json.dumps(
                {"completed": list(results), "criteria": SUCCESS_CRITERIA,
                 "results": results}, indent=2))
    learned = all(results[f"{s}_full"]["pooled_macro_f1_mean"] >= 0.65 for s in DELAYS)
    long_best = max(results["long_full"]["pooled_macro_f1_mean"],
                    results["long_lsb"]["pooled_macro_f1_mean"])
    scale_specific = long_best - results["long_ssb"]["pooled_macro_f1_mean"] >= 0.10
    final = {"experiment": "EXP-3.3", "criteria_prespecified": SUCCESS_CRITERIA,
             "criteria_results": {"task_learned": learned,
                                  "scale_specific": scale_specific,
                                  "claim_supported": learned and scale_specific},
             "results": results}
    (Path(out_root) / "delay_task.json").write_text(json.dumps(final, indent=2))
    return final
