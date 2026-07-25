"""EXP-1.3 StudentLife preprocessing robustness on fixed grouped folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.splits import subject_grouped_kfold
from ..data.studentlife import build_studentlife
from ..eval import statistics as st
from ..models import DMSTCN, DMSTCNConfig
from ..registry import write_completed_fit
from ..train.trainer import headline_cv

CONDITIONS = ("causal_ffill", "train_mean", "zero", "mask_aware_zero")
CFG = {"batch_size": 64, "lr": 3e-4, "lr_min": 1e-6, "weight_decay": 1e-4,
       "max_epochs": 100, "early_stop_patience": 15}


def run_preprocessing_robustness(seeds=(0, 1, 2), cfg=None,
                                 out_root="artifacts/resubmission/phase1", log=print):
    import torch
    cfg = cfg or CFG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results, per_fold = {}, {}
    split_hash = None
    for condition in CONDITIONS:
        cache = f"artifacts/cache/studentlife_v2_{condition}.npz"
        ds = build_studentlife(cache=cache, imputation=condition)
        folds, manifest = subject_grouped_kfold(ds.subject_id, ds.y, 5, seed=0)
        split_hash = split_hash or manifest["split_hash"]
        if manifest["split_hash"] != split_hash:
            raise AssertionError("preprocessing conditions do not share the same split")

        def build(n):
            return DMSTCN(DMSTCNConfig(input_dim=ds.F, n_classes=ds.n_classes, n_subjects=n))

        seed_fold = []
        for seed in seeds:
            run_cfg = {**cfg, "model": "dmstcn", "imputation": condition,
                       "input_dim": ds.F}
            run = headline_cv(
                build, ds, folds, cfg, device, seed=seed, personalize=True,
                on_fold_complete=lambda fold, fit, seed=seed, run_cfg=run_cfg: write_completed_fit(
                    experiment_id="EXP-1.3", condition=condition, dataset=ds,
                    protocol="subject_grouped_5fold", fold=fold, seed=seed,
                    split_hash=manifest["split_hash"], config=run_cfg, result=fit,
                ),
            )
            seed_fold.append(run["per_fold_macro_f1"])
            log(f"[preprocess] {condition} seed{seed}: "
                f"fold_mean={np.mean(run['per_fold_macro_f1']):.4f}")
        values = np.asarray(seed_fold).mean(0)
        per_fold[condition] = values
        point, lo, hi = st.bootstrap_ci(values)
        results[condition] = {"F": ds.F, "data_version": ds.version,
                              "data_hash": ds.data_version_hash(),
                              "fold_macro_f1": point, "fold_ci95": [lo, hi],
                              "per_fold": values.tolist()}
        Path(out_root).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / "studentlife_preprocessing_partial.json").write_text(
            json.dumps({"completed": list(results), "split_hash": split_hash,
                        "results": results}, indent=2)
        )
    reference = per_fold["causal_ffill"]
    comparisons = {}
    for condition in CONDITIONS[1:]:
        comparisons[condition] = {
            "hl_causal_ffill_minus_condition": st.hodges_lehmann_paired(
                reference, per_fold[condition]),
            "rank_biserial": st.paired_rank_biserial(reference, per_fold[condition]),
            "wilcoxon": st.wilcoxon_paired(reference, per_fold[condition]),
        }
    out = {"experiment": "EXP-1.3", "dataset": "studentlife",
           "protocol": "fixed_subject_grouped_5fold", "split_hash": split_hash,
           "seeds": list(seeds), "results": results,
           "causal_ffill_vs_conditions": comparisons}
    (Path(out_root) / "studentlife_preprocessing.json").write_text(json.dumps(out, indent=2))
    return out
