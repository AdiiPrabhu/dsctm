"""EXP-2.2/2.3 equal-trial, model-specific DAIC-WOZ tuning.

Every architecture receives eight prespecified development trials. The test set is not
loaded by ``train_model`` during search. One configuration per model is selected by
development macro-F1, then frozen and evaluated over five seeds on test exactly once per
seed. Search failures are preserved and do not transfer budget to another model.
"""
from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path

import numpy as np

from ..models import DMSTCN, DMSTCNConfig
from ..models.baselines import (ITransformerBaseline, LSTMBaseline, TCNBaseline,
                                TransformerBaseline)
from ..models.timesnet import OfficialTimesNetBaseline
from ..registry import write_completed_fit, write_failed_fit
from ..train.trainer import train_model, train_select_evaluate

MODELS = ("dmstcn", "lstm", "temporal-cnn", "transformer", "timesnet", "itransformer")
BASE_TRAIN = {"batch_size": 32, "lr_min": 1e-6, "max_epochs": 40,
              "early_stop_patience": 8, "class_weight": "balanced"}


def _grid(**values):
    keys = list(values)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(values[k] for k in keys))]


SEARCH = {
    "dmstcn": _grid(D=[64, 128], dropout=[0.0, 0.2], lr=[1e-4, 3e-4]),
    "lstm": _grid(hidden=[64, 128], layers=[1, 2], lr=[1e-4, 3e-4]),
    "temporal-cnn": _grid(D=[64, 128], dropout=[0.0, 0.2], lr=[1e-4, 3e-4]),
    "transformer": _grid(d_model=[64, 128], layers=[1, 2], lr=[1e-4, 3e-4]),
    "timesnet": _grid(d_model=[16, 32], layers=[1, 2], lr=[1e-4, 3e-4]),
    "itransformer": _grid(d_model=[64, 128], layers=[1, 2], lr=[1e-4, 3e-4]),
}


def _build(name, ds, params):
    def builder(n_subjects):
        p = dict(params)
        p.pop("lr", None)
        if name == "dmstcn":
            return DMSTCN(DMSTCNConfig(input_dim=ds.F, n_classes=ds.n_classes,
                                       n_subjects=n_subjects, **p))
        if name == "lstm":
            return LSTMBaseline(ds.F, ds.n_classes, **p)
        if name == "temporal-cnn":
            return TCNBaseline(ds.F, ds.n_classes, **p)
        if name == "transformer":
            return TransformerBaseline(ds.F, ds.n_classes, **p)
        if name == "timesnet":
            return OfficialTimesNetBaseline(ds.F, ds.n_classes, ds.T, d_ff=p["d_model"], **p)
        if name == "itransformer":
            return ITransformerBaseline(ds.F, ds.n_classes, **p)
        raise KeyError(name)
    return builder


def _indices(ds, manifest):
    return {split: np.asarray([i for i, sid in enumerate(ds.subject_id)
                               if manifest["split_of_subject"].get(str(sid)) == split])
            for split in ("train", "dev", "test")}


def run_fair_tuning(ds, manifest, seeds=(0, 1, 2, 3, 4),
                    out_root="artifacts/resubmission/phase2", log=print):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = _indices(ds, manifest)
    # Hash the protected participant-to-split mapping without writing identifiers.
    split_hash = hashlib.sha256(json.dumps(
        manifest["split_of_subject"], sort_keys=True).encode()).hexdigest()[:16]
    search_results, selected, confirmation = {}, {}, {}
    out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        trials = []
        for trial, params in enumerate(SEARCH[name]):
            cfg = {**BASE_TRAIN, "weight_decay": 1e-4, "lr": params["lr"]}
            try:
                fit = train_model(_build(name, ds, params), ds, idx["train"], idx["dev"],
                                  cfg, device, seed=0, personalize=(name == "dmstcn"))
                score = float(fit["val_metrics"]["macro_f1"])
                artifact = write_completed_fit(
                    experiment_id="EXP-2.2-2.3", condition=f"{name}_search_t{trial}",
                    dataset=ds, protocol="official_train_dev_search", fold=None, seed=0,
                    split_hash=split_hash, config={**cfg, **params}, result=fit)
                row = {"trial": trial, "params": params, "dev_macro_f1": score,
                       "status": "completed", "artifact": str(artifact)}
            except Exception as exc:
                artifact = write_failed_fit(
                    experiment_id="EXP-2.2-2.3", condition=f"{name}_search_t{trial}",
                    dataset=ds, protocol="official_train_dev_search", fold=None, seed=0,
                    split_hash=split_hash, config={**cfg, **params}, error=exc)
                row = {"trial": trial, "params": params, "dev_macro_f1": None,
                       "status": "model_failed", "error": f"{type(exc).__name__}: {exc}",
                       "artifact": str(artifact)}
            trials.append(row)
            log(f"[tune] {name} t{trial}: {row['status']} dev={row['dev_macro_f1']}")
            (out / "fair_tuning_partial.json").write_text(json.dumps(
                {"search": {**search_results, name: trials}, "selected": selected}, indent=2))
        valid = [r for r in trials if r["status"] == "completed"]
        if not valid:
            search_results[name] = trials
            selected[name] = None
            continue
        best = max(valid, key=lambda r: (r["dev_macro_f1"], -r["trial"]))
        search_results[name] = trials
        selected[name] = best

        cfg = {**BASE_TRAIN, "weight_decay": 1e-4, "lr": best["params"]["lr"]}
        seed_rows = []
        for seed in seeds:
            fit = train_select_evaluate(
                _build(name, ds, best["params"]), ds, idx["train"], idx["dev"], idx["test"],
                cfg, device, seed=seed, personalize=(name == "dmstcn"))
            artifact = write_completed_fit(
                experiment_id="EXP-2.2-2.3", condition=f"{name}_confirm",
                dataset=ds, protocol="official_dev_selected_test_once", fold=None, seed=seed,
                split_hash=split_hash, config={**cfg, **best["params"]}, result=fit)
            seed_rows.append({"seed": seed, "dev_metrics": fit["dev_metrics"],
                              "test_metrics": fit["test_metrics"], "artifact": str(artifact)})
            log(f"[confirm] {name} seed{seed}: "
                f"test={fit['test_metrics']['macro_f1']:.4f}")
        confirmation[name] = seed_rows
    final = {"experiment": "EXP-2.2/2.3", "dataset": ds.dataset,
             "search_budget_trials_per_model": 8, "selection_metric": "dev_macro_f1",
             "test_access_during_search": False, "seeds_confirmation": list(seeds),
             "search": search_results, "selected": selected, "confirmation": confirmation}
    (out / "fair_tuning.json").write_text(json.dumps(final, indent=2, default=str))
    return final
