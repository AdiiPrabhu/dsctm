"""Phase 4 — locked headline evaluation (master-prompt §9).

EXP-4.1  StudentLife subject-grouped 5-fold out-of-fold evaluation.
EXP-4.2  E-DAIC official-split train / dev-select / test evaluation.
EXP-4.3  fold-level bootstrap CIs + paired effect sizes; Wilcoxon reported with the
         n≤5 reachability guard (5 folds cannot reach p<0.05 two-sided — reported, not hidden).

Baseline set here = the 6 distinct architectures. DataParallel-LSTM and FedAvg-LSTM are
training-PROTOCOL variants of the LSTM: DP-LSTM has the same single-model quality as LSTM;
FedAvg-LSTM needs the federated-averaging loop (deferred). Both are handled in the systems
phase, not the single-GPU quality headline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..eval import statistics as st
from ..models import DMSTCN, DMSTCNConfig
from ..models.baselines import build_baseline
from ..train.trainer import headline_cv, train_select_evaluate

CORE_MODELS = ["dmstcn", "lstm", "temporal-cnn", "transformer", "timesnet", "itransformer"]
SL_CFG = {"batch_size": 64, "lr": 3e-4, "lr_min": 1e-6, "weight_decay": 1e-4,
          "max_epochs": 100, "early_stop_patience": 15}
DAIC_CFG = {"batch_size": 8, "lr": 3e-4, "lr_min": 1e-6, "weight_decay": 1e-4,
            "max_epochs": 40, "early_stop_patience": 8,
            # E-DAIC is ~24% positive; plain CE collapsed 5/6 models to the majority
            # baseline. Class-balanced CE (weights from TRAIN labels only, leakage-safe)
            # is the imbalance fix. Selection/metrics are unchanged.
            "class_weight": "balanced"}


def _builder(name, F, C):
    if name == "dmstcn":
        return (lambda n: DMSTCN(DMSTCNConfig(input_dim=F, n_classes=C, n_subjects=n)), True)
    return (lambda n: build_baseline(name, F, C), False)


def _dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_studentlife_headline(ds, seeds=(0, 1, 2), cfg=None, n_splits=5,
                             out_root="artifacts/resubmission/phase4", log=print):
    from ..data.splits import subject_grouped_kfold
    cfg = cfg or SL_CFG
    dev = _dev()
    folds, manifest = subject_grouped_kfold(ds.subject_id, ds.y, n_splits, seed=0)
    perfold_by_model, results = {}, {}
    for name in CORE_MODELS:
        build, personalize = _builder(name, ds.F, ds.n_classes)
        seed_pooled, seed_perfold = [], []
        for s in seeds:
            r = headline_cv(build, ds, folds, cfg, dev, seed=s, personalize=personalize)
            seed_pooled.append(r["pooled"]["macro_f1"])
            seed_perfold.append(r["per_fold_macro_f1"])
            log(f"[SL] {name} seed{s}: pooled_macroF1={r['pooled']['macro_f1']:.4f}")
        mean_perfold = np.array(seed_perfold).mean(0)
        perfold_by_model[name] = mean_perfold
        point, lo, hi = st.bootstrap_ci(mean_perfold)
        results[name] = {
            "pooled_macro_f1_mean": float(np.mean(seed_pooled)),
            "pooled_macro_f1_std": float(np.std(seed_pooled)),
            "fold_macro_f1_mean": float(point),
            "fold_ci95": [float(lo), float(hi)],
            "per_fold_macro_f1_avg_over_seeds": mean_perfold.tolist(),
        }
        # crash-resilience checkpoint: dump partial results after each model completes
        Path(out_root).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / "studentlife_headline_partial.json").write_text(
            json.dumps({"completed_models": list(results.keys()), "results": results},
                       indent=2, default=str))
    dm = perfold_by_model["dmstcn"]
    comparisons = {}
    for name in CORE_MODELS:
        if name == "dmstcn":
            continue
        b = perfold_by_model[name]
        comparisons[name] = {
            "hodges_lehmann_shift": st.hodges_lehmann_paired(dm, b),
            "rank_biserial": st.paired_rank_biserial(dm, b),
            "wilcoxon": st.wilcoxon_paired(dm, b),
        }
    out = {"experiment": "EXP-4.1", "dataset": "studentlife",
           "protocol": "subject_grouped_5fold", "split_hash": manifest["split_hash"],
           "seeds": list(seeds), "config": cfg, "results": results,
           "dmstcn_vs_baselines": comparisons}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "studentlife_headline.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def _macro_f1(y, pred, n_classes):
    """Fast macro-F1 (matches sklearn average='macro', zero_division=0) for bootstrap
    resamples — avoids sklearn's per-call overhead in a 10k-iteration loop."""
    tot = 0.0
    for c in range(n_classes):
        tp = np.count_nonzero((pred == c) & (y == c))
        fp = np.count_nonzero((pred == c) & (y != c))
        fn = np.count_nonzero((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        tot += 0.0 if denom == 0 else (2.0 * tp) / denom
    return tot / n_classes


def _participant_bootstrap(models_seed_probs, y_true, n_classes, n_boot=10000, seed=0):
    """Participant-level (test-session) bootstrap on the FIXED official test set — the
    primary independent unit per the §8 analysis plan (seeds are secondary). Resamples the
    test sessions with replacement; the statistic per model is the seed-averaged macro-F1
    on the resampled sessions. The SAME resample indices are used for every model, so
    D-MSTCN-minus-baseline differences are properly paired. Returns per-model point/CI and
    the raw boot matrices (for paired differences)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    N = len(y_true)
    names = list(models_seed_probs)
    preds = {n: [p.argmax(1) for p in models_seed_probs[n]] for n in names}
    point = {n: float(np.mean([_macro_f1(y_true, pr, n_classes) for pr in preds[n]]))
             for n in names}
    boots = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        ridx = rng.integers(0, N, N)
        yt = y_true[ridx]
        for n in names:
            boots[n][b] = np.mean([_macro_f1(yt, pr[ridx], n_classes) for pr in preds[n]])
    ci = {n: (float(np.quantile(boots[n], 0.025)), float(np.quantile(boots[n], 0.975)))
          for n in names}
    return point, ci, boots


def run_daic_headline(ds, manifest, seeds=(0, 1, 2, 3, 4), cfg=None,
                      out_root="artifacts/resubmission/phase4", log=print,
                      out_name="daic_headline.json", n_boot=10000):
    cfg = cfg or DAIC_CFG
    dev = _dev()
    split_of = manifest["split_of_subject"]
    sid = ds.subject_id
    idx = {sp: np.array([i for i in range(ds.N) if split_of.get(str(sid[i])) == sp])
           for sp in ("train", "dev", "test")}
    n_c = int(ds.n_classes)
    y_test = ds.y[idx["test"]]
    partial_name = out_name.replace(".json", "_partial.json")
    results, seed_test_f1, test_probs_by_model = {}, {}, {}
    for name in CORE_MODELS:
        build, personalize = _builder(name, ds.F, ds.n_classes)
        dev_s, test_s, bacc_s, test_full, probs_s = [], [], [], [], []
        for s in seeds:
            r = train_select_evaluate(build, ds, idx["train"], idx["dev"], idx["test"],
                                      cfg, dev, seed=s, personalize=personalize)
            dev_s.append(r["dev_metrics"]["macro_f1"])
            test_s.append(r["test_metrics"]["macro_f1"])
            bacc_s.append(r["test_metrics"].get("balanced_accuracy"))
            test_full.append(r["test_metrics"])
            probs_s.append(np.asarray(r["test_probs"]))
            log(f"[DAIC] {name} seed{s}: dev={r['dev_metrics']['macro_f1']:.4f} "
                f"test={r['test_metrics']['macro_f1']:.4f} "
                f"test_bacc={r['test_metrics'].get('balanced_accuracy', float('nan')):.4f}")
        seed_test_f1[name] = np.array(test_s)
        test_probs_by_model[name] = probs_s
        # imbalance diagnostics (seed 0) show collapse-vs-not directly: a majority-collapsed
        # model has a degenerate confusion matrix (one column empty).
        results[name] = {
            "dev_macro_f1_seed_mean": float(np.mean(dev_s)),
            "dev_macro_f1_seed_std": float(np.std(dev_s)),
            "test_macro_f1_seed_mean": float(np.mean(test_s)),
            "test_macro_f1_seed_std": float(np.std(test_s)),
            "per_seed_test_macro_f1": [float(x) for x in test_s],
            "test_balanced_acc_seed_mean": float(np.mean(bacc_s)),
            "test_confusion_seed0": test_full[0].get("confusion_matrix"),
        }
        # crash-resilience checkpoint after each model completes
        Path(out_root).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / partial_name).write_text(
            json.dumps({"completed_models": list(results.keys()), "results": results},
                       indent=2, default=str))

    # PRIMARY inference: participant-level bootstrap on the fixed official test set.
    boot_point, boot_ci, boots = _participant_bootstrap(
        test_probs_by_model, y_test, n_c, n_boot=n_boot, seed=0)
    for name in CORE_MODELS:
        results[name]["test_macro_f1_participant_boot"] = {
            "point": boot_point[name], "ci95": [boot_ci[name][0], boot_ci[name][1]]}

    # D-MSTCN vs each baseline: PRIMARY = paired participant bootstrap of the difference;
    # SECONDARY = seed-level effect sizes (optimization-stability only, not the data unit).
    dm_seed = seed_test_f1["dmstcn"]
    comparisons = {}
    for name in CORE_MODELS:
        if name == "dmstcn":
            continue
        diff_boot = boots["dmstcn"] - boots[name]
        comparisons[name] = {
            "primary_participant_paired_bootstrap": {
                "unit": "test session (participant)",
                "test_macro_f1_diff": float(boot_point["dmstcn"] - boot_point[name]),
                "ci95": [float(np.quantile(diff_boot, 0.025)),
                         float(np.quantile(diff_boot, 0.975))],
                "prob_dmstcn_better": float(np.mean(diff_boot > 0)),
                "n_boot": n_boot,
            },
            "secondary_seed_level": {
                "note": "optimization stability across seeds; NOT the primary data unit",
                "hodges_lehmann_shift": st.hodges_lehmann_paired(dm_seed, seed_test_f1[name]),
                "rank_biserial": st.paired_rank_biserial(dm_seed, seed_test_f1[name]),
                "wilcoxon": st.wilcoxon_paired(dm_seed, seed_test_f1[name]),
            },
        }
    out = {"experiment": "EXP-4.2", "dataset": "e-daic", "protocol": "official_split_dev_select_test",
           "seeds": list(seeds), "config": cfg,
           "training_loss": ("class_balanced_ce" if cfg.get("class_weight") == "balanced"
                             else "plain_ce"),
           "primary_unit": ("test session (participant) via bootstrap on the fixed official "
                            "test set; seeds are secondary optimization repeats"),
           "n_boot": n_boot,
           "class_balance_note": ("imbalanced ~24% positive; class-balanced CE (train-only "
                                  "weights, leakage-safe); macro-F1 primary, balanced-acc reported"),
           "results": results, "dmstcn_vs_baselines": comparisons}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / out_name).write_text(json.dumps(out, indent=2, default=str))
    return out
