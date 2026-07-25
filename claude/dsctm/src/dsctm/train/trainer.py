"""Training loop — scientific-quality mode (master-prompt §6.1).

Leakage-safe: normalization statistics are fit on the TRAIN fold only and applied to
val/test. Cross-subject evaluation is fair: subject indices come from TRAIN subjects
only, with index 0 reserved as an "unknown" subject; during training a fraction of
samples are mapped to 0 (embedding dropout) so unseen test subjects use a *trained*
neutral embedding instead of an accidental one.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..eval.metrics import classification_metrics
from ..repro import seed_worker, set_seed


def fit_normalizer(X):
    flat = X.reshape(-1, X.shape[-1])
    return flat.mean(0).astype(np.float32), (flat.std(0) + 1e-6).astype(np.float32)


def _build_loss(cfg, y_train, n_classes, device):
    """Construct the training loss. When ``cfg['class_weight']`` is requested, class
    weights are computed from the TRAIN labels ONLY (leakage-safe) and passed to
    CrossEntropyLoss. This counteracts majority-class collapse on imbalanced corpora
    (E-DAIC is ~24% positive) and touches no evaluation metric — selection stays on
    dev macro-F1 and test is untouched.

      - "balanced": sklearn convention w_c = n_samples / (n_classes * count_c)
      - list/tuple: explicit per-class weights
      - None / absent: plain (unweighted) cross-entropy — prior behaviour, so any
        config that does not opt in (e.g. StudentLife EXP-4.1) is byte-for-byte unchanged.
    """
    cw = cfg.get("class_weight")
    weight = None
    if cw == "balanced":
        counts = np.bincount(np.asarray(y_train, dtype=np.int64),
                             minlength=n_classes).astype(np.float64)
        counts = np.clip(counts, 1.0, None)  # guard empty class → no div-by-zero
        w = counts.sum() / (n_classes * counts)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    elif isinstance(cw, (list, tuple)):
        weight = torch.tensor(list(cw), dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight)


def _make_loader(ds, idx, subj_map, mean, std, batch_size, shuffle):
    X = ((ds.X[idx] - mean) / std).astype(np.float32)
    y = ds.y[idx].astype(np.int64)
    subj = np.array([subj_map.get(s, 0) for s in ds.subject_id[idx]], dtype=np.int64)
    tds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(subj))
    return DataLoader(tds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      worker_init_fn=seed_worker)


@torch.no_grad()
def evaluate(model, loader, device, personalize):
    model.eval()
    probs, ys = [], []
    for X, y, subj in loader:
        logits = model(X.to(device), subj.to(device) if personalize else None)
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return classification_metrics(ys, probs.argmax(1), probs), probs, ys


def train_model(build_model, ds, tr_idx, va_idx, cfg, device, seed=0,
                personalize=False, emb_dropout=0.1):
    """Train one model on tr_idx, early-stop on val macro-F1. Returns best val metrics,
    val probabilities (for pooled OOF), and the training curve."""
    set_seed(seed, "scientific")
    train_subjects = sorted(set(ds.subject_id[tr_idx].tolist()))
    subj_map = {s: i + 1 for i, s in enumerate(train_subjects)}  # 0 = unknown
    n_subjects = len(train_subjects) + 1

    model = build_model(n_subjects).to(device)
    mean, std = fit_normalizer(ds.X[tr_idx])
    tr = _make_loader(ds, tr_idx, subj_map, mean, std, cfg["batch_size"], True)
    va = _make_loader(ds, va_idx, subj_map, mean, std, cfg["batch_size"], False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"],
                                                       eta_min=cfg["lr_min"])
    lossf = _build_loss(cfg, ds.y[tr_idx], int(ds.n_classes), device)

    best = {"macro_f1": -1.0}
    best_probs, best_epoch, patience, curve = None, 0, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for X, y, subj in tr:
            X, y, subj = X.to(device), y.to(device), subj.to(device)
            if personalize and emb_dropout > 0:
                subj = subj.clone()
                subj[torch.rand(subj.shape[0], device=device) < emb_dropout] = 0
            opt.zero_grad()
            loss = lossf(model(X, subj if personalize else None), y)
            loss.backward()
            opt.step()
        sched.step()
        vm, probs, _ = evaluate(model, va, device, personalize)
        curve.append({"epoch": epoch, "val_macro_f1": vm["macro_f1"], "val_acc": vm["accuracy"]})
        if vm["macro_f1"] > best["macro_f1"]:
            best, best_probs, best_epoch, patience = vm, probs, epoch, 0
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                break
    return {
        "val_metrics": best,
        "val_probs": best_probs,
        "val_true": ds.y[va_idx],
        "val_subjects": ds.subject_id[va_idx],
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "curve": curve,
    }


def headline_cv(build_model, ds, folds, cfg, device, seed=0, personalize=False):
    """Grouped-CV out-of-fold evaluation (EXP-4.1). Returns pooled OOF metrics, per-fold
    macro-F1 (the paired unit for statistics), and the OOF probability matrix."""
    n_c = int(ds.n_classes)
    oof = np.zeros((ds.N, n_c))
    mask = np.zeros(ds.N, bool)
    per_fold = []
    for f, (tr, va) in enumerate(folds):
        r = train_model(build_model, ds, tr, va, cfg, device, seed, personalize)
        oof[va] = r["val_probs"]
        mask[va] = True
        per_fold.append({"fold": f, "macro_f1": r["val_metrics"]["macro_f1"],
                         "accuracy": r["val_metrics"]["accuracy"],
                         "epochs_run": r["epochs_run"]})
    pooled = classification_metrics(ds.y[mask], oof[mask].argmax(1), oof[mask])
    return {"pooled": pooled, "per_fold": per_fold, "oof_probs": oof, "oof_mask": mask,
            "per_fold_macro_f1": [p["macro_f1"] for p in per_fold]}


def train_select_evaluate(build_model, ds, tr_idx, dev_idx, test_idx, cfg, device,
                          seed=0, personalize=False, emb_dropout=0.1):
    """Official-split protocol (EXP-4.2): train on train, SELECT (early-stop) on dev,
    and report TEST metrics at the best-dev epoch — test never touches selection."""
    set_seed(seed, "scientific")
    train_subjects = sorted(set(ds.subject_id[tr_idx].tolist()))
    subj_map = {s: i + 1 for i, s in enumerate(train_subjects)}
    n_subjects = len(train_subjects) + 1
    model = build_model(n_subjects).to(device)
    mean, std = fit_normalizer(ds.X[tr_idx])
    tr = _make_loader(ds, tr_idx, subj_map, mean, std, cfg["batch_size"], True)
    dv = _make_loader(ds, dev_idx, subj_map, mean, std, cfg["batch_size"], False)
    te = _make_loader(ds, test_idx, subj_map, mean, std, cfg["batch_size"], False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"],
                                                       eta_min=cfg["lr_min"])
    lossf = _build_loss(cfg, ds.y[tr_idx], int(ds.n_classes), device)
    best_dev = {"macro_f1": -1.0}
    test_at_best, patience = None, 0
    test_probs_best, test_true_best = None, None
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for X, y, subj in tr:
            X, y, subj = X.to(device), y.to(device), subj.to(device)
            if personalize and emb_dropout > 0:
                subj = subj.clone()
                subj[torch.rand(subj.shape[0], device=device) < emb_dropout] = 0
            opt.zero_grad()
            lossf(model(X, subj if personalize else None), y).backward()
            opt.step()
        sched.step()
        dm, _, _ = evaluate(model, dv, device, personalize)
        if dm["macro_f1"] > best_dev["macro_f1"]:
            # capture the TEST predictions at the best-dev epoch so the caller can run a
            # participant-level bootstrap on the fixed official test set (test never drives
            # selection — only dev macro-F1 does).
            tm, tprobs, ttrue = evaluate(model, te, device, personalize)
            best_dev, test_at_best, patience = dm, tm, 0
            test_probs_best, test_true_best = tprobs, ttrue
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                break
    return {"dev_metrics": best_dev, "test_metrics": test_at_best, "epochs_run": epoch + 1,
            "test_probs": test_probs_best, "test_true": test_true_best}
