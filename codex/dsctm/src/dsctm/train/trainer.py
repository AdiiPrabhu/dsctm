"""Training loop — scientific-quality mode (master-prompt §6.1).

Leakage-safe: normalization statistics are fit on the TRAIN fold only and applied to
val/test. Cross-subject evaluation is fair: subject indices come from TRAIN subjects
only, with index 0 reserved as an "unknown" subject; during training a fraction of
samples are mapped to 0 (embedding dropout) so unseen test subjects use a *trained*
neutral embedding instead of an accidental one.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..eval.metrics import classification_metrics
from ..repro import seed_worker, set_seed


def fit_normalizer(X, lengths=None):
    if lengths is None:
        flat = X.reshape(-1, X.shape[-1])
    else:
        valid = np.arange(X.shape[1])[None, :] < np.asarray(lengths)[:, None]
        flat = X[valid]
    return np.nanmean(flat, axis=0).astype(np.float32), (
        np.nanstd(flat, axis=0) + 1e-6
    ).astype(np.float32)


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


def _build_tensor_dataset(ds, idx, subj_map, mean, std):
    """Normalize, mask padding, and carry the DATASET-GLOBAL sample id.

    The sample id is what makes distributed evaluation auditable: it lets the gather step
    prove each sample was scored exactly once. It is the index into the full dataset, not
    the position within this split, so ids remain unique across folds.
    """
    X = ((ds.X[idx] - mean) / std).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lengths = np.asarray(ds.lengths[idx], dtype=np.int64)
    mask = np.arange(ds.T)[None, :] < lengths[:, None]
    X[~mask] = 0.0
    y = ds.y[idx].astype(np.int64)
    subj = np.array([subj_map.get(s, 0) for s in ds.subject_id[idx]], dtype=np.int64)
    sample_id = np.asarray(idx, dtype=np.int64)
    return TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(subj),
        torch.from_numpy(mask), torch.from_numpy(sample_id),
    )


def _make_loader(ds, idx, subj_map, mean, std, batch_size, shuffle,
                 ctx=None, train: bool | None = None, seed: int = 0):
    """Build a DataLoader.

    Single-process (``ctx`` None or world_size 1) reproduces the pre-Gate-2 behaviour
    exactly. Distributed uses a padded sampler for training (DDP needs equal step counts)
    and the UNPADDED sampler for evaluation (no sample may be scored twice).
    """
    tds = _build_tensor_dataset(ds, idx, subj_map, mean, std)
    is_train = shuffle if train is None else train

    if ctx is None or not getattr(ctx, "is_distributed", False):
        return DataLoader(tds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                          worker_init_fn=seed_worker)

    from ..distributed import (loader_kwargs_for_param, make_eval_sampler,
                               make_train_sampler)
    sampler = (make_train_sampler(tds, ctx, shuffle=shuffle, seed=seed) if is_train
               else make_eval_sampler(tds, ctx))
    kwargs = loader_kwargs_for_param(num_workers=int(os.environ.get("DSCTM_NUM_WORKERS", 4)))
    return DataLoader(tds, batch_size=batch_size, sampler=sampler,
                      worker_init_fn=seed_worker, **kwargs)


@torch.no_grad()
def evaluate(model, loader, device, personalize, ctx=None, expected_n=None,
             autocast_dtype=None, subject_lookup=None):
    """Evaluate and return (metrics, probabilities, labels).

    Distributed: every rank scores its shard, predictions are all-gathered, and coverage is
    validated BEFORE any metric is computed. A duplicate or missing sample raises rather
    than silently distorting macro-F1.
    """
    model.eval()
    distributed = ctx is not None and getattr(ctx, "is_distributed", False)

    if not distributed:
        probs, ys = [], []
        for batch in loader:
            X, y, subj, mask = batch[0], batch[1], batch[2], batch[3]
            logits = model(
                X.to(device), subj.to(device) if personalize else None, mask=mask.to(device)
            )
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            ys.append(y.numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        return classification_metrics(ys, probs.argmax(1), probs), probs, ys

    from ..distributed import build_records, gather_and_validate, records_to_arrays

    records = []
    for X, y, subj, mask, sample_id in loader:
        with torch.autocast(device_type=device if isinstance(device, str) else device.type,
                            dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = model(
                X.to(device, non_blocking=True),
                subj.to(device, non_blocking=True) if personalize else None,
                mask=mask.to(device, non_blocking=True),
            )
        ids = sample_id.tolist()
        subjects = ([str(subject_lookup[i]) for i in ids] if subject_lookup is not None
                    else [str(i) for i in ids])
        records.extend(build_records(ids, subjects, y.tolist(),
                                     logits.float(), rank=ctx.rank))
    n = expected_n if expected_n is not None else len(loader.dataset)
    merged, audit = gather_and_validate(records, n)
    y_true, y_pred, y_prob, _ = records_to_arrays(merged)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics["_coverage_audit"] = audit
    return metrics, y_prob, y_true


def _prepare_distributed(model, ds, ctx, device, precision):
    """Wrap in DDP (materialising any lazy parameter first) and resolve AMP settings.

    Single-process returns the model untouched and AMP disabled, so the audited
    single-process numerics are bit-for-bit unchanged.
    """
    if ctx is None or not getattr(ctx, "is_distributed", False):
        return model, None, None
    from ..distributed import autocast_dtype as _autocast_dtype
    from ..distributed import build_grad_scaler, wrap_ddp

    example = (
        torch.zeros(2, int(ds.T), int(ds.F), device=device),
        torch.zeros(2, dtype=torch.long, device=device),
        torch.ones(2, int(ds.T), dtype=torch.bool, device=device),
    )
    wrapped = wrap_ddp(model, ctx, example_input=example)
    amp_dtype = _autocast_dtype(precision, torch.device(device) if isinstance(device, str)
                                else device)
    scaler = build_grad_scaler(precision, torch.device(device) if isinstance(device, str)
                               else device)
    return wrapped, amp_dtype, scaler


def _train_one_epoch(model, loader, opt, lossf, device, personalize, emb_dropout,
                     amp_dtype=None, scaler=None, epoch=0, sampler=None):
    """One training epoch. AMP-aware; identical to the pre-Gate-2 loop when amp is off."""
    model.train()
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)          # without this every epoch reuses one shuffle
    device_type = device if isinstance(device, str) else device.type
    for X, y, subj, mask, _sid in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        subj, mask = subj.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        if personalize and emb_dropout > 0:
            subj = subj.clone()
            subj[torch.rand(subj.shape[0], device=device) < emb_dropout] = 0
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss = lossf(model(X, subj if personalize else None, mask=mask), y)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()


def train_model(build_model, ds, tr_idx, va_idx, cfg, device, seed=0,
                personalize=False, emb_dropout=0.1, ctx=None, precision="fp32"):
    """Train one model on tr_idx, early-stop on val macro-F1. Returns best val metrics,
    val probabilities (for pooled OOF), and the training curve.

    ``ctx`` None (default) reproduces the audited single-process path exactly.
    """
    set_seed(seed, "scientific")
    train_subjects = sorted(set(ds.subject_id[tr_idx].tolist()))
    subj_map = {s: i + 1 for i, s in enumerate(train_subjects)}  # 0 = unknown
    n_subjects = len(train_subjects) + 1

    model = build_model(n_subjects).to(device)
    model, amp_dtype, scaler = _prepare_distributed(model, ds, ctx, device, precision)
    mean, std = fit_normalizer(ds.X[tr_idx], ds.lengths[tr_idx])
    tr = _make_loader(ds, tr_idx, subj_map, mean, std, cfg["batch_size"], True,
                      ctx=ctx, train=True, seed=seed)
    va = _make_loader(ds, va_idx, subj_map, mean, std, cfg["batch_size"], False,
                      ctx=ctx, train=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"],
                                                       eta_min=cfg["lr_min"])
    lossf = _build_loss(cfg, ds.y[tr_idx], int(ds.n_classes), device)

    stopper = None
    if ctx is not None and getattr(ctx, "is_distributed", False):
        from ..distributed import EarlyStopCoordinator
        stopper = EarlyStopCoordinator(cfg["early_stop_patience"], ctx=ctx, mode="max")

    best = {"macro_f1": -1.0}
    best_probs, best_epoch, patience, curve = None, 0, 0, []
    for epoch in range(cfg["max_epochs"]):
        _train_one_epoch(model, tr, opt, lossf, device, personalize, emb_dropout,
                         amp_dtype, scaler, epoch, getattr(tr, "sampler", None))
        sched.step()
        vm, probs, _ = evaluate(model, va, device, personalize, ctx=ctx,
                                expected_n=len(va_idx), autocast_dtype=amp_dtype,
                                subject_lookup=ds.subject_id)
        curve.append({"epoch": epoch, "val_macro_f1": vm["macro_f1"], "val_acc": vm["accuracy"]})
        if stopper is not None:
            # Decided on rank 0 and broadcast: every rank leaves the loop on the same epoch.
            decision = stopper.step(vm["macro_f1"], epoch)
            if decision.improved:
                best, best_probs, best_epoch = vm, probs, epoch
            patience = decision.patience
            if decision.should_stop:
                break
        elif vm["macro_f1"] > best["macro_f1"]:
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


def headline_cv(build_model, ds, folds, cfg, device, seed=0, personalize=False,
                on_fold_complete=None):
    """Grouped-CV out-of-fold evaluation (EXP-4.1). Returns pooled OOF metrics, per-fold
    macro-F1 (the paired unit for statistics), and the OOF probability matrix."""
    n_c = int(ds.n_classes)
    oof = np.zeros((ds.N, n_c))
    mask = np.zeros(ds.N, bool)
    per_fold = []
    for f, (tr, va) in enumerate(folds):
        r = train_model(build_model, ds, tr, va, cfg, device, seed, personalize)
        if on_fold_complete is not None:
            on_fold_complete(f, r)
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
    mean, std = fit_normalizer(ds.X[tr_idx], ds.lengths[tr_idx])
    tr = _make_loader(ds, tr_idx, subj_map, mean, std, cfg["batch_size"], True)
    dv = _make_loader(ds, dev_idx, subj_map, mean, std, cfg["batch_size"], False)
    te = _make_loader(ds, test_idx, subj_map, mean, std, cfg["batch_size"], False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"],
                                                       eta_min=cfg["lr_min"])
    lossf = _build_loss(cfg, ds.y[tr_idx], int(ds.n_classes), device)
    best_dev = {"macro_f1": -1.0}
    best_state, patience, best_epoch = None, 0, -1
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for X, y, subj, mask, _sid in tr:
            X, y, subj, mask = X.to(device), y.to(device), subj.to(device), mask.to(device)
            if personalize and emb_dropout > 0:
                subj = subj.clone()
                subj[torch.rand(subj.shape[0], device=device) < emb_dropout] = 0
            opt.zero_grad()
            lossf(model(X, subj if personalize else None, mask=mask), y).backward()
            opt.step()
        sched.step()
        dm, _, _ = evaluate(model, dv, device, personalize)
        if dm["macro_f1"] > best_dev["macro_f1"]:
            # Select only on development data. Test is evaluated exactly once after
            # training, using the frozen best-development state.
            best_dev, patience, best_epoch = dm, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                break
    if best_state is None:
        raise RuntimeError("no development checkpoint selected")
    model.load_state_dict(best_state)
    test_at_best, test_probs_best, test_true_best = evaluate(model, te, device, personalize)
    return {"dev_metrics": best_dev, "test_metrics": test_at_best,
            "best_epoch": best_epoch, "epochs_run": epoch + 1,
            "test_probs": test_probs_best, "test_true": test_true_best}
