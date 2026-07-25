"""StudentLife loader → WindowedDataset (master-prompt §7.1).

Label: Stress EMA (`EMA/response/Stress/Stress_u*.json`), 5-point NON-monotonic scale
  [1]a little stressed [2]definitely stressed [3]stressed out [4]feeling good [5]feeling great
mapped to 3 classes (documented, configurable):
  {4,5} → 0 low  ·  {1} → 1 moderate  ·  {2,3} → 2 high

Features (F=8, 1-minute bins over the 60 min preceding each EMA response):
  0 activity_mean      (accelerometer/activity inference)
  1 audio_mean         (ambient audio inference)
  2 conversation_frac  (fraction of minute in conversation)
  3 dark_frac          (fraction of minute phone in dark)
  4 phonelock_frac     (screen-off fraction)
  5 phonecharge_frac   (charging fraction)
  6 gps_speed_mean     (m/s)
  7 gps_moving_frac    (travelstate != stationary)

Windows carry RAW features; leakage-safe normalization/imputation happen AFTER the
subject split (master-prompt §7.1). Missing minute-bins are forward-filled here and the
per-window missingness fraction is recorded.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from .contract import WindowedDataset

SL_ROOT_DEFAULT = os.environ.get(
    "DSCTM_STUDENTLIFE_ROOT",
    "/mnt/adissd/phd/dsctm-resubmission/dataset/StudentLife/dataset",
)
FEATURE_NAMES = [
    "activity_mean", "audio_mean", "conversation_frac", "dark_frac",
    "phonelock_frac", "phonecharge_frac", "gps_speed_mean", "gps_moving_frac",
]
T = 60
BIN_S = 60
WINDOW_S = T * BIN_S  # 3600

# Stress level (1-5) → 3-class stress. Documented, configurable.
DEFAULT_STRESS_MAP = {4: 0, 5: 0, 1: 1, 2: 2, 3: 2}


def _uid(path):
    m = re.search(r"_u(\d+)", os.path.basename(path))
    return f"u{int(m.group(1)):02d}" if m else None


def _read_csv(path, cols=None):
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd

        # index_col=False: some files (gps) have a trailing comma; without this
        # pandas auto-promotes the first column to the index and shifts the rest.
        df = pd.read_csv(path, skipinitialspace=True, index_col=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


def _bin_point(times, values, t0):
    """Mean of `values` per 1-min bin over [t0, t0+3600). NaN where empty."""
    out = np.full(T, np.nan)
    if times is None or len(times) == 0:
        return out
    rel = (np.asarray(times, float) - t0)
    m = (rel >= 0) & (rel < WINDOW_S)
    if not m.any():
        return out
    b = (rel[m] // BIN_S).astype(int)
    v = np.asarray(values, float)[m]
    sums = np.zeros(T)
    cnts = np.zeros(T)
    np.add.at(sums, b, v)
    np.add.at(cnts, b, 1.0)
    nz = cnts > 0
    out[nz] = sums[nz] / cnts[nz]
    return out


def _bin_interval(starts, ends, t0):
    """Fraction of each 1-min bin covered by any [start,end) interval."""
    if starts is None or len(starts) == 0:
        # These streams encode event occurrence; no recorded interval is represented
        # as zero coverage, matching the published feature contract.
        return np.zeros(T)
    out = np.zeros(T)
    starts = np.asarray(starts, float)
    ends = np.asarray(ends, float)
    for k in range(T):
        bs = t0 + k * BIN_S
        be = bs + BIN_S
        ov = np.clip(np.minimum(ends, be) - np.maximum(starts, bs), 0, None).sum()
        out[k] = min(ov / BIN_S, 1.0)
    return out


def _ffill(x):
    """Strictly causal forward-fill; leading NaNs remain zero.

    No value at time ``t`` may be filled from an observation at ``t' > t``. The
    previous implementation replaced a leading missing prefix with the first later
    observation, which was backward-fill leakage despite the function name.
    """
    idx = np.arange(len(x))
    valid = ~np.isnan(x)
    if not valid.any():
        return np.zeros_like(x), 1.0
    last = np.where(valid, idx, -1)
    np.maximum.accumulate(last, out=last)
    filled = np.zeros_like(x)
    seen = last >= 0
    filled[seen] = x[last[seen]]
    miss = float((~valid).mean())
    return filled, miss


def _user_sensors(root, uid):
    s = {}
    a = _read_csv(f"{root}/sensing/activity/activity_{uid}.csv")
    s["activity"] = (a["timestamp"].values, a["activity inference"].values) if a is not None and "activity inference" in a else (None, None)
    au = _read_csv(f"{root}/sensing/audio/audio_{uid}.csv")
    s["audio"] = (au["timestamp"].values, au["audio inference"].values) if au is not None and "audio inference" in au else (None, None)
    g = _read_csv(f"{root}/sensing/gps/gps_{uid}.csv")
    if g is not None and "time" in g:
        spd = g["speed"].values if "speed" in g else np.zeros(len(g))
        mov = (g["travelstate"].values != "stationary").astype(float) if "travelstate" in g else np.zeros(len(g))
        s["gps"] = (g["time"].values, spd, mov)
    else:
        s["gps"] = (None, None, None)
    for name, folder in [("conversation", "conversation"), ("dark", "dark"),
                         ("phonelock", "phonelock"), ("phonecharge", "phonecharge")]:
        d = _read_csv(f"{root}/sensing/{folder}/{folder}_{uid}.csv")
        if d is not None and d.shape[1] >= 2:
            s[name] = (d.iloc[:, 0].values, d.iloc[:, 1].values)
        else:
            s[name] = (None, None)
    return s


def _stress_events(root, uid, stress_map):
    path = f"{root}/EMA/response/Stress/Stress_{uid}.json"
    if not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
    except Exception:
        return []
    out = []
    for r in data:
        if isinstance(r, dict) and str(r.get("level", "")).strip().isdigit():
            lvl = int(r["level"])
            if lvl in stress_map and "resp_time" in r:
                out.append((int(r["resp_time"]), stress_map[lvl]))
    return out


def build_studentlife(root=SL_ROOT_DEFAULT, stress_map=None, min_windows_per_subject=5,
                      cache=None, imputation="causal_ffill"):
    """Build the full StudentLife WindowedDataset. Optionally cache to an .npz."""
    stress_map = stress_map or DEFAULT_STRESS_MAP
    if cache and os.path.exists(cache):
        return _load_cache(cache)

    uids = sorted({_uid(p) for p in glob.glob(f"{root}/EMA/response/Stress/Stress_u*.json")})
    X, y, sid, ts, miss = [], [], [], [], []
    for uid in uids:
        events = _stress_events(root, uid, stress_map)
        if len(events) < min_windows_per_subject:
            continue
        sensors = _user_sensors(root, uid)
        for resp_time, label in events:
            t0 = resp_time - WINDOW_S
            feats = np.zeros((T, len(FEATURE_NAMES)))
            at, av = sensors["activity"]
            feats[:, 0] = _bin_point(at, av, t0)
            aut, auv = sensors["audio"]
            feats[:, 1] = _bin_point(aut, auv, t0)
            feats[:, 2] = _bin_interval(*sensors["conversation"], t0)
            feats[:, 3] = _bin_interval(*sensors["dark"], t0)
            feats[:, 4] = _bin_interval(*sensors["phonelock"], t0)
            feats[:, 5] = _bin_interval(*sensors["phonecharge"], t0)
            gt, gs, gm = sensors["gps"]
            feats[:, 6] = _bin_point(gt, gs, t0)
            feats[:, 7] = _bin_point(gt, gm, t0)
            missing = np.isnan(feats)
            if imputation == "causal_ffill":
                feats = np.column_stack([_ffill(feats[:, j])[0]
                                         for j in range(feats.shape[1])])
            elif imputation == "zero":
                feats = np.nan_to_num(feats, nan=0.0)
            elif imputation == "train_mean":
                # Preserve NaNs. trainer.fit_normalizer estimates each feature using
                # TRAIN-fold observed values only; _make_loader maps normalized NaNs to 0.
                pass
            elif imputation == "mask_aware_zero":
                feats = np.concatenate(
                    [np.nan_to_num(feats, nan=0.0), (~missing).astype(float)], axis=1
                )
            else:
                raise ValueError(f"unknown StudentLife imputation {imputation!r}")
            X.append(feats.astype(np.float32)); y.append(label); sid.append(uid)
            ts.append(resp_time); miss.append(float(missing.mean()))

    names = (FEATURE_NAMES + [f"{n}_observed" for n in FEATURE_NAMES]
             if imputation == "mask_aware_zero" else FEATURE_NAMES)

    ds = WindowedDataset(
        X=np.asarray(X, np.float32), y=np.asarray(y, int), subject_id=np.asarray(sid),
        timestamp=np.asarray(ts), feature_names=names, label_type="multiclass",
        n_classes=3, sampling_interval_s=60.0, dataset="studentlife",
        version=f"studentlife-v2-{imputation}",
    )
    ds._missingness = np.asarray(miss)  # attached for the provenance report
    if cache:
        _save_cache(ds, cache)
    return ds


def _save_cache(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=ds.X, y=ds.y, subject_id=ds.subject_id, timestamp=ds.timestamp,
                        missingness=getattr(ds, "_missingness", np.zeros(ds.N)),
                        feature_names=np.asarray(ds.feature_names), version=ds.version)


def _load_cache(path):
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z["feature_names"]] if "feature_names" in z else FEATURE_NAMES
    if "version" in z:
        version = str(z["version"])
    else:
        # Early v2 caches were generated before version/feature-name metadata was added
        # to the NPZ schema. Recover only versions that are unambiguous from the
        # prespecified cache filename; genuinely ambiguous old caches remain legacy.
        stem = os.path.basename(os.fspath(path)).removesuffix(".npz")
        known = {
            "studentlife_causal_ffill_v2": "causal_ffill",
            "studentlife_v2_causal_ffill": "causal_ffill",
            "studentlife_v2_train_mean": "train_mean",
            "studentlife_v2_zero": "zero",
            "studentlife_v2_mask_aware_zero": "mask_aware_zero",
        }
        version = (f"studentlife-v2-{known[stem]}" if stem in known
                   else "studentlife-v1-legacy")
    ds = WindowedDataset(X=z["X"], y=z["y"], subject_id=z["subject_id"], timestamp=z["timestamp"],
                         feature_names=names, label_type="multiclass", n_classes=3,
                         sampling_interval_s=60.0, dataset="studentlife",
                         version=version)
    ds._missingness = z["missingness"]
    return ds
