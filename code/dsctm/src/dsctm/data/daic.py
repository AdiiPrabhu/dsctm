"""E-DAIC loader → WindowedDataset (master-prompt §7.2).

IMPORTANT provenance facts (surfaced for the response letter):
  * On disk is **E-DAIC (AVEC-2019)**: 274 sessions with OFFICIAL train/dev/test
    splits (163/56/55) — NOT the classic DAIC-WOZ (189 sessions, 107/82) the
    manuscript cites. We use the official splits (this also resolves the reviewer's
    107/82 objection — there is no dev+test merge).
  * Acoustic features on disk are eGeMAPS **LLDs (23-dim) @ 100 Hz, openSMILE 2.3.0**.
    The manuscript states 88-dim @ 0.5 s (openSMILE 3.0). We aggregate the 23 LLDs into
    0.5 s frames; feature count therefore differs from the paper — flagged, not hidden.

Label: PHQ_Binary (PHQ-8 ≥ 10) → 2 classes.  Split membership is stored in a manifest
(official split), never merged.
"""
from __future__ import annotations

import glob
import os

import numpy as np

DAIC_ROOT_DEFAULT = os.environ.get(
    "DSCTM_EDAIC_ROOT", "/mnt/adissd/phd/dsctm-resubmission/dataset/daicwoz"
)
EGEMAPS_GLOB = "{root}/data/{pid}_P/features/{pid}_OpenSMILE2.3.0_egemaps.csv"
FRAME_S = 0.5
T_MAX = 2000


def _read_splits(root):
    """Return {participant_id: (split_name, phq_binary, phq_score)} from official files."""
    import pandas as pd

    out = {}
    for split in ("train", "dev", "test"):
        path = f"{root}/labels/{split}_split.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.iterrows():
            pid = int(r["Participant_ID"])
            binv = int(r["PHQ_Binary"]) if "PHQ_Binary" in df.columns and not np.isnan(r["PHQ_Binary"]) else None
            score = float(r["PHQ_Score"]) if "PHQ_Score" in df.columns else None
            out[pid] = (split, binv, score)
    return out


def _aggregate_egemaps(path, frame_s=FRAME_S):
    """Aggregate 100 Hz eGeMAPS LLDs into `frame_s`-second mean frames → (n_frames, 23)."""
    import pandas as pd

    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    cols = [c for c in df.columns if c not in ("name", "frameTime")]
    ft = df["frameTime"].to_numpy(float)
    vals = df[cols].to_numpy(float)
    bins = (ft // frame_s).astype(int)
    nb = int(bins.max()) + 1
    agg = np.zeros((nb, len(cols)))
    cnt = np.zeros(nb)
    np.add.at(agg, bins, vals)
    np.add.at(cnt, bins, 1.0)
    return agg / np.maximum(cnt, 1.0)[:, None], cols


def _fit_length(seq, T):
    """Truncate (head) or zero-pad a (n,F) sequence to exactly T frames; return (T,F), true_len."""
    n = seq.shape[0]
    if n >= T:
        return seq[:T], T
    out = np.zeros((T, seq.shape[1]), seq.dtype)
    out[:n] = seq
    return out, n


def cache_participant(root, pid, cache_dir, T=T_MAX):
    """Aggregate + length-fit one participant, cache to <cache_dir>/<pid>.npz. Resumable."""
    os.makedirs(cache_dir, exist_ok=True)
    out_path = f"{cache_dir}/{pid}.npz"
    if os.path.exists(out_path):
        return out_path
    feat_path = EGEMAPS_GLOB.format(root=root, pid=pid)
    if not os.path.exists(feat_path):
        return None
    agg, cols = _aggregate_egemaps(feat_path)
    seq, true_len = _fit_length(agg.astype(np.float32), T)
    np.savez_compressed(out_path, X=seq, true_len=true_len, feature_names=np.array(cols))
    return out_path


def build_daic(root=DAIC_ROOT_DEFAULT, T=T_MAX, cache_dir=None, limit=None):
    """Assemble the E-DAIC WindowedDataset from official splits.

    Returns (WindowedDataset, split_manifest). Participant-level caches are reused if
    present, so this can follow a background caching pass.
    """
    from .contract import WindowedDataset

    cache_dir = cache_dir or f"{root}/../cache/daic_egemaps"
    splits = _read_splits(root)
    pids = sorted(splits.keys())
    if limit:
        pids = pids[:limit]

    X, y, sid, split_of, lengths = [], [], [], {}, []
    feat_names = None
    skipped = []
    for pid in pids:
        split, binv, _ = splits[pid]
        if binv is None:
            skipped.append((pid, "no_label"))
            continue
        cpath = cache_participant(root, pid, cache_dir, T)
        if cpath is None:
            skipped.append((pid, "no_features"))
            continue
        z = np.load(cpath, allow_pickle=True)
        X.append(z["X"])
        y.append(int(binv))
        sid.append(str(pid))
        lengths.append(int(z["true_len"]))
        split_of[str(pid)] = split
        if feat_names is None:
            feat_names = [str(c) for c in z["feature_names"]]

    ds = WindowedDataset(
        X=np.asarray(X, np.float32), y=np.asarray(y, int), subject_id=np.asarray(sid),
        timestamp=None, feature_names=feat_names, label_type="binary", n_classes=2,
        sampling_interval_s=FRAME_S, dataset="e-daic", version="edaic-egemaps-lld23-v1",
        lengths=np.asarray(lengths),
    )
    ds._lengths = ds.lengths
    manifest = {
        "scheme": "official_edaic_split",
        "split_of_subject": split_of,
        "counts": {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "dev", "test")},
        "skipped": skipped,
        "note": "E-DAIC 274 official splits; manuscript cites classic DAIC-WOZ 189 / 107-82 (discrepancy).",
    }
    return ds, manifest


_EGEMAPS88_FEATURE_NOTE = (
    "eGeMAPSv02 Functionals (88-dim) over consecutive 0.5s windows; openSMILE 3.0 core via "
    "opensmile 2.6.0; reproduction of the manuscript's '88-dim eGeMAPS @0.5s openSMILE 3.0'.")


def _assemble88(splits, cache_dir, dataset, version, scheme, note):
    """Shared assembler: build a WindowedDataset from PRE-EXTRACTED 88-dim eGeMAPS
    functionals caches (built from raw audio). NEVER falls back to other features — a session
    without an 88-dim cache is skipped and recorded, so feature sets can't be silently mixed.
    `splits` maps pid -> (split_name, phq_binary, phq_score)."""
    from .contract import WindowedDataset

    pids = sorted(splits.keys())
    X, y, sid, split_of, lengths, skipped = [], [], [], {}, [], []
    feat_names = None
    for pid in pids:
        split, binv, _ = splits[pid]
        if binv is None:
            skipped.append((pid, "no_label"))
            continue
        cpath = f"{cache_dir}/{pid}.npz"
        if not os.path.exists(cpath):
            skipped.append((pid, "no_features88"))
            continue
        z = np.load(cpath, allow_pickle=True)
        X.append(z["X"])
        y.append(int(binv))
        sid.append(str(pid))
        lengths.append(int(z["true_len"]))
        split_of[str(pid)] = split
        if feat_names is None:
            feat_names = [str(c) for c in z["feature_names"]]

    ds = WindowedDataset(
        X=np.asarray(X, np.float32), y=np.asarray(y, int), subject_id=np.asarray(sid),
        timestamp=None, feature_names=feat_names, label_type="binary", n_classes=2,
        sampling_interval_s=FRAME_S, dataset=dataset, version=version,
        lengths=np.asarray(lengths),
    )
    ds._lengths = ds.lengths
    manifest = {
        "scheme": scheme,
        "split_of_subject": split_of,
        "counts": {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "dev", "test")},
        "skipped": skipped,
        "feature_set": _EGEMAPS88_FEATURE_NOTE,
        "note": note,
    }
    return ds, manifest


def build_daic88(root=DAIC_ROOT_DEFAULT, T=T_MAX, cache_dir="artifacts/cache/daic_egemaps88"):
    """E-DAIC WindowedDataset from 88-dim eGeMAPS functionals caches (the manuscript's stated
    feature set). Official E-DAIC split; see `_assemble88`."""
    return _assemble88(_read_splits(root), cache_dir, "e-daic",
                       "edaic-egemaps88-functionals-v1", "official_edaic_split",
                       "88-dim feature-fidelity reproduction; corpus is E-DAIC official splits.")


DAICWOZ_ROOT_DEFAULT = os.environ.get(
    "DSCTM_DAICWOZ_ROOT", "/mnt/adissd/phd/dsctm-resubmission/dataset/DAIC-WOZ"
)
DAICWOZ_AUDIO_GLOB = "{root}/{pid}_P/{pid}_AUDIO.wav"


def _read_daicwoz_splits(root):
    """Official AVEC2017 DAIC-WOZ split. train/dev from *_split_Depression_AVEC2017.csv
    (label column PHQ8_Binary); test from full_test_split.csv (the RELEASED test labels,
    column PHQ_Binary — the blind test_split has no labels). Uses the proper 107/35/47
    train/dev/test with NO dev+test merge (this is what answers the reviewer's 107/82
    objection). Returns {pid: (split, phq_binary, phq_score)}."""
    import pandas as pd

    out = {}
    spec = [
        ("train", "train_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
        ("dev", "dev_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
        ("test", "full_test_split.csv", "PHQ_Binary", "PHQ_Score"),
    ]
    for split, fname, bcol, scol in spec:
        path = f"{root}/{fname}"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.iterrows():
            pid = int(r["Participant_ID"])
            binv = int(r[bcol]) if bcol in df.columns and not pd.isna(r[bcol]) else None
            score = float(r[scol]) if scol in df.columns and not pd.isna(r[scol]) else None
            out[pid] = (split, binv, score)
    return out


def build_daicwoz88(root=DAICWOZ_ROOT_DEFAULT, T=T_MAX,
                    cache_dir="artifacts/cache/daicwoz_egemaps88"):
    """Classic DAIC-WOZ (AVEC2017) WindowedDataset from 88-dim eGeMAPS functionals caches
    (extracted from the on-disk audio) — the corpus the manuscript actually cites and the
    feature set it states. Official 107/35/47 train/dev-select/test, no merge; see `_assemble88`."""
    return _assemble88(_read_daicwoz_splits(root), cache_dir, "daic-woz",
                       "daicwoz-egemaps88-functionals-v1", "official_avec2017_split",
                       "classic DAIC-WOZ (AVEC2017); official train/dev/test, no dev+test merge; "
                       "88-dim eGeMAPS reproduction from raw audio.")
