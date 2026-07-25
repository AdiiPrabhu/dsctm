#!/usr/bin/env python
"""Build 88-dim eGeMAPS sequences from DAIC-WOZ participant speech only.

Participant intervals are read from the released transcript, clipped to audio bounds,
concatenated chronologically without synthetic gaps, then divided into non-overlapping
0.5-second windows. Interviewer speech is never included. Local cache filenames use the
released coded IDs but the aggregate manifest contains no participant identifiers.
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dsctm.data.daic import (DAICWOZ_ROOT_DEFAULT, FRAME_S, T_MAX, _fit_length,
                             _read_daicwoz_splits)

CACHE_DIR = "artifacts/cache/daicwoz_participant_egemaps88"
_smile = None


def _worker_init():
    global _smile
    import opensmile
    _smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                             feature_level=opensmile.FeatureLevel.Functionals)


def participant_intervals(transcript):
    df = pd.read_csv(transcript, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    rows = df[df["speaker"].astype(str).str.strip().str.casefold() == "participant"]
    return [(float(a), float(b)) for a, b in zip(rows["start_time"], rows["stop_time"])
            if np.isfinite(a) and np.isfinite(b) and b > a]


def _extract_one(args):
    pid, root = args
    out = f"{CACHE_DIR}/{pid}.npz"
    if os.path.exists(out):
        return "cached", None
    wav = f"{root}/{pid}_P/{pid}_AUDIO.wav"
    transcript = f"{root}/{pid}_P/{pid}_TRANSCRIPT.csv"
    if not os.path.exists(wav) or not os.path.exists(transcript):
        return "missing_input", None
    import audiofile
    signal, sr = audiofile.read(wav, always_2d=False)
    if getattr(signal, "ndim", 1) > 1:
        signal = signal.mean(0)
    clips = []
    for start, stop in participant_intervals(transcript):
        lo = max(0, int(round(start * sr)))
        hi = min(len(signal), int(round(stop * sr)))
        if hi > lo:
            clips.append(signal[lo:hi])
    if not clips:
        return "no_participant_audio", None
    participant = np.concatenate(clips)
    width = int(FRAME_S * sr)
    n_windows = min(len(participant) // width, T_MAX)
    if n_windows == 0:
        return "too_short", None
    features = np.empty((n_windows, 88), np.float32)
    for i in range(n_windows):
        window = participant[i * width:(i + 1) * width]
        features[i] = _smile.process_signal(window, sr).values.reshape(-1)
    nan_fraction = float(np.isnan(features).mean())
    features = np.nan_to_num(features, nan=0.0)
    sequence, true_len = _fit_length(features, T_MAX)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(out, X=sequence.astype(np.float32), true_len=true_len,
                        feature_names=np.asarray(_smile.feature_names))
    return "ok", {"true_len": int(true_len), "nan_fraction": nan_fraction}


def main():
    root = DAICWOZ_ROOT_DEFAULT
    splits = _read_daicwoz_splits(root)
    tasks = [(pid, root) for pid in sorted(splits)]
    counts, lengths, nan_fractions = {}, [], []
    started = time.time()
    with Pool(min(8, os.cpu_count() or 4), initializer=_worker_init) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(_extract_one, tasks), 1):
            counts[status] = counts.get(status, 0) + 1
            if info:
                lengths.append(info["true_len"])
                nan_fractions.append(info["nan_fraction"])
            if i % 20 == 0:
                print(f"{i}/{len(tasks)} status_counts={counts}", flush=True)
    manifest = {
        "corpus": "DAIC-WOZ", "representation": "participant-only concatenated speech",
        "window_seconds": FRAME_S, "feature_set": "eGeMAPSv02 Functionals 88",
        "n_requested": len(tasks), "status_counts": counts,
        "true_length_summary": ({"min": min(lengths), "median": float(np.median(lengths)),
                                 "max": max(lengths)} if lengths else None),
        "mean_nan_fraction": float(np.mean(nan_fractions)) if nan_fractions else None,
        "participant_ids_in_manifest": False, "elapsed_seconds": time.time() - started,
    }
    Path("artifacts/resubmission/phase1").mkdir(parents=True, exist_ok=True)
    Path("artifacts/resubmission/phase1/daicwoz_participant_features.json").write_text(
        json.dumps(manifest, indent=2))
    print(manifest, flush=True)
    print("PARTICIPANT_EGEMAPS88_DONE", flush=True)


if __name__ == "__main__":
    main()
