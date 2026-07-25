#!/usr/bin/env python
"""Extract eGeMAPSv02 **88-dim FUNCTIONALS** over non-overlapping 0.5 s windows from the raw
E-DAIC audio -> a (T, 88) sequence per session, cached in the same npz format as the 23-dim
LLD cache so the existing loader can consume it.

WHY: the manuscript states it used **88-dim eGeMAPS @ 0.5 s, openSMILE 3.0**, but the disk
ships only 23-dim eGeMAPS LLDs (openSMILE 2.3.0). This rebuilds the paper's actual feature
set from the on-disk audio (opensmile 2.6.0 bundles the openSMILE 3.0 core), applied
identically to every model. This is fidelity to the paper, NOT tuning toward a result.

DOCUMENTED ASSUMPTIONS (confirm with the author):
  * "88-dim" = eGeMAPSv02 Functionals (verified: 88 features).
  * "@0.5 s" = functionals computed over consecutive 0.5 s windows -> a temporal sequence
    (the representation a temporal model like D-MSTCN consumes). NOTE the classic AVEC/DAIC
    "88-dim eGeMAPS" is usually ONE functional vector over the whole recording; the windowed
    reading here is an interpretation for the sequence model and is flagged as such.
  * T capped at 2000 windows (=1000 s) to match the existing pipeline; head-truncate longer,
    zero-pad shorter.
  * Silence windows can yield NaN functionals (e.g. F0 over unvoiced frames) -> filled with 0;
    per-session NaN fraction recorded.

Run (parallel across sessions, ~8-10 min):  PYTHONPATH=src python -u scripts/build_daic_egemaps88.py
"""
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dsctm.data.daic import DAIC_ROOT_DEFAULT, FRAME_S, T_MAX, _fit_length, _read_splits

AUDIO_GLOB = "{root}/data/{pid}_P/{pid}_AUDIO.wav"
CACHE_DIR = "artifacts/cache/daic_egemaps88"
_ROOT_CANDIDATES = [
    DAIC_ROOT_DEFAULT,
    "/mnt/adissd/phd/dsctm-resubmission/dataset/daicwoz",
    str(Path(__file__).resolve().parents[3] / "dataset" / "daicwoz"),
]


def _resolve_root():
    for r in _ROOT_CANDIDATES:
        if (Path(r) / "labels" / "train_split.csv").exists():
            return r
    return DAIC_ROOT_DEFAULT


_smile = None


def _worker_init():
    global _smile
    import opensmile
    _smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                             feature_level=opensmile.FeatureLevel.Functionals)


def _extract_one(args):
    pid, root, audio_glob, cache_dir = args
    out = f"{cache_dir}/{pid}.npz"
    if os.path.exists(out):
        return (pid, "cached", None)
    wav = audio_glob.format(root=root, pid=pid)
    if not os.path.exists(wav):
        return (pid, "no_audio", None)
    import audiofile
    sig, sr = audiofile.read(wav, always_2d=False)
    if getattr(sig, "ndim", 1) > 1:
        sig = sig.mean(0)  # collapse to mono
    W = int(FRAME_S * sr)
    n = min(len(sig) // W, T_MAX)
    if n == 0:
        return (pid, "too_short", None)
    feats = np.empty((n, 88), np.float32)
    for i in range(n):
        feats[i] = _smile.process_signal(sig[i * W:(i + 1) * W], sr).values.reshape(-1)
    nanfrac = float(np.isnan(feats).mean())
    feats = np.nan_to_num(feats, nan=0.0)
    seq, true_len = _fit_length(feats, T_MAX)  # pad/truncate to T_MAX
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(out, X=seq.astype(np.float32), true_len=true_len,
                        feature_names=np.array(list(_smile.feature_names)))
    return (pid, "ok", {"true_len": int(true_len), "n_win": int(n), "nanfrac": round(nanfrac, 4)})


def extract_corpus(pids, root, audio_glob, cache_dir, workers=None):
    """Extract 88-dim eGeMAPS functionals (0.5s windows) for `pids` from `audio_glob`
    into `cache_dir` (npz per session). Corpus-agnostic; shared by E-DAIC and DAIC-WOZ.
    Returns {pid: {status, ...}}."""
    os.makedirs(cache_dir, exist_ok=True)
    workers = workers or min(8, (os.cpu_count() or 4))
    print(f"extracting 88-dim eGeMAPS functionals for {len(pids)} sessions with {workers} "
          f"workers -> {cache_dir}/", flush=True)
    t0 = time.time()
    results = {}
    tasks = [(p, root, audio_glob, cache_dir) for p in pids]
    with Pool(workers, initializer=_worker_init) as pool:
        for k, (pid, status, info) in enumerate(pool.imap_unordered(_extract_one, tasks), 1):
            results[str(pid)] = {"status": status, **(info or {})}
            if status != "cached" or k % 25 == 0:
                print(f"[{time.time()-t0:.0f}s] {k}/{len(pids)} pid={pid} {status} {info or ''}",
                      flush=True)
    return results


def main():
    root = _resolve_root()
    print(f"E-DAIC root: {root}", flush=True)
    splits = _read_splits(root)
    pids = sorted(splits.keys())
    results = extract_corpus(pids, root, AUDIO_GLOB, CACHE_DIR)
    by = lambda st: [p for p, v in results.items() if v["status"] == st]
    ok, cached, noaud = by("ok"), by("cached"), by("no_audio")
    nanmean = np.mean([v["nanfrac"] for v in results.values() if v["status"] == "ok"]) if ok else 0.0
    print(f"DONE ok={len(ok)} cached={len(cached)} no_audio={len(noaud)} "
          f"other={len(pids)-len(ok)-len(cached)-len(noaud)} in {time.time()-t0:.0f}s "
          f"| mean per-session NaN frac (silence) = {nanmean:.4f}", flush=True)
    if noaud:
        print("NO AUDIO pids:", noaud, flush=True)
    Path(CACHE_DIR).parent.mkdir(parents=True, exist_ok=True)
    (Path("artifacts") / "daic_egemaps88_manifest.json").write_text(
        json.dumps({"root": root, "n_sessions": len(pids), "results": results}, indent=2))
    print("EXTRACT88_DONE", flush=True)


if __name__ == "__main__":
    main()
