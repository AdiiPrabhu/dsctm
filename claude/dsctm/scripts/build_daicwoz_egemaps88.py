#!/usr/bin/env python
"""Extract 88-dim eGeMAPS functionals (0.5s windows) from the classic DAIC-WOZ audio ->
artifacts/cache/daicwoz_egemaps88/{pid}.npz. Reuses the shared corpus-agnostic extractor.

PROVENANCE CAVEAT (recorded, not hidden): DAIC-WOZ audio (XXX_AUDIO.wav) is the full
Wizard-of-Oz interview and includes the virtual interviewer (Ellie) turns; these functionals
are computed over the whole recording. A stricter setup would restrict to participant speech
via the transcript. We keep whole-recording functionals here to match the E-DAIC 88-dim
pipeline exactly (same treatment for both corpora), and flag this for the response letter.

Run:  PYTHONPATH=src python -u scripts/build_daicwoz_egemaps88.py
"""
import json
from pathlib import Path

from dsctm.data.daic import DAICWOZ_AUDIO_GLOB, DAICWOZ_ROOT_DEFAULT, _read_daicwoz_splits
from scripts.build_daic_egemaps88 import extract_corpus

CACHE_DIR = "artifacts/cache/daicwoz_egemaps88"
_ROOT_CANDIDATES = [
    DAICWOZ_ROOT_DEFAULT,
    "/mnt/adissd/phd/dsctm-resubmission/dataset/DAIC-WOZ",
    str(Path(__file__).resolve().parents[3] / "dataset" / "DAIC-WOZ"),
]


def main():
    root = next((r for r in _ROOT_CANDIDATES
                 if (Path(r) / "train_split_Depression_AVEC2017.csv").exists()),
                DAICWOZ_ROOT_DEFAULT)
    print(f"DAIC-WOZ root: {root}", flush=True)
    splits = _read_daicwoz_splits(root)
    pids = sorted(splits.keys())
    counts = {s: sum(1 for v in splits.values() if v[0] == s) for s in ("train", "dev", "test")}
    print(f"{len(pids)} labeled sessions {counts}", flush=True)
    results = extract_corpus(pids, root, DAICWOZ_AUDIO_GLOB, CACHE_DIR)
    ok = [p for p, v in results.items() if v["status"] == "ok"]
    cached = [p for p, v in results.items() if v["status"] == "cached"]
    noaud = [p for p, v in results.items() if v["status"] == "no_audio"]
    print(f"DONE ok={len(ok)} cached={len(cached)} no_audio={len(noaud)} "
          f"other={len(pids)-len(ok)-len(cached)-len(noaud)}", flush=True)
    if noaud:
        print("NO AUDIO pids:", noaud, flush=True)
    Path("artifacts").mkdir(exist_ok=True)
    (Path("artifacts") / "daicwoz_egemaps88_manifest.json").write_text(
        json.dumps({"root": root, "counts": counts, "results": results}, indent=2))
    print("EXTRACT_WOZ88_DONE", flush=True)


if __name__ == "__main__":
    main()
