#!/usr/bin/env python
"""Cache E-DAIC eGeMAPS features (0.5s-aggregated), assemble the WindowedDataset,
and run the Gate 1 audit. Heavy (274 sessions × ~18 MB) — run in the background:

    python scripts/build_daic.py
"""
import json
import time

from dsctm.data.daic import build_daic
from dsctm.experiments.gate1 import run_gate1_daic

t0 = time.time()
ds, manifest = build_daic(cache_dir="artifacts/cache/daic_egemaps")
print(f"[{time.time()-t0:.0f}s] assembled: N={ds.N} T={ds.T} F={ds.F} "
      f"subjects={len(set(ds.subject_id))} splits={manifest['counts']} skipped={len(manifest['skipped'])}")
res = run_gate1_daic(ds, manifest)
print(json.dumps(res["provenance"], indent=2, default=str))
print("leakage_free:", res["leakage"]["leakage_free"])
print("DAIC_BUILD_DONE")
