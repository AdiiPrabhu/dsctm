#!/usr/bin/env python
import os

from dsctm.data.daic import build_daicwoz88
from dsctm.experiments.fair_tuning import run_fair_tuning

cache = os.environ.get("DSCTM_DAICWOZ_CACHE", "artifacts/cache/daicwoz_egemaps88")
variant = os.environ.get("DSCTM_FAIR_VARIANT", "").strip()
suffix = f"__{variant}" if variant else ""
out_root = os.environ.get(
    "DSCTM_FAIR_OUT_ROOT",
    f"artifacts/resubmission/phase2/{variant}" if variant else "artifacts/resubmission/phase2",
)
ds, manifest = build_daicwoz88(cache_dir=cache)
result = run_fair_tuning(ds, manifest, out_root=out_root, condition_suffix=suffix)
print({k: (None if v is None else v["params"]) for k, v in result["selected"].items()},
      flush=True)
print("EXP22_FAIR_TUNING_DONE", flush=True)
