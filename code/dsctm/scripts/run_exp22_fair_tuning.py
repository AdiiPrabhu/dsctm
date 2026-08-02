#!/usr/bin/env python
import os

from dsctm.data.daic import build_daicwoz88
from dsctm.experiments.fair_tuning import run_fair_tuning

cache = os.environ.get("DSCTM_DAICWOZ_CACHE", "artifacts/cache/daicwoz_egemaps88")
ds, manifest = build_daicwoz88(cache_dir=cache)
result = run_fair_tuning(ds, manifest)
print({k: (None if v is None else v["params"]) for k, v in result["selected"].items()},
      flush=True)
print("EXP22_FAIR_TUNING_DONE", flush=True)
