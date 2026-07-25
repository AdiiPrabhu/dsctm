#!/usr/bin/env python
"""Run EXP-5.1/5.2/5.5 on the corrected StudentLife folds."""
from dsctm.data.studentlife import build_studentlife
from dsctm.experiments.ablation import run_studentlife_ablation

ds = build_studentlife(cache="artifacts/cache/studentlife_causal_ffill_v2.npz")
print(f"StudentLife N={ds.N} T={ds.T} F={ds.F}", flush=True)
result = run_studentlife_ablation(ds, seeds=(0, 1, 2))
print(f"split_hash={result['split_hash']}", flush=True)
for name, metrics in result["results"].items():
    print(name, metrics, flush=True)
print("PHASE5_ABLATION_DONE", flush=True)
