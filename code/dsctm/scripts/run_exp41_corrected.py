#!/usr/bin/env python
"""EXP-4.1 confirmatory StudentLife run using the corrected grouped split."""
from dsctm.data.studentlife import build_studentlife
from dsctm.experiments.headline import run_studentlife_headline

ds = build_studentlife(cache="artifacts/cache/studentlife_causal_ffill_v2.npz")
print(f"StudentLife N={ds.N} T={ds.T} F={ds.F}", flush=True)
result = run_studentlife_headline(
    ds, seeds=(0, 1, 2), out_name="studentlife_headline_corrected.json"
)
print(f"split_hash={result['split_hash']}", flush=True)
for name, metrics in result["results"].items():
    print(name, metrics, flush=True)
print("EXP41_CORRECTED_DONE", flush=True)
