#!/usr/bin/env python
"""Phase 4 headline evaluation on this GPU:
  EXP-4.1 StudentLife subject-grouped 5-fold  (3 seeds)
  EXP-4.2 E-DAIC official split, dev-select/test (2 seeds)
Writes artifacts/resubmission/phase4/{studentlife,daic}_headline.json
"""
import json

from dsctm.data.daic import build_daic
from dsctm.data.studentlife import build_studentlife
from dsctm.experiments.headline import DAIC_CFG, run_daic_headline, run_studentlife_headline

print("=== StudentLife headline (EXP-4.1) ===", flush=True)
sl = build_studentlife(cache="artifacts/cache/studentlife.npz")
r1 = run_studentlife_headline(sl, seeds=(0, 1, 2))
for m, v in r1["results"].items():
    print(f"  {m:14s} foldF1={v['fold_macro_f1_mean']:.4f} CI95={[round(x,3) for x in v['fold_ci95']]} "
          f"pooled={v['pooled_macro_f1_mean']:.4f}±{v['pooled_macro_f1_std']:.4f}", flush=True)

print("=== E-DAIC headline (EXP-4.2) ===", flush=True)
ds, manifest = build_daic(cache_dir="artifacts/cache/daic_egemaps")
cfg = dict(DAIC_CFG); cfg["batch_size"] = 32
r2 = run_daic_headline(ds, manifest, seeds=(0, 1), cfg=cfg)
for m, v in r2["results"].items():
    print(f"  {m:14s} dev={v['dev_macro_f1_mean']:.4f} test={v['test_macro_f1_mean']:.4f}", flush=True)

print("PHASE4_DONE", flush=True)
