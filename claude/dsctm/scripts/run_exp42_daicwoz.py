#!/usr/bin/env python
"""EXP-4.2 on the classic DAIC-WOZ (AVEC2017) — the corpus the manuscript actually cites.

88-dim eGeMAPS functionals (from raw audio), official 107/35/47 train / dev-select / test
(NO dev+test merge — answers the reviewer's 107/82 objection), class-balanced CE, 5 seeds,
participant bootstrap. Same protocol as the E-DAIC runs, so results are directly comparable.
Writes daicwoz_headline_egemaps88.json.

Prereq: PYTHONPATH=src python -u scripts/build_daicwoz_egemaps88.py
Run:    PYTHONPATH=src python -u scripts/run_exp42_daicwoz.py
"""
from pathlib import Path

from dsctm.data.daic import DAICWOZ_ROOT_DEFAULT, build_daicwoz88
from dsctm.experiments.gate1 import run_gate1_daic
from dsctm.experiments.headline import DAIC_CFG, run_daic_headline

_ROOT_CANDIDATES = [
    DAICWOZ_ROOT_DEFAULT,
    "/mnt/adissd/phd/dsctm-resubmission/dataset/DAIC-WOZ",
    str(Path(__file__).resolve().parents[3] / "dataset" / "DAIC-WOZ"),
]
root = next((r for r in _ROOT_CANDIDATES
             if (Path(r) / "train_split_Depression_AVEC2017.csv").exists()), DAICWOZ_ROOT_DEFAULT)
print(f"DAIC-WOZ root: {root}", flush=True)

print("=== EXP-4.2 on classic DAIC-WOZ (AVEC2017), 88-dim eGeMAPS functionals ===", flush=True)
ds, manifest = build_daicwoz88(root=root)
print(f"assembled: N={ds.N} T={ds.T} F={ds.F} n_classes={ds.n_classes} "
      f"splits={manifest['counts']} skipped={len(manifest['skipped'])}", flush=True)
assert ds.F == 88, f"expected F=88, got {ds.F} — is the DAIC-WOZ 88-dim cache fully built?"
import numpy as np
print("class balance per split:",
      {sp: np.bincount(ds.y[[i for i in range(ds.N)
                             if manifest['split_of_subject'].get(str(ds.subject_id[i])) == sp]],
                       minlength=2).tolist() for sp in ("train", "dev", "test")}, flush=True)

try:
    g = run_gate1_daic(ds, manifest)
    print(f"Gate 1 leakage_free={g['leakage']['leakage_free']}", flush=True)
except Exception as e:  # pragma: no cover
    print(f"[warn] Gate 1 check skipped: {e}", flush=True)

cfg = dict(DAIC_CFG)
cfg["batch_size"] = 32
assert cfg.get("class_weight") == "balanced", "expected class-balanced CE"

r = run_daic_headline(ds, manifest, seeds=(0, 1, 2, 3, 4), cfg=cfg,
                      out_name="daicwoz_headline_egemaps88.json")
print(f"training_loss={r['training_loss']}", flush=True)
for m, v in r["results"].items():
    b = v["test_macro_f1_participant_boot"]
    print(f"  {m:14s} dev={v['dev_macro_f1_seed_mean']:.4f}±{v['dev_macro_f1_seed_std']:.4f} "
          f"test={v['test_macro_f1_seed_mean']:.4f}±{v['test_macro_f1_seed_std']:.4f} "
          f"boot95=[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}] "
          f"bacc={v['test_balanced_acc_seed_mean']:.4f}", flush=True)
print("--- D-MSTCN vs baselines (paired participant bootstrap of test macro-F1 diff) ---",
      flush=True)
for m, c in r["dmstcn_vs_baselines"].items():
    p = c["primary_participant_paired_bootstrap"]
    print(f"  vs {m:12s} Δ={p['test_macro_f1_diff']:+.4f} "
          f"CI95=[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] "
          f"P(D-MSTCN better)={p['prob_dmstcn_better']:.3f}", flush=True)
print("EXP42_WOZ_DONE", flush=True)
