#!/usr/bin/env python
"""EXP-4.2 on the manuscript's **88-dim eGeMAPS functionals** (feature-fidelity reproduction).

Same official E-DAIC split / dev-select-test protocol / class-balanced CE / 5 seeds /
participant bootstrap as the 23-dim run — ONLY the input features change (23-dim LLDs ->
88-dim functionals from raw audio). Writes daic_headline_egemaps88.json; does NOT touch the
23-dim daic_headline.json, so the two are directly comparable.

Prereq: PYTHONPATH=src python -u scripts/build_daic_egemaps88.py   (populates the 88-dim cache)
Run:    PYTHONPATH=src python -u scripts/run_exp42_88.py
"""
from pathlib import Path

from dsctm.data.daic import DAIC_ROOT_DEFAULT, build_daic88
from dsctm.experiments.gate1 import run_gate1_daic
from dsctm.experiments.headline import DAIC_CFG, run_daic_headline

_ROOT_CANDIDATES = [
    DAIC_ROOT_DEFAULT,
    "/mnt/adissd/phd/dsctm-resubmission/dataset/daicwoz",
    str(Path(__file__).resolve().parents[3] / "dataset" / "daicwoz"),
]
DAIC_ROOT = next((r for r in _ROOT_CANDIDATES
                  if (Path(r) / "labels" / "train_split.csv").exists()), DAIC_ROOT_DEFAULT)
print(f"E-DAIC root: {DAIC_ROOT}", flush=True)

print("=== EXP-4.2 on 88-dim eGeMAPS functionals (feature-fidelity reproduction) ===", flush=True)
ds, manifest = build_daic88(root=DAIC_ROOT, cache_dir="artifacts/cache/daic_egemaps88")
print(f"assembled: N={ds.N} T={ds.T} F={ds.F} n_classes={ds.n_classes} "
      f"splits={manifest['counts']} skipped={len(manifest['skipped'])}", flush=True)
assert ds.F == 88, f"expected F=88, got {ds.F} — is the 88-dim cache fully built?"

# leakage / provenance re-check on the new feature build (features don't change split
# membership, but re-verify no cross-split participant overlap and record provenance).
try:
    g = run_gate1_daic(ds, manifest)
    print(f"Gate 1 leakage_free={g['leakage']['leakage_free']}", flush=True)
except Exception as e:  # pragma: no cover
    print(f"[warn] Gate 1 check skipped: {e}", flush=True)

cfg = dict(DAIC_CFG)
cfg["batch_size"] = 32
assert cfg.get("class_weight") == "balanced", "expected class-balanced CE"

r = run_daic_headline(ds, manifest, seeds=(0, 1, 2, 3, 4), cfg=cfg,
                      out_name="daic_headline_egemaps88.json")
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
print("EXP42_88_DONE", flush=True)
