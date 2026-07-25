#!/usr/bin/env python
"""Re-run ONLY EXP-4.2 (E-DAIC official split) with the imbalance fix.

Fix: class-balanced cross-entropy — class weights computed from the TRAIN labels only
(leakage-safe, sklearn 'balanced' convention) — to counteract the majority-class
collapse seen under plain CE (5/6 models collapsed to the ~0.41 majority baseline).

EXP-4.1 (StudentLife) is UNCHANGED and is NOT re-run here. The prior plain-CE E-DAIC
result is preserved to daic_headline_plainCE.json for an honest before/after comparison;
the new class-balanced result becomes the canonical daic_headline.json.

Run in the background (writes are flushed; per-model partial checkpoints are dumped):

    python -u scripts/run_exp42.py
"""
import shutil
from pathlib import Path

from dsctm.data.daic import DAIC_ROOT_DEFAULT, build_daic
from dsctm.experiments.headline import DAIC_CFG, run_daic_headline

OUT = Path("artifacts/resubmission/phase4")
OUT.mkdir(parents=True, exist_ok=True)

# The E-DAIC labels/splits live under the raw dataset root. The module default points at
# the old /media mount; the disk is now under /mnt. Resolve the first root that has the
# official split files (features themselves are already cached, so root is only needed for
# labels + any cache miss).
_ROOT_CANDIDATES = [
    DAIC_ROOT_DEFAULT,
    "/mnt/adissd/phd/dsctm-resubmission/dataset/daicwoz",
    str(Path(__file__).resolve().parents[3] / "dataset" / "daicwoz"),
]
DAIC_ROOT = next((r for r in _ROOT_CANDIDATES
                  if (Path(r) / "labels" / "train_split.csv").exists()), DAIC_ROOT_DEFAULT)
print(f"E-DAIC root: {DAIC_ROOT}", flush=True)

# preserve the prior plain-CE run (idempotent: only on the first re-run)
prior, preserved = OUT / "daic_headline.json", OUT / "daic_headline_plainCE.json"
if prior.exists() and not preserved.exists():
    shutil.copy2(prior, preserved)
    print(f"preserved prior plain-CE result -> {preserved}", flush=True)

print("=== E-DAIC headline (EXP-4.2) — imbalance-aware: class-balanced CE ===", flush=True)
ds, manifest = build_daic(root=DAIC_ROOT, cache_dir="artifacts/cache/daic_egemaps")
print(f"assembled: N={ds.N} T={ds.T} F={ds.F} n_classes={ds.n_classes} "
      f"splits={manifest['counts']}", flush=True)

# match the original EXP-4.2 protocol exactly except for the loss: batch_size=32.
# 5 seeds so the fixed official test set gets a participant-level bootstrap CI + a paired
# D-MSTCN-vs-baseline test (primary unit = participant; seeds are secondary stability).
# class_weight='balanced' is already set in DAIC_CFG.
cfg = dict(DAIC_CFG)
cfg["batch_size"] = 32
assert cfg.get("class_weight") == "balanced", "expected class-balanced CE for the fix"

r = run_daic_headline(ds, manifest, seeds=(0, 1, 2, 3, 4), cfg=cfg)
print(f"training_loss={r['training_loss']}  primary_unit={r['primary_unit']}", flush=True)
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
print("EXP42_DONE", flush=True)
