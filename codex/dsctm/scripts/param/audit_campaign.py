#!/usr/bin/env python
"""Fail-closed campaign admission. Nothing reaches a table without passing this.

    python scripts/param/audit_campaign.py --all
    python scripts/param/audit_campaign.py --family ablation --aggregate
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dsctm.campaign import FAMILIES, aggregate_family, audit_family   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report on an incomplete family; NEVER use for a citable result")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os
    root = args.results_root or os.environ.get("DSCTM_RESULTS_ROOT")
    if not root:
        print("FATAL: --results-root or DSCTM_RESULTS_ROOT required", file=sys.stderr)
        return 2

    families = sorted(FAMILIES) if args.all else [args.family]
    if not families or families == [None]:
        print("FATAL: pass --family NAME or --all", file=sys.stderr)
        return 2

    report, any_rejected = {}, False
    for fam in families:
        res = audit_family(fam, root, require_all=not args.allow_partial)
        entry = res.to_dict()
        print(f"\n{'='*72}\nFAMILY {fam}: "
              f"{'ADMITTED' if res.admitted else 'REJECTED'}"
              f"   ({res.n_completed}/{res.n_expected} completed)")
        for e in res.errors:
            print(f"  ERROR  {e}")
        for w in res.warnings:
            print(f"  WARN   {w}")
        if res.admitted:
            print(f"  receipt {res.receipt}")
            if args.aggregate:
                entry["aggregate"] = aggregate_family(fam, root)
                groups = entry["aggregate"]["groups"]
                print(f"  {'group':28s} {'n':>3s} {'macroF1':>9s}  CI95")
                for k, v in sorted(groups.items(),
                                   key=lambda kv: -kv[1]["macro_f1_mean"]):
                    print(f"  {k:28s} {v['n_runs']:3d} {v['macro_f1_mean']:9.4f}  "
                          f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
        else:
            any_rejected = True
        report[fam] = entry

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nwritten: {args.out}")
    print(f"\n{'='*72}")
    print("VERDICT:", "ONE OR MORE FAMILIES REJECTED" if any_rejected else "ALL ADMITTED")
    return 1 if any_rejected else 0


if __name__ == "__main__":
    sys.exit(main())
