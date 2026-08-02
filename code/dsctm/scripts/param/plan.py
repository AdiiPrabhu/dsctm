#!/usr/bin/env python
"""Inspect the campaign plan. Use this to SET the sbatch --array bounds, never to guess.

    python scripts/param/plan.py --summary
    python scripts/param/plan.py --sbatch-array tuning-daicwoz
    python scripts/param/plan.py --list ablation | head -20
    python scripts/param/plan.py --show ablation 17
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dsctm.campaign import FAMILIES, build_plan, get_task, sbatch_array_spec, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--sbatch-array", metavar="FAMILY")
    ap.add_argument("--list", metavar="FAMILY")
    ap.add_argument("--show", nargs=2, metavar=("FAMILY", "INDEX"))
    ap.add_argument("--throttle", type=int, default=4)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    if args.sbatch_array:
        print(sbatch_array_spec(args.sbatch_array, args.throttle))
        return
    if args.list:
        for i, t in enumerate(build_plan(args.list)):
            print(f"{i:4d}  {t.task_id}")
        return
    if args.show:
        t = get_task(args.show[0], int(args.show[1]))
        print(json.dumps(t.to_dict(), indent=2, default=str))
        return

    report = {"families": {}, "total": summarize()}
    print(f"{'family':24s} {'tasks':>6s}  {'--array':>16s}  plan_digest")
    print("-" * 78)
    for name in sorted(FAMILIES):
        s = summarize(name)
        report["families"][name] = s
        print(f"{name:24s} {s['n_tasks']:6d}  {sbatch_array_spec(name, args.throttle):>16s}  "
              f"{s['plan_digest']}")
    print("-" * 78)
    t = report["total"]
    print(f"{'TOTAL':24s} {t['n_tasks']:6d}  {'':>16s}  {t['plan_digest']}")
    print(f"\nby experiment: {t['by_experiment']}")
    print(f"unique task ids: {t['unique_task_ids']} / {t['n_tasks']}")
    if t["unique_task_ids"] != t["n_tasks"]:
        print("ERROR: task id collision — the plan is not uniquely addressable")
        sys.exit(1)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"written: {args.json}")


if __name__ == "__main__":
    main()
