#!/usr/bin/env python
"""Gate 6 — receptive fields for every planned dilation schedule, DERIVED not typed.

The manuscript printed 47/383/1535. Those match no standard derivation; they satisfy
6*r_max - 1, which is a formula error. This script exists so the revised paper's numbers
come out of the implementation rather than out of a person.

    python scripts/param/rf_report.py --json artifacts/gate6/receptive_fields.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dsctm.campaign.plan import DILATION_SCHEDULES          # noqa: E402
from dsctm.models.blocks import Branch                       # noqa: E402
from dsctm.models.dmstcn import DMSTCNConfig                 # noqa: E402

MANUSCRIPT_PRINTED = {"ssb": 47, "msb": 383, "lsb": 1535}
SAMPLING = {"studentlife": ("1 min", 60.0), "daicwoz": ("0.5 s", 0.5)}


def span(rf: int, seconds_per_step: float) -> str:
    s = rf * seconds_per_step
    if s < 90:      return f"{s:.0f} s"
    if s < 5400:    return f"{s/60:.1f} min"
    if s < 172800:  return f"{s/3600:.1f} h"
    return f"{s/86400:.1f} d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--kernel", type=int, default=DMSTCNConfig().K)
    args = ap.parse_args()
    K = args.kernel
    rows = []
    for sched_name, branches in DILATION_SCHEDULES.items():
        for branch, dilations in branches.items():
            b = Branch(16, K, dilations)
            rf2, rf1 = b.theoretical_rf_two_conv(), b.theoretical_rf_one_conv()
            assert rf2 == 1 + 2 * (K - 1) * sum(dilations)
            row = {"schedule": sched_name, "branch": branch, "dilations": list(dilations),
                   "sum_dilations": sum(dilations), "kernel_size": K,
                   "rf_two_conv": rf2, "rf_one_conv": rf1,
                   "formula": "R = 1 + 2(K-1)*sum(r_l)"}
            for ds, (label, secs) in SAMPLING.items():
                row[f"span_{ds}"] = span(rf2, secs)
            if sched_name == "original":
                row["manuscript_printed"] = MANUSCRIPT_PRINTED[branch]
                row["manuscript_correct"] = rf2 == MANUSCRIPT_PRINTED[branch]
                row["manuscript_formula_6rmax_minus_1"] = 6 * max(dilations) - 1
            rows.append(row)

    hdr = f"{'schedule':18s} {'br':4s} {'dilations':22s} {'RF':>6s} {'1conv':>6s} {'SL span':>10s} {'DW span':>10s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['schedule']:18s} {r['branch']:4s} {str(r['dilations']):22s} "
              f"{r['rf_two_conv']:6d} {r['rf_one_conv']:6d} "
              f"{r['span_studentlife']:>10s} {r['span_daicwoz']:>10s}")

    print("\nManuscript check (original schedule):")
    for r in rows:
        if r["schedule"] == "original":
            print(f"  {r['branch']}: derived {r['rf_two_conv']:5d} | printed "
                  f"{r['manuscript_printed']:5d} | 6*r_max-1 = "
                  f"{r['manuscript_formula_6rmax_minus_1']:5d} -> printed value is "
                  f"{'CORRECT' if r['manuscript_correct'] else 'WRONG'}")

    report = {"kernel_size": K, "formula": "R = 1 + 2(K-1)*sum(r_l)",
              "note": ("Every value derived from Branch.theoretical_rf_two_conv(). "
                       "The manuscript's 47/383/1535 satisfy 6*r_max-1, which corresponds "
                       "to no standard dilated-TCN derivation."),
              "rows": rows}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
