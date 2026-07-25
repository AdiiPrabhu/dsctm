#!/usr/bin/env python
"""Generate reliability tables/plot from a Phase-4 JSON containing saved probabilities.

Probabilities are averaged across seeds for each fixed test participant. The plot is
descriptive; no calibration is fitted and no threshold is selected on test data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def reliability_rows(y, prob, n_bins=10):
    conf = prob.max(1)
    pred = prob.argmax(1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        take = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if take.any():
            rows.append({"bin_lower": lo, "bin_upper": hi, "n": int(take.sum()),
                         "mean_confidence": float(conf[take].mean()),
                         "empirical_accuracy": float(correct[take].mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--out-dir", default="artifacts/resubmission/figures")
    args = ap.parse_args()
    data = json.loads(Path(args.result_json).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for name, result in data["results"].items():
        probs = np.asarray(result["per_seed_test_probabilities"], dtype=float).mean(0)
        # Test labels are reconstructed from the first seed's confusion-independent
        # probability array only when the artifact provides explicit labels.
        y = np.asarray(data["test_true"], dtype=int)
        rows = reliability_rows(y, probs)
        csv_path = out / f"{Path(args.result_json).stem}_{name}_reliability.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        ax.plot([r["mean_confidence"] for r in rows],
                [r["empirical_accuracy"] for r in rows], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1),
           title="DAIC-WOZ test reliability (seed-mean probabilities)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"{Path(args.result_json).stem}_reliability.png", dpi=200)


if __name__ == "__main__":
    main()
