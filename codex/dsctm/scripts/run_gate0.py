#!/usr/bin/env python
"""Run Gate 0 (implementation-correctness) experiments and write evidence under
artifacts/resubmission/gate0/. Requires no dataset — runs on the single GPU today.

    python scripts/run_gate0.py
"""
import json

from dsctm.experiments.gate0 import run_gate0

if __name__ == "__main__":
    res = run_gate0()
    print(json.dumps(
        {e["experiment"]: e.get("checks", e.get("branch_rf", "ok")) for e in res["experiments"]},
        indent=2, default=str,
    ))
    print("\nEvidence written to artifacts/resubmission/gate0/")
