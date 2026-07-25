#!/usr/bin/env python
"""Audit the final corrected EXP-4.1 JSON and write a machine-readable receipt."""
import json
from pathlib import Path

from dsctm.experiments.result_audit import audit_studentlife_headline

SOURCE = Path("artifacts/resubmission/phase4/studentlife_headline_corrected.json")
OUTPUT = Path("artifacts/resubmission/phase4/studentlife_headline_corrected_audit.json")

audit = audit_studentlife_headline(
    SOURCE, expected_split_hash="6208d08f0b8db52b",
    expected_data_hash="a9cbaa3a22c2bf4e", expected_seeds=(0, 1, 2),
)
OUTPUT.write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2), flush=True)
print("EXP41_CORRECTED_AUDIT_DONE", flush=True)
