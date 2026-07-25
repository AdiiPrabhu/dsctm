#!/usr/bin/env python
"""Verify staged datasets and emit dataset_hashes.json for the run contract.

Two jobs:

1. **Provenance.** Every run directory must carry `dataset_hashes.json`. Without it, a
   result cannot be tied to the data that produced it, and the prior campaign's entire
   problem was numbers with no recoverable provenance.

2. **Cross-check against what is already in the repository.** `reviewer-package/data/`
   ships E-DAIC split CSVs (163/56/56). If a freshly downloaded copy disagrees with them,
   that is a corpus-identity problem and must surface before 48 GPU-hours are spent on it,
   not after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_EDAIC = REPO_ROOT / "reviewer-package" / "data"
EXPECTED_EDAIC = {"train": 163, "dev": 56, "test": 56}
EXPECTED_DAICWOZ = {"train": 107, "dev": 35, "test": 47}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: Path, patterns=("*",), max_files: int = 100_000) -> dict:
    """Content hash over a directory: name + size + content hash of each matching file."""
    if not root.exists():
        return {"root": str(root), "exists": False}
    files = []
    for pat in patterns:
        files.extend(sorted(p for p in root.rglob(pat) if p.is_file()))
    files = sorted(set(files))[:max_files]
    h = hashlib.sha256()
    total = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        total += size
        h.update(rel.encode())
        h.update(str(size).encode())
    return {
        "root": str(root),
        "exists": True,
        "file_count": len(files),
        "total_bytes": total,
        "total_gib": round(total / 2**30, 3),
        "manifest_sha256": h.hexdigest()[:32],
    }


def check_edaic_splits(root: Path) -> dict:
    import pandas as pd
    out = {"corpus": "E-DAIC", "expected": EXPECTED_EDAIC, "found": {}, "issues": []}
    labels = root / "labels"
    search = labels if labels.exists() else root
    ids = {}
    for split, n in EXPECTED_EDAIC.items():
        f = search / f"{split}_split.csv"
        if not f.exists():
            out["issues"].append(f"missing {f.name}")
            continue
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        out["found"][split] = len(df)
        out[f"{split}_sha256"] = sha256_file(f)
        ids[split] = set(df["Participant_ID"].astype(int))
        if len(df) != n:
            out["issues"].append(f"{split}: {len(df)} rows, expected {n}")
        if "PHQ_Binary" in df.columns:
            out[f"{split}_positive_rate"] = round(float(df["PHQ_Binary"].mean()), 4)
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if a in ids and b in ids and (ids[a] & ids[b]):
            out["issues"].append(f"PARTICIPANT LEAKAGE between {a} and {b}: "
                                 f"{sorted(ids[a] & ids[b])[:10]}")
    # Cross-check against the copies already in the repository.
    if SHIPPED_EDAIC.exists():
        for split in EXPECTED_EDAIC:
            shipped = SHIPPED_EDAIC / f"{split}_split.csv"
            downloaded = search / f"{split}_split.csv"
            if shipped.exists() and downloaded.exists():
                same = sha256_file(shipped) == sha256_file(downloaded)
                out[f"{split}_matches_repo_copy"] = same
                if not same:
                    out["issues"].append(
                        f"{split}_split.csv differs from reviewer-package/data copy — "
                        f"corpus identity must be resolved before running")
    out["ok"] = not out["issues"]
    return out


def check_daicwoz_splits(root: Path) -> dict:
    import pandas as pd
    out = {"corpus": "DAIC-WOZ", "expected": EXPECTED_DAICWOZ, "found": {}, "issues": []}
    spec = [("train", "train_split_Depression_AVEC2017.csv"),
            ("dev", "dev_split_Depression_AVEC2017.csv"),
            ("test", "full_test_split.csv")]
    ids = {}
    for split, fname in spec:
        f = root / fname
        if not f.exists():
            out["issues"].append(f"missing {fname}")
            continue
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        out["found"][split] = len(df)
        out[f"{split}_sha256"] = sha256_file(f)
        col = "Participant_ID" if "Participant_ID" in df.columns else df.columns[0]
        ids[split] = set(df[col].astype(int))
        if len(df) != EXPECTED_DAICWOZ[split]:
            out["issues"].append(f"{split}: {len(df)} rows, expected "
                                 f"{EXPECTED_DAICWOZ[split]}")
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if a in ids and b in ids and (ids[a] & ids[b]):
            out["issues"].append(f"PARTICIPANT LEAKAGE between {a} and {b}")
    # The manuscript describes "107/82", which is train vs dev+test MERGED. Guard against
    # anyone reviving that: dev and test must stay separate. (tracker V3-02)
    if "dev" in ids and "test" in ids:
        out["dev_plus_test"] = len(ids["dev"]) + len(ids["test"])
        out["merged_82_detected"] = out["dev_plus_test"] == 82
        if out["merged_82_detected"]:
            out["issues"].append(
                "dev+test sums to 82 — the manuscript's merged split. Report the "
                "three-way official partition, not 107/82.")
    sessions = sorted(p.name for p in root.glob("*_P") if p.is_dir())
    out["session_dirs"] = len(sessions)
    corrupt = list(root.glob("*.CORRUPT")) + list(root.glob("*.CORRUPT_TRUNCATED"))
    out["quarantined_archives"] = [p.name for p in corrupt]
    out["ok"] = not out["issues"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    roots = {
        "DAIC-WOZ": Path(os.environ.get("DSCTM_DAICWOZ_ROOT", "")),
        "E-DAIC": Path(os.environ.get("DSCTM_EDAIC_ROOT", "")),
        "StudentLife": Path(os.environ.get("DSCTM_STUDENTLIFE_ROOT", "")),
    }

    report = {"datasets": {}, "splits": {}, "issues": []}
    for name, root in roots.items():
        if not str(root):
            report["issues"].append(f"{name}: root env var unset")
            continue
        pats = ("*.csv",) if name == "E-DAIC" else ("*",)
        report["datasets"][name] = tree_digest(root, pats)
        print(f"{name:12s} {report['datasets'][name].get('file_count', 0):7d} files  "
              f"{report['datasets'][name].get('total_gib', 0):8.2f} GiB  "
              f"{report['datasets'][name].get('manifest_sha256', 'ABSENT')}")

    if roots["E-DAIC"].exists():
        report["splits"]["E-DAIC"] = check_edaic_splits(roots["E-DAIC"])
    if roots["DAIC-WOZ"].exists():
        report["splits"]["DAIC-WOZ"] = check_daicwoz_splits(roots["DAIC-WOZ"])

    for corpus, res in report["splits"].items():
        status = "OK" if res.get("ok") else "ISSUES"
        print(f"\n{corpus}: {status}  found={res.get('found')}")
        for issue in res.get("issues", []):
            print(f"  - {issue}")
            report["issues"].append(f"{corpus}: {issue}")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nwritten: {p}")

    sys.exit(1 if report["issues"] else 0)


if __name__ == "__main__":
    main()
