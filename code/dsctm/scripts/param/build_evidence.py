#!/usr/bin/env python
"""Gate 12 — generate every final artifact from ADMITTED PARAM runs. Nothing by hand.

    python scripts/param/build_evidence.py --out artifacts/final

Reads ONLY results/param_utkarsh_authoritative/, and only families the fail-closed auditor
admits. Every displayed number resolves to run id, git SHA, config hash, dataset hash,
split hash, seed, prediction file, metric computation and receipt.

If a family is rejected, its numbers are absent from the output rather than downgraded to a
footnote. That is the point of a fail-closed pipeline.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dsctm.campaign import FAMILIES, aggregate_family, audit_family, build_plan  # noqa: E402


def _rows_for_manifest(results_root: Path):
    for fam_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for run in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            sp, mp = run / "status.json", run / "metrics.json"
            if not sp.exists():
                continue
            try:
                status = json.loads(sp.read_text())
            except Exception:
                continue
            metrics = {}
            if mp.exists():
                try:
                    metrics = json.loads(mp.read_text())
                except Exception:
                    pass
            git = {}
            if (run / "git.json").exists():
                try:
                    git = json.loads((run / "git.json").read_text())
                except Exception:
                    pass
            dh = {}
            if (run / "dataset_hashes.json").exists():
                try:
                    dh = json.loads((run / "dataset_hashes.json").read_text())
                except Exception:
                    pass
            task = metrics.get("task", {})
            yield {
                "family": fam_dir.name, "run_id": run.name,
                "experiment_id": task.get("experiment_id"), "dataset": task.get("dataset"),
                "model": task.get("model"), "condition": task.get("condition"),
                "seed": task.get("seed"), "trial": task.get("trial"),
                "protocol": task.get("protocol"), "config_hash": task.get("config_hash"),
                "plan_digest": metrics.get("plan_digest"),
                "dataset_hash": dh.get("data_version_hash"),
                "git_commit": git.get("commit"), "git_dirty": git.get("dirty"),
                "status": status.get("status"),
                "receipt": status.get("receipt_sha256"),
                "contract_complete": status.get("contract", {}).get("complete"),
                "predictions": str((run / "predictions.parquet").relative_to(results_root))
                               if (run / "predictions.parquet").exists() else "",
                "run_dir": str(run.relative_to(results_root)),
            }


def markdown_table(agg: dict) -> str:
    groups = agg.get("groups", {})
    if not groups:
        return "_No admitted runs._\n"
    lines = [f"| {agg['group_by']} | n | macro-F1 | ± std | 95% CI |",
             "|---|---:|---:|---:|---|"]
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["macro_f1_mean"]):
        lines.append(f"| {k} | {v['n_runs']} | {v['macro_f1_mean']:.4f} | "
                     f"{v['macro_f1_std']:.4f} | "
                     f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |")
    comps = agg.get("comparisons", {})
    if comps:
        ref = agg.get("reference")
        lines += ["", f"**Paired comparisons vs `{ref}`** "
                      f"(family: {agg.get('multiplicity_family')})", "",
                  "| other | n pairs | HL shift | rank-biserial | p | Holm p | BH p | reachable |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for k, c in sorted(comps.items()):
            w = c["wilcoxon"]
            lines.append(
                f"| {k} | {c['n_pairs']} | {c['hodges_lehmann_ref_minus_other']:+.4f} | "
                f"{c['rank_biserial']:+.3f} | {w.get('p_value', float('nan')):.4f} | "
                f"{w.get('holm_adjusted_p', float('nan')):.4f} | "
                f"{w.get('bh_adjusted_p', float('nan')):.4f} | "
                f"{'yes' if w.get('significance_reachable') else '**no**'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root",
                    default=os.environ.get("DSCTM_RESULTS_ROOT",
                                           "results/param_utkarsh_authoritative"))
    ap.add_argument("--out", default="artifacts/final")
    args = ap.parse_args()

    root, out = Path(args.results_root), Path(args.out)
    for sub in ("tables", "figures", "receipts"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    admitted, rejected, aggregates = {}, {}, {}
    for fam in sorted(FAMILIES):
        res = audit_family(fam, root)
        (admitted if res.admitted else rejected)[fam] = res.to_dict()
        if res.admitted:
            aggregates[fam] = aggregate_family(fam, root)
            (out / "receipts" / f"{fam}.sha256").write_text(f"{res.receipt}  {fam}\n")

    manifest = list(_rows_for_manifest(root)) if root.exists() else []
    fields = ["family", "run_id", "experiment_id", "dataset", "model", "condition", "seed",
              "trial", "protocol", "config_hash", "plan_digest", "dataset_hash",
              "git_commit", "git_dirty", "status", "receipt", "contract_complete",
              "predictions", "run_dir"]
    with (out / "evidence_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(manifest)

    with (out / "experiment_matrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "planned", "found", "completed", "admitted"])
        for fam in sorted(FAMILIES):
            e = (admitted.get(fam) or rejected.get(fam) or {})
            w.writerow([fam, len(build_plan(fam)), e.get("n_found", 0),
                        e.get("n_completed", 0), fam in admitted])

    for name, blob in (("admitted_runs.csv", admitted), ("rejected_runs.csv", rejected)):
        with (out / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["family", "n_expected", "n_completed", "receipt", "errors"])
            for fam, e in sorted(blob.items()):
                w.writerow([fam, e["n_expected"], e["n_completed"], e.get("receipt", ""),
                            " | ".join(e.get("errors", []))[:800]])

    for fam, agg in aggregates.items():
        (out / "tables" / f"{fam}.md").write_text(markdown_table(agg))
        (out / "tables" / f"{fam}.json").write_text(json.dumps(agg, indent=2, default=str))

    lines = [f"# Final Evidence — generated {time.strftime('%FT%TZ', time.gmtime())}", "",
             f"Source: `{root}` (the only citable root).", "",
             "## Admission", "", "| family | planned | completed | admitted | receipt |",
             "|---|---:|---:|---|---|"]
    for fam in sorted(FAMILIES):
        e = (admitted.get(fam) or rejected.get(fam) or {})
        lines.append(f"| {fam} | {len(build_plan(fam))} | {e.get('n_completed',0)} | "
                     f"{'YES' if fam in admitted else '**NO**'} | "
                     f"`{(e.get('receipt') or '')[:16]}` |")
    if rejected:
        lines += ["", "## Rejected families", ""]
        for fam, e in sorted(rejected.items()):
            lines.append(f"### {fam}")
            for err in e.get("errors", [])[:10]:
                lines.append(f"- {err}")
            lines.append("")
    if not admitted:
        lines += ["", "> **No family was admitted. No number in this directory may be "
                      "cited.** This is the expected state until the PARAM campaign runs.",
                  ""]
    for fam, agg in aggregates.items():
        lines += [f"## {fam}", "", markdown_table(agg)]
    (out / "FINAL_EVIDENCE.md").write_text("\n".join(lines) + "\n")

    print(f"admitted : {sorted(admitted) or 'none'}")
    print(f"rejected : {sorted(rejected) or 'none'}")
    print(f"manifest : {len(manifest)} run(s)")
    print(f"written  : {out}")
    return 0 if admitted else 1


if __name__ == "__main__":
    sys.exit(main())
