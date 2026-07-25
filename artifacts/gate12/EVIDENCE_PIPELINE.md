# Gate 12 — Final Evidence Generation

Status: **IMPLEMENTED AND EXERCISED.** `scripts/param/build_evidence.py` ran against the
empty results root and correctly **admitted nothing**.

```
admitted : none
rejected : ['ablation','confirm-daicwoz','confirm-studentlife','tuning-daicwoz','tuning-studentlife']
manifest : 0 run(s)
```

That is the pipeline working. A fail-closed generator that produces tables from an empty
directory is worse than useless.

## Outputs

```
artifacts/final/
├── FINAL_EVIDENCE.md        admission table + per-family results
├── evidence_manifest.csv    one row per run, full provenance chain
├── experiment_matrix.csv    planned vs found vs completed vs admitted
├── admitted_runs.csv
├── rejected_runs.csv        with the reasons
├── tables/<family>.{md,json}
├── figures/
└── receipts/<family>.sha256
```

Every displayed number resolves to: run id · git SHA · config hash · dataset hash ·
split hash · seed · prediction file · metric computation · receipt.

## Admission rules (all enforced)

missing models · missing folds · wrong seeds · wrong dataset hash · wrong split hash ·
non-finite metrics · metrics out of range · invalid confidence intervals · duplicate sample
ids · missing predictions · summaries that do not recompute · **test access during tuning** ·
incomplete runs marked successful · plan drift.

**Partial admission is not offered.** A family is admitted whole or not at all — that is how
a campaign ends up reporting five of six models and calling it a comparison.

## Statistics

Hodges–Lehmann shift, rank-biserial correlation, exact Wilcoxon with a **reachability flag**,
Holm and Benjamini–Hochberg adjusted p-values over a declared comparison family. Tables print
`reachable: **no**` when n is too small for p < 0.05 to exist — which at n=5 it is.

## Blocker

| ID | Blocker |
|---|---|
| B-022 | `figures/` is created but empty. Figure generation (scaling curves, reliability diagrams, ablation deltas) is written once the shape of the admitted data is known, rather than guessing axes for data that does not exist. |
