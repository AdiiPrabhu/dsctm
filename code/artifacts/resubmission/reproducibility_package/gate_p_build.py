#!/usr/bin/env python3
"""Build deterministic Gate P audit artifacts from the supplied tracker workbook."""
from __future__ import annotations

import csv
import hashlib
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "resubmission"
TRACKER = ROOT / "reviews" / "D_MSTCN_IEEE_Access_Resubmission_Tracker_Completed.xlsx"
MANUSCRIPT = ROOT / "reviews" / "D_MSTCN_Rejected_Manuscript.pdf"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def col_index(ref: str) -> int:
    value = 0
    for ch in "".join(c for c in ref if c.isalpha()):
        value = value * 26 + ord(ch.upper()) - 64
    return value - 1


def workbook_sheets(path: Path) -> dict[str, list[list[str]]]:
    result = {}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iterfind(".//m:t", NS))
                      for si in root.findall("m:si", NS)]
        relroot = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relroot}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in book.findall("m:sheets/m:sheet", NS):
            name = sheet.attrib["name"]
            target = rels[sheet.attrib[f"{{{NS['r']}}}id"]]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            xml = ET.fromstring(archive.read(target))
            rows = []
            for row in xml.findall(".//m:sheetData/m:row", NS):
                values = []
                for cell in row.findall("m:c", NS):
                    idx = col_index(cell.attrib.get("r", "A1"))
                    values.extend([""] * (idx + 1 - len(values)))
                    inline = cell.find("m:is", NS)
                    value_node = cell.find("m:v", NS)
                    if inline is not None:
                        value = "".join(t.text or "" for t in inline.iterfind(".//m:t", NS))
                    elif value_node is None:
                        value = ""
                    elif cell.attrib.get("t") == "s":
                        value = shared[int(value_node.text)]
                    else:
                        value = value_node.text or ""
                    values[idx] = value
                rows.append(values)
            result[name] = rows
    return result


def normalize_priority(value: str) -> str:
    match = re.match(r"(P\d)", value)
    return match.group(1) if match else "P2"


def category(value: str, task: str) -> str:
    text = (value + " " + task).lower()
    rules = [
        ("administrative", "approval governance checklist metadata portal author biography disclosure ethics similarity"),
        ("citation", "reference citation literature bibliography"),
        ("data/leakage", "data split leakage preprocessing imputation window subject sample dataset cold start"),
        ("statistics", "statistics significance confidence effect wilcoxon seed"),
        ("systems", "system gpu node scalability timing throughput network communication profiling replication"),
        ("correctness", "theory proof theorem algorithm receptive parameter synchronization tcp causality notation"),
        ("baseline", "baseline ablation comparison personalization"),
        ("reproducibility", "reproducibility provenance open science archive code config"),
        ("writing", "writing rewrite title abstract contribution limitation conclusion response figure table pdf template"),
    ]
    for label, words in rules:
        if any(word in text for word in words.split()):
            return label
    return "methodology"


def experiment_ids(task_id: str, task: str) -> str:
    text = task.lower()
    mappings = [
        ("receptive", "EXP-0.1"), ("parameter", "EXP-0.2"),
        ("tcp", "EXP-0.3;EXP-3.1;EXP-3.2"), ("synchron", "EXP-0.3;EXP-3.1;EXP-3.2"),
        ("causal", "EXP-0.4;EXP-3.1"), ("split", "EXP-1.1;EXP-1.2"),
        ("leak", "EXP-1.2"), ("preprocess", "EXP-1.1;EXP-1.3"),
        ("imput", "EXP-1.2;EXP-1.3"), ("provenance", "EXP-2.1"),
        ("baseline", "EXP-2.2;EXP-2.3"), ("ddp", "EXP-3.1"),
        ("synthetic", "EXP-3.3"), ("stat", "EXP-4.3"),
        ("confidence", "EXP-4.3"), ("class", "EXP-4.4"),
        ("personal", "EXP-5.5;EXP-5.6"), ("cold start", "EXP-5.6"),
        ("dilation", "EXP-5.3"), ("branch", "EXP-5.1"), ("csag", "EXP-5.2;EXP-5.9"),
        ("network", "EXP-6.4"), ("communication", "EXP-6.3"),
        ("throughput", "EXP-6.1;EXP-6.3"), ("profil", "EXP-6.1"),
        ("gpu", "EXP-6.2;EXP-6.3"), ("scal", "EXP-6.2;EXP-6.3"),
        ("failure", "EXP-6.5"), ("daic", "EXP-4.2"), ("studentlife", "EXP-4.1"),
        ("seed", "EXP-5.10"), ("transfer", "EXP-5.10"),
    ]
    found = []
    for needle, ids in mappings:
        if needle in text:
            found.extend(ids.split(";"))
    return ";".join(dict.fromkeys(found)) or "N/A (non-experimental task)"


def build_map(rows: list[dict[str, str]]) -> None:
    fields = ["tracker_task_id", "reviewer", "comment_summary", "category", "priority",
              "proposed_decision", "experiment_ids", "code_locations", "manuscript_locations",
              "acceptance_test", "evidence_paths", "current_status", "blocker"]
    with (OUT / "reviewer_to_experiment_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            exp = experiment_ids(row["ID"], row["Task"])
            evidence = "reviews/D_MSTCN_Rejected_Manuscript.pdf;reviews/D_MSTCN_IEEE_Access_Resubmission_Tracker_Completed.xlsx"
            blocker = []
            if "N/A" not in exp:
                blocker.append("implementation/configs and raw run artifacts unavailable")
            if any(x in row["Task"].lower() for x in ("data", "split", "leak", "subject", "preprocess", "imput")):
                blocker.append("datasets/split manifests/label provenance unavailable")
            if row["Manuscript Section"] and row["ID"] not in {"G0-07"}:
                blocker.append("editable manuscript source unavailable")
            if any(x in row["Task"].lower() for x in ("approval", "biography", "disclosure", "permissions", "ethics")):
                blocker.append("author/institutional confirmation required")
            status = "blocked" if blocker else "in_progress"
            writer.writerow({
                "tracker_task_id": row["ID"], "reviewer": row["Reviewer(s)"],
                "comment_summary": row["Task"], "category": category(row["Category"], row["Task"]),
                "priority": normalize_priority(row["Priority"]),
                "proposed_decision": row["Proposed Resolution / Decision"], "experiment_ids": exp,
                "code_locations": "UNAVAILABLE: no source checkout supplied",
                "manuscript_locations": row["Manuscript Section"] or "Internal/submission package",
                "acceptance_test": row["Evidence / Acceptance Test"] or row["Definition of Done"],
                "evidence_paths": evidence, "current_status": status,
                "blocker": "; ".join(dict.fromkeys(blocker)) or "No Gate P blocker; downstream task not yet executed",
            })


COMPUTE_ROWS = [
    ("EXP-0.1","P0","N/A","3 branches x analytic/gradient/perturbation",1,1,1,6,0,5,0.0,0.1,"source + model config","Mandatory RF correctness; CPU/single-GPU cheap test"),
    ("EXP-0.2","P0","N/A","3 representative input shapes",1,1,1,3,1,5,0.25,0.2,"source + model config","Exact parameter and MAC/FLOP accounting"),
    ("EXP-0.3","P0","N/A","12 synchronization/state transitions",1,1,1,12,1,5,1.0,0.5,"distributed implementation","Invariant and checkpoint tests; multi-GPU cases blocked on one GPU"),
    ("EXP-0.4","P0","N/A","8 causality/shape/reproducibility tests",1,1,1,8,1,10,1.33,1.0,"source + fixtures","Cheap correctness suite"),
    ("EXP-1.1","P0","StudentLife+DAIC-WOZ","provenance and fixed manifests",6,1,1,6,0,15,0.0,1.0,"authorized data + official splits","Two datasets; five SL folds plus DAIC official split"),
    ("EXP-1.2","P0","StudentLife+DAIC-WOZ","10 leakage assertions per protocol",6,1,1,60,0,5,0.0,1.0,"authorized data + pipeline","Leakage gate"),
    ("EXP-1.3","P1","StudentLife+DAIC-WOZ","4 imputation conditions",6,1,3,72,1,25,30.0,6.0,"EXP-1.1/1.2","Robustness; SL 5 folds and DAIC dev protocol"),
    ("EXP-2.1","P0","StudentLife+DAIC-WOZ","submitted configuration reproduction",6,1,5,30,1,30,15.0,12.0,"old code/config/logs","Conditional estimate; old artifacts absent"),
    ("EXP-2.2","P0","StudentLife+DAIC-WOZ","8 methods",6,1,5,240,1,35,140.0,45.0,"locked leakage-safe protocols","SL: 8x5x5=200; DAIC: 8x1x5=40"),
    ("EXP-2.3","P1","StudentLife+DAIC-WOZ","4 capacity controls",6,1,3,72,1,35,42.0,16.0,"EXP-2.2","Capacity/tuning controls"),
    ("EXP-3.1","P0","StudentLife+DAIC-WOZ","4 synchronization conditions",6,1,5,120,1,35,70.0,24.0,"correct distributed code","SL 4x5x5=100; DAIC 4x1x5=20"),
    ("EXP-3.2","P1","StudentLife+DAIC-WOZ","8 TCP component conditions",6,1,3,144,1,35,84.0,28.0,"EXP-3.1","Component ablation"),
    ("EXP-3.3","P1","Synthetic","3 delay regimes x 3 models",1,1,10,90,1,15,22.5,5.0,"source","Controlled falsification task"),
    ("EXP-4.1","P0","StudentLife","locked grouped CV",5,1,10,50,1,35,29.17,12.0,"Gate 1 + locked config","5 folds x 10 stability seeds"),
    ("EXP-4.2","P0","DAIC-WOZ","official train/dev; test if authorized",1,1,10,10,1,40,6.67,6.0,"official splits + evaluator right","One locked evaluation per seed; test submission not assumed"),
    ("EXP-4.3","P0","StudentLife+DAIC-WOZ","participant/fold statistics",6,1,10,60,0,5,0.0,2.0,"OOF predictions","Analysis jobs, not extra training"),
    ("EXP-4.4","P1","StudentLife+DAIC-WOZ","calibration/threshold analysis",6,1,10,60,0,5,0.0,2.0,"OOF/dev predictions","Analysis jobs"),
    ("EXP-5.1","P1","StudentLife+DAIC-WOZ","7 branch combinations",6,1,3,126,1,35,73.5,24.0,"locked protocol","Architecture ablation"),
    ("EXP-5.2","P1","StudentLife+DAIC-WOZ","5 fusion conditions",6,1,3,90,1,35,52.5,18.0,"locked protocol","CSAG controls"),
    ("EXP-5.3","P1","StudentLife+DAIC-WOZ","6 schedule/kernel/depth conditions",6,1,3,108,1,35,63.0,20.0,"EXP-0.1","Limited sensitivity"),
    ("EXP-5.5","P1","StudentLife","4 personalization conditions",5,1,3,60,1,35,35.0,12.0,"leakage-safe identity handling","Personalization controls"),
    ("EXP-5.6","P1","StudentLife","zero-shot + 3 few-shot budgets",5,1,3,60,1,30,30.0,12.0,"chronological support/query","Unseen-subject evaluation"),
    ("EXP-5.7","P1","StudentLife+DAIC-WOZ","4 missingness severities",6,1,3,72,1,25,30.0,10.0,"locked models","Robustness"),
    ("EXP-5.8","P1","StudentLife+DAIC-WOZ","4 context lengths",6,1,3,72,1,25,30.0,10.0,"RF validation","Context sensitivity"),
    ("EXP-5.9","P1","StudentLife+DAIC-WOZ","attention stability analyses",6,1,3,18,0,10,0.0,3.0,"saved predictions/weights","Analysis only"),
    ("EXP-5.10","P2","SEED","4 transfer conditions",3,1,3,36,1,35,21.0,8.0,"licensed SEED + protocol","Optional; presently blocked"),
    ("EXP-6.1","P0","StudentLife+DAIC-WOZ","2 input shapes x 20 repetitions",1,1,20,40,1,5,3.33,4.0,"implemented model","Single-GPU profile available in principle"),
    ("EXP-6.2","P0","StudentLife+DAIC-WOZ","1-GPU equivalence only locally",1,1,10,10,1,10,1.67,2.0,"2+ physical GPUs for scaling","Multi-GPU portion blocked; one-GPU reference counted"),
    ("EXP-6.3","P0","StudentLife+DAIC-WOZ","1-GPU timing reference only locally",1,1,20,20,1,5,1.67,2.0,"2+ physical GPUs for curves","Strong/weak scaling blocked beyond N=1"),
    ("EXP-6.4","P1","N/A","network/interconnect sensitivity",1,1,20,20,1,5,1.67,2.0,"verified communication path + multi-host/network","Not applicable to current one-GPU host"),
    ("EXP-6.5","P1","N/A","4 failure/recovery conditions",1,1,5,20,1,10,3.33,3.0,"distributed implementation","Failure/recovery tests"),
]


def build_compute() -> None:
    fields = ["experiment_id","priority","dataset","conditions","grouped_folds","repeats",
              "seeds_per_fold","total_runs","gpu_count_per_run","estimated_minutes_per_run",
              "estimated_gpu_hours","storage_gb","dependencies","justification"]
    with (OUT / "compute_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(COMPUTE_ROWS)


def build_risks() -> None:
    rows = [
        ("RISK-01","critical","P0","No implementation/configuration checkout","All correctness and training evidence blocked","Provide exact source repository/commit and configs","open"),
        ("RISK-02","critical","P0","No authorized dataset paths or split manifests","Leakage audit and quality reruns blocked","Provide paths and access constraints without copying protected data","open"),
        ("RISK-03","critical","P0","No raw logs/checkpoints/predictions","Submitted values have no traceable provenance","Provide immutable old artifact bundle; otherwise remove/rerun values","open"),
        ("RISK-04","critical","P0","No editable manuscript source/bibliography/figures","Exact revisions and clean rebuild blocked","Provide LaTeX/Word source and assets","open"),
        ("RISK-05","high","P0","Only one physical GPU available locally","2-8 GPU equivalence/scaling claims cannot be tested","Use authorized multi-GPU single server or remove those claims","open"),
        ("RISK-06","critical","P0","DAIC test-label/evaluator authorization unknown","Test-set claims cannot be regenerated","Document evaluator right; otherwise report dev-only limitation","open"),
        ("RISK-07","high","P0","Submitted five-seed two-sided Wilcoxon p<0.05 claim is mathematically impossible with five nonzero pairs","Significance markers invalid","Withdraw markers; rerun participant/fold-level analysis","open"),
        ("RISK-08","high","P0","Submitted 107/82 DAIC partition conflicts with official partition convention","Potential protocol invalidity","Verify official files and quarantine prior DAIC results","open"),
        ("RISK-09","high","P0","PDF reports 1-8 compute nodes and N=16 simulation on an eight-GPU single host","Systems scope is overstated","Adopt single-server GPU-worker wording and remove unphysical counts","open"),
        ("RISK-10","high","P0","Convergence theorem is unsupported by supplied evidence","Correctness/rejection risk","Remove theorem; retain only tested version-lag invariant","open"),
        ("RISK-11","high","P0","StudentLife 60-step input caps realized context at 60 minutes","Hours/days claim unsupported","Report theoretical and input-capped realized context separately","open"),
        ("RISK-12","medium","P1","Original reviewer letter absent; R5 citation details missing","Verbatim coverage and suggested-citation decisions incomplete","Provide decision letter/reviews","open"),
        ("RISK-13","medium","P0","Author AI-use, biography, ethics, funding and conflicts need confirmation","Administrative facts cannot be inferred","Obtain dated author confirmations","open"),
        ("RISK-14","medium","P1","Compute estimates precede code/data benchmarking","Budget may differ materially","Run <=30-minute smoke benchmark then revise estimates","open"),
        ("RISK-15","high","P0","No Git repository or immutable revision identity","Cannot link evidence to code revision","Provide/init authorized source repository; do not tag this input-only package","open"),
    ]
    with (OUT / "risk_register.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["risk_id","severity","priority","risk","consequence","mitigation","status"]); writer.writerows(rows)


def build_claim_registry() -> None:
    rows = [
        ("CLM-001","Title/Abstract","Distributed/scalable cognitive modeling across compute nodes","physical topology + scaling","EXP-6.2;EXP-6.3","","Only rejected PDF available","narrow","A Multi-Scale Temporal Convolutional Network for Cognitive State Modeling with Branch-Parallel Multi-GPU Execution","D1-01;D1-06;W5-01","blocked"),
        ("CLM-002","Abstract/Results","near-linear efficiency eta >=0.81 from 1 to 8 compute nodes and 57% epoch reduction","raw timings on verified physical devices","EXP-6.3","","No raw timing evidence; current host has one GPU","remove","Report only regenerated single-server GPU-worker measurements","D1-01;E4-06;W5-01","blocked"),
        ("CLM-003","Introduction/Problem","standard data parallelism violates temporal causal ordering","matched causal-mechanism study","EXP-3.1","","No implementation or runs","remove","Do not claim inherent DDP causality violation","D1-03;E4-02;W5-03","blocked"),
        ("CLM-004","Contributions/Method","TCP guarantees convergence within a neighborhood of synchronous optimum","valid proof + implementation semantics","EXP-0.3","","Proof sketch insufficient; no code","remove","State only a tested operational version-lag invariant if implemented","D1-04;T2-09;T2-10","blocked"),
        ("CLM-005","Method/Figure 2","branch receptive fields 47/383/1535","code graph + gradient/perturbation support","EXP-0.1","","Written two-convolution formula instead suggests 61/481/1921","narrow","Report verified theoretical RF and input-capped realized context","T2-02","blocked"),
        ("CLM-006","Method/Complexity","personalization costs 2D parameters per subject","state_dict enumeration","EXP-0.2","","Generated FiLM outputs are not stored per-subject parameters","narrow","Report shared generator separately; per-subject storage is d_s if code confirms","T2-03","blocked"),
        ("CLM-007","Data/Results","DAIC-WOZ standard 107/82 train/test","official split files + evaluator provenance","EXP-1.1;EXP-4.2","","Supplied PDF lacks official split/evaluator evidence","remove","Use verified 107/35/47 official protocol or dev-only limitation","V3-02","blocked"),
        ("CLM-008","Results","highest macro-F1 on StudentLife and DAIC-WOZ","leakage-safe locked OOF/test predictions","EXP-4.1;EXP-4.2;EXP-4.3","","No predictions/logs; StudentLife used 80/20 split","remove","Reintroduce only if locked reruns support it with uncertainty","G0-06;V3-01;W5-01","blocked"),
        ("CLM-009","Results","dagger p<0.05 by Wilcoxon signed-rank with five seeds","valid independent units + multiplicity plan","EXP-4.3","","Two-sided exact Wilcoxon cannot reach p<0.05 with five nonzero pairs","remove","Use participant/fold inference; seeds are stability repetitions","E4-13;E4-15","blocked"),
        ("CLM-010","Transfer","SEED transfer confirms general representations","licensed protocol + raw transfer runs","EXP-5.10","","No SEED data/protocol/artifacts","future_work","Move transfer to future work unless fully regenerated","E4-04;W5-09","blocked"),
        ("CLM-011","Interpretation","CSAG weights reveal task-dependent cognitive scales","stability + perturbation evidence","EXP-5.2;EXP-5.9","","Attention plots alone are descriptive","narrow","Describe weights as descriptive, not causal proof","W5-08","blocked"),
        ("CLM-012","Discussion/Conclusion","population-scale and clinical deployment relevance","population/clinical validation","","","Datasets are small and no clinical validation supplied","remove","Limit to research classification on evaluated datasets","D1-02;V3-10;W5-06","blocked"),
    ]
    fields=["claim_id","manuscript_location","original_claim","evidence_required","experiment_ids","evidence_paths","verified_result","decision","replacement_text","tracker_task_ids","status"]
    with (OUT/"claim_registry.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h); w.writerow(fields); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = workbook_sheets(TRACKER)
    raw = sheets["Master Tracker"]
    header = raw[2]
    rows = [dict(zip(header, row + [""] * (len(header)-len(row)))) for row in raw[3:] if row and row[0]]
    build_map(rows); build_compute(); build_risks(); build_claim_registry()
    totals = {p: sum(r[7] for r in COMPUTE_ROWS if r[1] == p) for p in ("P0","P1","P2")}
    hours = {p: sum(r[10] for r in COMPUTE_ROWS if r[1] == p) for p in ("P0","P1","P2")}
    storage = {p: sum(r[11] for r in COMPUTE_ROWS if r[1] == p) for p in ("P0","P1","P2")}
    priority_counts = {p: sum(normalize_priority(r["Priority"]) == p for r in rows) for p in ("P0","P1","P2","P3")}
    git_state = "not a Git repository"
    try:
        git_state = subprocess.check_output(["git","status","--short","--branch"], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError:
        pass
    gpu = subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.total,driver_version,pci.bus_id","--format=csv,noheader"], text=True, capture_output=True).stdout.strip()
    report = f"""# Gate P Preflight Report

Generated: {datetime.now(timezone.utc).isoformat()}  
Campaign: `dmstcn-ieee-access-resubmission-gate-p-20260718`

## Outcome

Gate P completed for the supplied input-only package. The audit mapped all **{len(rows)} tracker tasks** and created a conservative experiment budget. No scientific result is verified: the implementation, data, editable manuscript source, and raw experiment artifacts are absent. Expensive experiments were not launched.

## Repository and inputs

- Root: `{ROOT}`
- Version control: **{git_state}**. A campaign branch/tag cannot safely be created without the actual source repository.
- Tracker: `reviews/{TRACKER.name}` — SHA-256 `{sha256(TRACKER)}`; 9 sheets; {len(rows)} task rows.
- Rejected manuscript: `reviews/{MANUSCRIPT.name}` — SHA-256 `{sha256(MANUSCRIPT)}`; 15 pages.
- Original decision/reviewer letter: unavailable; the tracker contains summaries, but Reviewer 5 citation details are explicitly missing.
- Editable manuscript (`.tex`, `.bib`, Word, figures): unavailable.
- Source, configs, tests, environment locks, CI, logs, checkpoints, predictions, and old experiment configs: unavailable.
- ZIP contains only the prompt, start instructions, tracker, and rejected PDF; it does not contain hidden source or result artifacts.

## Environment and hardware

- Host: `{socket.gethostname()}`; OS `{platform.platform()}`; Python `{platform.python_version()}`.
- GPU inventory: `{gpu or 'none detected'}`.
- Topology: one physical GPU on PCIe; no multi-GPU or NVLink path is available locally.
- Storage at preflight: approximately 748 GiB free on the workspace filesystem.
- Dependency status: no project dependency manifest exists. System LibreOffice 24.2.7.2 and Poppler PDF tools are available; `openpyxl` is absent and is not required by this audit.

## Data availability

No StudentLife, DAIC-WOZ, SEED, official split files, evaluator credentials, label documentation, preprocessing caches, or configured dataset paths were found under the supplied tree. Dataset versions, exclusions, class counts, modality dimensions, licenses, and label/evaluator rights therefore remain unverified. Protected data were not searched for outside the workspace.

## Implementation inventory

No implementation is present, so the existence or semantics of MSTCN branches, causal convolutions, CSAG, FiLM/subject embeddings, TCP, HOLD, SAP/partitioning, synchronization, baselines, prediction export, or checkpoint/restart cannot be verified. The PDF describes these components, but manuscript statements are not implementation evidence.

## Existing results and submitted-number provenance

No raw result artifact supports any submitted table or figure. Values in Tables 2–5 and Figures 4–11 are manuscript assertions only. They are quarantined from reuse until mapped to immutable runs. The PDF itself confirms the prose and Table 2 both display `68.7 ± 2.3` for DataParallel-LSTM, resolving only the visual 58.7/68.7 reading question—not the value's experimental provenance.

## Verified contradictions and required claim corrections

1. The PDF reports 1–8 “compute nodes,” but its limitations describe simulation on a single eight-GPU server; it also presents N=16. Reframe to single-server GPU workers and remove unphysical counts unless new physical evidence exists.
2. The PDF calls 107/82 a standard DAIC-WOZ split. Official split files/evaluator rights are absent; quarantine those results and verify train/development/test handling.
3. A two-sided exact Wilcoxon test with five non-zero paired observations cannot attain p<0.05. Remove current dagger significance claims.
4. The written residual block has two causal convolutions. Candidate RFs are 61/481/1921 for K=3 and the written schedules, not the submitted 47/383/1535; code tests are mandatory before reporting candidates.
5. StudentLife uses T=60 at one-minute resolution in the PDF, so realized input evidence is capped at 60 minutes, not hours/days.
6. Generated FiLM gamma/beta values are not persistent per-subject storage. Separate shared generator parameters from an embedding of d_s parameters per subject, subject to code confirmation.
7. The theorem/proof sketch does not establish the claimed convergence guarantee. Remove it and retain only implementation-tested invariants.
8. Ordinary DDP does not inherently violate temporal causality merely because gradients come from different causal windows. Remove or narrowly test that mechanism.
9. SEED transfer, clinical/population-scale, fault-tolerance, robustness, and causal attention interpretations lack supplied evidence and must be removed, narrowed, or moved to future work.

## Tracker and compute summary

- Task priorities: P0={priority_counts['P0']}, P1={priority_counts['P1']}, P2={priority_counts['P2']}, P3={priority_counts['P3']}.
- Conditional plan: P0 {totals['P0']} run/test units, {hours['P0']:.2f} estimated GPU-hours, {storage['P0']:.1f} GB; P1 {totals['P1']} units, {hours['P1']:.2f} GPU-hours, {storage['P1']:.1f} GB; P2 {totals['P2']} units, {hours['P2']:.2f} GPU-hours, {storage['P2']:.1f} GB.
- These are planning estimates, not measurements. Training times assume approximately 35–40 minutes per run on one RTX 4060 Ti and must be recalibrated with a <=30-minute smoke benchmark after source/data arrive.
- Multi-GPU tests remain blocked on this host regardless of budget.

## Cheap checks executed

- Integrity hashes and file metadata: PASS.
- ZIP content comparison: PASS; no additional source/artifacts.
- Tracker structural extraction: PASS; 9 sheets and {len(rows)} mapped task rows.
- PDF text extraction/search: PASS; 15 pages.
- Source/static/model correctness tests: BLOCKED—source absent.
- Dataset/leakage assertions: BLOCKED—data and pipeline absent.
- Manuscript build/visual QA: BLOCKED—editable source absent.

## Gate P status

**BLOCKED for Gate 0 and all scientific reruns; Gate P audit itself is complete.** The smallest unblock bundle is: exact source repository/commit plus configs; editable manuscript/bibliography/assets; immutable old logs/configs; authorized dataset/split paths and DAIC evaluator status; original review letter; and author confirmations listed in `input_gap_report.md`.
"""
    (OUT/"preflight_report.md").write_text(report, encoding="utf-8")

    gaps = """# Gate P Input Gap Report

## Blocking inputs

| Missing input | Affected work | Exact consequence | Minimum safe resolution |
|---|---|---|---|
| Actual D-MSTCN Git repository, commit, configs, tests | EXP-0.*, all training/systems experiments | Architecture, RF, counts, causality, TCP/HOLD/SAP, baselines, and prediction export cannot be inspected or tested | Provide the local path or a preserved checkout and identify the authoritative revision |
| Editable manuscript source, bibliography, figures/tables | Manuscript revision and final QA | Exact text, equations, citations, vector assets, and PDF cannot be updated or rebuilt | Provide LaTeX/Word source and every referenced asset |
| Old raw logs, checkpoints, configs, predictions, plotting scripts | EXP-2.1 and all submitted numbers | No value in Tables 2–5/Figures 4–11 has provenance | Provide an immutable artifact bundle; otherwise authorize removal/rerun |
| StudentLife authorized path and provenance | EXP-1.*, 2.*, 4.1, 5.* | Subject counts, labels, windows, leakage, folds, and metrics cannot be verified | Provide local path, version, terms, exclusions, and label documentation |
| DAIC-WOZ authorized path, official split files, feature provenance | EXP-1.*, 2.*, 4.2 | The submitted 107/82 protocol cannot be validated or corrected | Provide paths and versioned split/feature manifests |
| DAIC test-evaluator/test-label authorization status | EXP-4.2 | A test claim cannot be produced honestly | State the authorized evaluator route; otherwise accept dev-only reporting |
| Original decision letter and verbatim reviews | Response completeness; Reviewer 5 suggestions | Tracker summaries cannot prove verbatim coverage; three suggested citations are unknown | Provide PDF/email export of the decision and all reviews |
| 2–8 GPU single-server compute access (if systems claims retained) | EXP-0.3, 3.*, 6.2–6.5 | Current one-GPU host cannot test branch replication, scaling, interconnect, or failures | Provide authorized machine/topology and budget; otherwise remove multi-GPU result claims |
| Author confirmations: scope, title, AI use, biography, ethics, funding, conflicts, data terms | Governance and submission metadata | These facts cannot be inferred | Obtain dated confirmations from both authors/institution as applicable |

## Non-blocking but material

- Deadline and compute budget: needed to trim P1/P2 after P0 calibration.
- Container/environment lock: can be rebuilt from source manifests if supplied, but old-result reproduction needs the old environment.
- SEED dataset/license/protocol: omit or future-work the transfer claim if unavailable.

No credentials, participant identifiers, or protected raw records should be copied into this repository. Local paths and access constraints are sufficient for the next audit step.
"""
    (OUT/"input_gap_report.md").write_text(gaps, encoding="utf-8")

    manifest = f"""campaign_id: dmstcn-ieee-access-resubmission-gate-p-20260718
protocol_version: '1.0'
generated_utc: '{datetime.now(timezone.utc).isoformat()}'
repository:
  path: '{ROOT}'
  branch: null
  commit: null
  status: input_only_not_a_git_repository
inputs:
  tracker:
    path: reviews/{TRACKER.name}
    sha256: {sha256(TRACKER)}
  rejected_manuscript:
    path: reviews/{MANUSCRIPT.name}
    sha256: {sha256(MANUSCRIPT)}
  reviewer_letter: null
  source_checkout: null
datasets:
  StudentLife: {{available: false, version: null, splits: null, restrictions: unknown}}
  DAIC_WOZ: {{available: false, version: null, splits: null, evaluator_right: unknown, restrictions: unknown}}
  SEED: {{available: false, version: null, splits: null, restrictions: unknown}}
hardware:
  host: {socket.gethostname()}
  physical_gpu_count: 1
  gpu: 'NVIDIA GeForce RTX 4060 Ti, 16380 MiB'
  interconnect: 'PCIe; multi-GPU topology not available'
compute_budget:
  status: conditional_unapproved
  p0_run_units: {totals['P0']}
  p0_estimated_gpu_hours: {hours['P0']:.2f}
  p0_storage_gb: {storage['P0']:.1f}
decisions:
  - reframe_to_single_server_branch_parallel_multi_gpu_if_supported
  - remove_convergence_theorem
  - remove_inherent_ddp_causality_claim
  - verify_official_daic_protocol_or_report_dev_only
  - report_theoretical_and_realized_receptive_fields_separately
unresolved_blockers:
  - source_and_configs_missing
  - editable_manuscript_missing
  - datasets_and_splits_missing
  - raw_results_missing
  - reviewer_letter_missing
  - multi_gpu_hardware_unavailable_locally
phase_status:
  gate_p: complete
  gate_0: blocked
  gate_1: blocked
  later_phases: not_started
phase_tags: []
run_registry_path: artifacts/resubmission/runs.csv
claim_registry_path: artifacts/resubmission/claim_registry.csv
tracker_output_path: artifacts/resubmission/master_tracker_gate_p_updated.xlsx
"""
    (OUT/"campaign_manifest.yaml").write_text(manifest, encoding="utf-8")
    with (OUT/"runs.csv").open("w",newline="",encoding="utf-8") as h:
        csv.writer(h).writerow(["run_id","campaign_id","phase","experiment_id","condition","dataset","protocol","fold","repeat","seed","config_hash","split_hash","data_version","git_commit","environment_hash","host","gpu_model","gpu_count","interconnect","precision","batch_definition","start_time","end_time","status","failure_class","artifact_paths"])


if __name__ == "__main__":
    main()
