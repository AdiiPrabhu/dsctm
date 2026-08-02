#!/usr/bin/env python
"""PARAM Utkarsh monitoring: poller + self-contained dashboard.

    python scripts/param/monitor.py --once                 # one snapshot
    python scripts/param/monitor.py --watch --interval 60  # continuous poller
    python scripts/param/monitor.py --dashboard out.html   # render from history

Why not Grafana. A user account on a shared HPC facility cannot run a Prometheus scraper or
open a listening port, and compute nodes may have no egress at all. So this does what
Grafana would do, in a form the cluster permits: poll SLURM, append immutable JSONL, and
render a self-contained HTML file with NO external assets (no CDN, no fonts, no JS
libraries). Open it over `scp`, or in a browser on the login node. It works offline because
it has to.

Collected:
  * job states from `squeue` and completed-job accounting from `sacct`
  * partition/GPU availability from `sinfo`
  * per-job elapsed, CPU time, max RSS, exit codes
  * campaign progress by scanning run directories for status.json
  * cluster GPU pressure (allocated vs total of the 20 V100s)
"""
from __future__ import annotations

import argparse, getpass, json, os, shutil, socket, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HISTORY = "monitor_history.jsonl"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL, timeout=30).strip()
    except Exception:
        return ""


def _has(tool: str) -> bool:
    return shutil.which(tool) is not None


def poll_squeue(user: str) -> list[dict]:
    if not _has("squeue"):
        return []
    fmt = "%i|%j|%T|%M|%l|%D|%C|%P|%R|%V"
    out = _sh(f"squeue -h -u {user} -o '{fmt}'")
    rows = []
    for line in filter(None, out.splitlines()):
        f = line.split("|")
        if len(f) < 10:
            continue
        rows.append({"job_id": f[0], "name": f[1], "state": f[2], "elapsed": f[3],
                     "time_limit": f[4], "nodes": f[5], "cpus": f[6], "partition": f[7],
                     "reason_or_nodelist": f[8], "submit": f[9]})
    return rows


def poll_sacct(user: str, hours: int = 48) -> list[dict]:
    if not _has("sacct"):
        return []
    fields = "JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,ReqTRES,ExitCode,Start,End"
    out = _sh(f"sacct -u {user} -S now-{hours}hours -P -n --format={fields}")
    rows = []
    for line in filter(None, out.splitlines()):
        f = line.split("|")
        if len(f) < 10 or "." in f[0]:      # skip .batch/.extern steps
            continue
        rows.append(dict(zip(["job_id", "name", "state", "elapsed", "total_cpu",
                              "max_rss", "req_tres", "exit_code", "start", "end"], f)))
    return rows


def poll_sinfo() -> dict:
    if not _has("sinfo"):
        return {}
    partitions = []
    for line in filter(None, _sh("sinfo -h -o '%P|%D|%t|%C|%G'").splitlines()):
        f = line.split("|")
        if len(f) >= 5:
            partitions.append({"partition": f[0], "nodes": f[1], "state": f[2],
                               "cpus_a_i_o_t": f[3], "gres": f[4]})
    gpu_alloc = _sh("squeue -h -p gpu -t RUNNING -o '%b' | grep -o '[0-9]*$' | "
                    "awk '{s+=$1} END {print s+0}'")
    return {"partitions": partitions,
            "gpu_allocated_cluster_wide": int(gpu_alloc) if gpu_alloc.isdigit() else None,
            "gpu_total_documented": 20}


def scan_campaign(results_root: Path) -> dict:
    """Campaign progress by reading status.json in every run directory."""
    families: dict[str, dict] = {}
    if not results_root.exists():
        return {"families": families, "note": f"{results_root} does not exist"}
    for fam_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        counts, failures, receipts = Counter(), [], 0
        for run in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            sp = run / "status.json"
            if not sp.exists():
                counts["no_status"] += 1
                continue
            try:
                blob = json.loads(sp.read_text())
            except Exception:
                counts["unreadable"] += 1
                continue
            counts[blob.get("status", "unknown")] += 1
            if blob.get("status") in ("model_failed", "infrastructure_failed"):
                failures.append({"task": run.name,
                                 "class": blob.get("failure_class")
                                          or blob.get("contract_violation")})
            if (run / "receipt.sha256").exists():
                receipts += 1
        families[fam_dir.name] = {"counts": dict(counts), "receipts": receipts,
                                  "failures": failures[:20],
                                  "total": sum(counts.values())}
    return {"families": families}


def snapshot(user: str, results_root: Path) -> dict:
    return {"timestamp_utc": _utc(), "host": socket.gethostname(), "user": user,
            "squeue": poll_squeue(user), "sacct": poll_sacct(user),
            "cluster": poll_sinfo(), "campaign": scan_campaign(results_root),
            "slurm_available": _has("squeue")}


# --------------------------------------------------------------------------- #
def render_dashboard(history: list[dict], out_path: Path, plan_totals: dict) -> Path:
    latest = history[-1] if history else {}
    camp = latest.get("campaign", {}).get("families", {})
    queue = latest.get("squeue", [])
    sacct = latest.get("sacct", [])
    cluster = latest.get("cluster", {})

    state_counts = Counter(j["state"] for j in queue)
    done = Counter()
    for rows in camp.values():
        done.update(rows["counts"])
    total_planned = sum(plan_totals.values()) or 1
    total_done = done.get("completed", 0)

    def bar(label, value, total, colour):
        pct = 0 if not total else 100 * value / total
        return (f'<div class="row"><span class="lbl">{label}</span>'
                f'<span class="track"><i style="width:{pct:.1f}%;background:{colour}"></i></span>'
                f'<span class="val">{value}/{total}</span></div>')

    fam_rows = ""
    for fam, planned in sorted(plan_totals.items()):
        c = camp.get(fam, {}).get("counts", {})
        ok, failed = c.get("completed", 0), c.get("model_failed", 0) + c.get("infrastructure_failed", 0)
        running = c.get("running", 0)
        pct = 100 * ok / planned if planned else 0
        fam_rows += (f"<tr><td>{fam}</td><td class=n>{planned}</td><td class=n>{ok}</td>"
                     f"<td class=n>{running}</td>"
                     f"<td class='n {'bad' if failed else ''}'>{failed}</td>"
                     f"<td><span class='track sm'><i style='width:{pct:.1f}%;background:var(--ok)'></i></span></td>"
                     f"<td class=n>{pct:.0f}%</td></tr>")

    queue_rows = "".join(
        f"<tr><td>{j['job_id']}</td><td>{j['name']}</td>"
        f"<td><span class='pill {j['state'].lower()}'>{j['state']}</span></td>"
        f"<td>{j['elapsed']}/{j['time_limit']}</td><td class=n>{j['nodes']}</td>"
        f"<td>{j['partition']}</td><td class=dim>{j['reason_or_nodelist']}</td></tr>"
        for j in queue) or "<tr><td colspan=7 class=dim>no jobs in queue</td></tr>"

    recent = "".join(
        f"<tr><td>{j['job_id']}</td><td>{j['name']}</td>"
        f"<td><span class='pill {j['state'].split()[0].lower()}'>{j['state']}</span></td>"
        f"<td>{j['elapsed']}</td><td>{j['max_rss']}</td><td>{j['exit_code']}</td></tr>"
        for j in sacct[-25:]) or "<tr><td colspan=6 class=dim>no accounting records</td></tr>"

    failures = []
    for fam, rows in camp.items():
        for f in rows.get("failures", []):
            failures.append(f"<tr><td>{fam}</td><td class=mono>{f['task']}</td>"
                            f"<td class=dim>{(f['class'] or '')[:120]}</td></tr>")
    fail_html = "".join(failures) or "<tr><td colspan=3 class=dim>no failures recorded</td></tr>"

    # Sparkline of completed-over-time from the history file.
    series = []
    for snap in history[-200:]:
        c = Counter()
        for rows in snap.get("campaign", {}).get("families", {}).values():
            c.update(rows.get("counts", {}))
        series.append(c.get("completed", 0))
    spark = ""
    if len(series) > 1:
        mx = max(series) or 1
        pts = " ".join(f"{i * 600 / (len(series) - 1):.1f},{100 - 95 * v / mx:.1f}"
                       for i, v in enumerate(series))
        spark = (f'<svg viewBox="0 0 600 100" preserveAspectRatio="none" class="spark">'
                 f'<polyline points="{pts}"/></svg>')

    gpu_alloc = cluster.get("gpu_allocated_cluster_wide")
    gpu_total = cluster.get("gpu_total_documented", 20)

    html = f"""<meta charset="utf-8"><title>D-MSTCN · PARAM Utkarsh</title>
<style>
:root{{--bg:#0f1116;--fg:#e6e9ef;--dim:#8b93a7;--card:#171a21;--line:#252a35;
--ok:#3fb950;--warn:#d29922;--bad:#f85149;--run:#58a6ff}}
@media(prefers-color-scheme:light){{:root{{--bg:#f6f7f9;--fg:#1c2027;--dim:#6b7280;
--card:#fff;--line:#e3e6ea}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;padding:24px}}
h1{{font-size:18px;margin:0 0 4px}}h2{{font-size:13px;text-transform:uppercase;
letter-spacing:.08em;color:var(--dim);margin:28px 0 10px}}
.meta{{color:var(--dim);font-size:12px;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}}
.big{{font-size:26px;font-weight:600}}.card .k{{color:var(--dim);font-size:11px;
text-transform:uppercase;letter-spacing:.06em}}
table{{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:8px;overflow:hidden}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--dim);padding:8px 10px;border-bottom:1px solid var(--line)}}
td{{padding:7px 10px;border-bottom:1px solid var(--line);font-size:13px}}
tr:last-child td{{border-bottom:0}}.n{{text-align:right;font-variant-numeric:tabular-nums}}
.dim{{color:var(--dim)}}.mono{{font-size:11px}}.bad{{color:var(--bad)}}
.pill{{padding:1px 7px;border-radius:99px;font-size:11px;background:var(--line)}}
.pill.running{{background:rgba(88,166,255,.18);color:var(--run)}}
.pill.completed{{background:rgba(63,185,80,.18);color:var(--ok)}}
.pill.pending{{background:rgba(210,153,34,.18);color:var(--warn)}}
.pill.failed,.pill.cancelled,.pill.timeout{{background:rgba(248,81,73,.18);color:var(--bad)}}
.track{{display:inline-block;height:7px;background:var(--line);border-radius:99px;
overflow:hidden;width:100%;vertical-align:middle}}
.track.sm{{width:120px}}.track i{{display:block;height:100%}}
.row{{display:grid;grid-template-columns:120px 1fr 70px;gap:10px;align-items:center;
margin:6px 0}}.lbl{{color:var(--dim);font-size:12px}}.val{{text-align:right;font-size:12px}}
.spark{{width:100%;height:70px;margin-top:8px}}
.spark polyline{{fill:none;stroke:var(--ok);stroke-width:2;vector-effect:non-scaling-stroke}}
.note{{color:var(--dim);font-size:11px;margin-top:8px}}
</style>
<h1>D-MSTCN — PARAM Utkarsh</h1>
<div class="meta">{latest.get('host','?')} · user {latest.get('user','?')} ·
snapshot {latest.get('timestamp_utc','never')} · {len(history)} samples ·
{'SLURM detected' if latest.get('slurm_available') else 'SLURM NOT DETECTED (off-cluster render)'}</div>

<div class="grid">
  <div class="card"><div class="k">Campaign complete</div>
    <div class="big">{total_done}<span class="dim" style="font-size:15px">/{total_planned}</span></div>
    <div class="note">{100*total_done/total_planned:.1f}% of planned tasks</div></div>
  <div class="card"><div class="k">Jobs running</div>
    <div class="big">{state_counts.get('RUNNING',0)}</div>
    <div class="note">{state_counts.get('PENDING',0)} pending</div></div>
  <div class="card"><div class="k">Failures</div>
    <div class="big {'bad' if done.get('model_failed',0)+done.get('infrastructure_failed',0) else ''}">
      {done.get('model_failed',0)+done.get('infrastructure_failed',0)}</div>
    <div class="note">model + infrastructure</div></div>
  <div class="card"><div class="k">Cluster GPUs in use</div>
    <div class="big">{gpu_alloc if gpu_alloc is not None else '—'}<span class="dim" style="font-size:15px">/{gpu_total}</span></div>
    <div class="note">V100s across all users</div></div>
</div>

<h2>Progress by family</h2>
<table><tr><th>family</th><th class=n>planned</th><th class=n>done</th><th class=n>running</th>
<th class=n>failed</th><th>progress</th><th class=n>%</th></tr>{fam_rows}</table>

{'<h2>Completed tasks over time</h2><div class="card">' + spark + '</div>' if spark else ''}

<h2>Queue</h2>
<table><tr><th>job</th><th>name</th><th>state</th><th>elapsed/limit</th><th class=n>nodes</th>
<th>partition</th><th>node list / reason</th></tr>{queue_rows}</table>

<h2>Recent accounting (48 h)</h2>
<table><tr><th>job</th><th>name</th><th>state</th><th>elapsed</th><th>max RSS</th>
<th>exit</th></tr>{recent}</table>

<h2>Failures</h2>
<table><tr><th>family</th><th>task</th><th>class</th></tr>{fail_html}</table>

<div class="note" style="margin-top:24px">
Self-contained: no external assets, works offline. Regenerate with
<code>python scripts/param/monitor.py --dashboard &lt;path&gt;</code>.
Only <code>results/param_utkarsh_authoritative/</code> is citable.
</div>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--user", default=os.environ.get("USER") or getpass.getuser())
    ap.add_argument("--results-root",
                    default=os.environ.get("DSCTM_RESULTS_ROOT", "results/param_utkarsh_authoritative"))
    ap.add_argument("--history", default=None)
    ap.add_argument("--dashboard", default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    hist_path = Path(args.history or (results_root.parent / HISTORY))
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from dsctm.campaign import FAMILIES, build_plan
        plan_totals = {f: len(build_plan(f)) for f in FAMILIES}
    except Exception:
        plan_totals = {}

    def one():
        snap = snapshot(args.user, results_root)
        with hist_path.open("a") as fh:
            fh.write(json.dumps(snap, default=str) + "\n")
        camp = snap["campaign"].get("families", {})
        done = sum(v["counts"].get("completed", 0) for v in camp.values())
        print(f"[{snap['timestamp_utc']}] queue={len(snap['squeue'])} "
              f"completed={done}/{sum(plan_totals.values())}", flush=True)
        return snap

    if args.watch:
        print(f"polling every {args.interval}s -> {hist_path}   (ctrl-c to stop)")
        try:
            while True:
                one()
                if args.dashboard:
                    history = [json.loads(l) for l in hist_path.read_text().splitlines() if l]
                    render_dashboard(history, Path(args.dashboard), plan_totals)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped")
        return 0

    if args.once or not args.dashboard:
        one()

    if args.dashboard:
        history = ([json.loads(l) for l in hist_path.read_text().splitlines() if l]
                   if hist_path.exists() else [])
        p = render_dashboard(history, Path(args.dashboard), plan_totals)
        print(f"dashboard: {p}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.exit(main())
