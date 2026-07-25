"""Aggregate scaling reports and calculate speedup/efficiency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import utc_now, write_json


def read_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("validation") != "scaling" or report.get("status") != "pass":
        raise ValueError(f"not a passing scaling report: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="artifacts/multigpu")
    parser.add_argument("--output", default="artifacts/multigpu/summary.json")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    reports = sorted(
        (read_report(path) for path in input_dir.glob("scaling-w*.json")),
        key=lambda report: report["hardware"]["world_size"],
    )
    if not reports or reports[0]["hardware"]["world_size"] != 1:
        raise RuntimeError("same-host scaling-w1.json baseline is required")

    baseline = reports[0]
    baseline_host = baseline["hardware"]["hostname"]
    baseline_commit = baseline["hardware"]["git_commit"]
    base_throughput = baseline["strong"]["throughput_samples_per_second"]
    rows = []
    for report in reports:
        hardware = report["hardware"]
        if hardware["hostname"] != baseline_host:
            raise RuntimeError("all scaling reports must come from the same host")
        if hardware["git_commit"] != baseline_commit:
            raise RuntimeError("all scaling reports must use the same Git commit")
        world_size = hardware["world_size"]
        throughput = report["strong"]["throughput_samples_per_second"]
        speedup = throughput / base_throughput
        rows.append(
            {
                "world_size": world_size,
                "strong_throughput_samples_per_second": throughput,
                "strong_speedup": speedup,
                "strong_scaling_efficiency": speedup / world_size,
                "weak_throughput_samples_per_second": report["weak"][
                    "throughput_samples_per_second"
                ],
                "strong_peak_memory_bytes_per_rank": report["strong"]["peak_memory_bytes"],
                "weak_peak_memory_bytes_per_rank": report["weak"]["peak_memory_bytes"],
            }
        )
    write_json(
        args.output,
        {
            "status": "pass",
            "timestamp_utc": utc_now(),
            "hostname": baseline_host,
            "git_commit": baseline_commit,
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
