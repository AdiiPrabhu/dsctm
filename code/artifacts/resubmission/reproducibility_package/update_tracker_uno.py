#!/usr/bin/python3
"""Copy the original tracker and append evidence-status columns using LibreOffice UNO."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "resubmission"
SOURCE = ROOT / "reviews" / "D_MSTCN_IEEE_Access_Resubmission_Tracker_Completed.xlsx"
TARGET = OUT / "master_tracker_gate_p_updated.xlsx"
MAP = OUT / "reviewer_to_experiment_map.csv"


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def connect():
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(60):
        try:
            return resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Could not connect to LibreOffice UNO listener")


def main():
    shutil.copy2(SOURCE, TARGET)
    process = subprocess.Popen([
        "libreoffice", "--headless", "--nologo", "--nodefault", "--nofirststartwizard", "--norestore",
        "--accept=socket,host=localhost,port=2002;urp;StarOffice.ServiceManager"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    document = None
    try:
        context = connect()
        service = context.ServiceManager
        desktop = service.createInstanceWithContext("com.sun.star.frame.Desktop", context)
        document = desktop.loadComponentFromURL(uno.systemPathToFileUrl(str(TARGET)), "_blank", 0, (prop("Hidden", True),))
        if document.Sheets.hasByName("Gate P Audit"):
            document.Sheets.removeByName("Gate P Audit")
        document.Sheets.insertNewByName("Gate P Audit", document.Sheets.getCount())
        sheet = document.Sheets.getByName("Gate P Audit")
        headers = ["Tracker Task ID", "Verified Status", "Decision / Proposed Resolution", "Detailed Scientific Justification",
                   "Exact Code or Manuscript Change", "Experiment and Run IDs", "Results with Uncertainty",
                   "Evidence / Artifact Paths", "Draft Response to Reviewer", "Acceptance Test and Result",
                   "Remaining Limitation or Blocker"]
        with MAP.open(newline="", encoding="utf-8") as handle:
            mapped = {row["tracker_task_id"]: row for row in csv.DictReader(handle)}
        for offset, header in enumerate(headers):
            cell = sheet.getCellByPosition(offset, 0)
            cell.String = header
            cell.CharWeight = 150.0
            cell.CellBackColor = 0xD9EAF7
        source_sheet = document.Sheets.getByName("Master Tracker")
        cursor = source_sheet.createCursor()
        cursor.gotoEndOfUsedArea(True)
        source_last_row = cursor.RangeAddress.EndRow
        output_row = 1
        for row_idx in range(3, source_last_row + 1):
            task_id = source_sheet.getCellByPosition(0, row_idx).String
            if task_id not in mapped:
                continue
            record = mapped[task_id]
            proposed = source_sheet.getCellByPosition(20, row_idx).String
            justification = source_sheet.getCellByPosition(21, row_idx).String
            exact_change = source_sheet.getCellByPosition(22, row_idx).String
            response = source_sheet.getCellByPosition(23, row_idx).String
            acceptance = source_sheet.getCellByPosition(24, row_idx).String
            values = [task_id, record["current_status"], proposed, justification, exact_change,
                      record["experiment_ids"] + " (no run IDs yet)",
                      "Not available; no raw results supplied or generated at Gate P",
                      record["evidence_paths"] + ";artifacts/resubmission/preflight_report.md;artifacts/resubmission/reviewer_to_experiment_map.csv",
                      response,
                      "PENDING/BLOCKED — " + acceptance,
                      record["blocker"]]
            for offset, value in enumerate(values):
                cell = sheet.getCellByPosition(offset, output_row)
                cell.String = value
                cell.IsTextWrapped = True
            output_row += 1
        for col in range(0, 11):
            sheet.Columns.getByIndex(col).Width = 6500
        sheet.Columns.getByIndex(0).Width = 2500
        sheet.getCellRangeByPosition(0, 0, 10, output_row - 1).IsTextWrapped = True
        sheet.getCellRangeByPosition(0, 0, 10, 0).CharWeight = 150.0
        document.storeAsURL(uno.systemPathToFileUrl(str(TARGET)), (prop("FilterName", "Calc MS Excel 2007 XML"), prop("Overwrite", True)))
        document.close(True)
    finally:
        if document is not None:
            try:
                document.close(True)
            except Exception:
                pass
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
