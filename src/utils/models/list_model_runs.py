#!/usr/bin/env python3
"""Summarize archived YOLO runs under models/experiments/active."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


def read_last_row(csv_path: Path) -> Dict[str, str]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows[-1] if rows else {}


def summarize_run(run_dir: Path) -> Dict[str, Optional[str]]:
    weights_dir = run_dir / "weights"
    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"

    summary: Dict[str, Optional[str]] = {
        "run": run_dir.name,
        "best_pt": "yes" if (weights_dir / "best.pt").exists() else "no",
        "results": "yes" if results_path.exists() else "no",
        "epochs": None,
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
    }

    if results_path.exists():
        last = read_last_row(results_path)
        summary["epochs"] = last.get("epoch")
        summary["map50"] = last.get("metrics/mAP50(B)")
        summary["map50_95"] = last.get("metrics/mAP50-95(B)")
        summary["precision"] = last.get("metrics/precision(B)")
        summary["recall"] = last.get("metrics/recall(B)")

    if args_path.exists():
        summary["args"] = str(args_path)
    else:
        summary["args"] = None

    return summary


def print_table(rows: List[Dict[str, Optional[str]]]) -> None:
    headers = [
        "run",
        "best_pt",
        "results",
        "epochs",
        "map50",
        "map50_95",
        "precision",
        "recall",
    ]
    col_widths = {
        h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers
    }
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(
            "  ".join(str(row.get(h, "") or "").ljust(col_widths[h]) for h in headers)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List YOLO run metadata under models/experiments/active."
    )
    parser.add_argument(
        "--runs-dir",
        default="models/experiments/active",
        help="Path to the runs directory (default: models/experiments/active).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV to stdout instead of a formatted table.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.exists():
        raise SystemExit(f"{runs_dir} not found")

    summaries: List[Dict[str, Optional[str]]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir():
            summaries.append(summarize_run(run_dir))

    if not summaries:
        print("No runs found.")
        return

    if args.csv:
        import csv as _csv

        writer = _csv.writer(sys.stdout)
        headers = [
            "run",
            "best_pt",
            "results",
            "epochs",
            "map50",
            "map50_95",
            "precision",
            "recall",
        ]
        writer.writerow(headers)
        for row in summaries:
            writer.writerow([row.get(h, "") or "" for h in headers])
    else:
        print_table(summaries)


if __name__ == "__main__":
    main()
