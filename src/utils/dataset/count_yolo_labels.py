#!/usr/bin/env python3
"""Utility to summarize YOLO class counts per split for any dataset config."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report YOLO label counts for each split defined in a dataset YAML."
    )
    parser.add_argument(
        "--data-config",
        required=True,
        help="Path to finance-image-parser.yaml (or any YOLO dataset YAML).",
    )
    parser.add_argument(
        "--labels-subdir",
        default="labels",
        help="Subdirectory name containing label .txt files (default: labels).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV instead of human-readable table.",
    )
    return parser.parse_args()


def resolve_split(base: Path, entry: str | None) -> Path | None:
    if not entry:
        return None
    split_path = Path(entry)
    return split_path if split_path.is_absolute() else (base / split_path).resolve()


def load_config(path: Path) -> Tuple[Dict[str, str], Dict[int, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    config_root = path.parent
    dataset_root = Path(config.get("path", "")) if config.get("path") else None
    base = (
        dataset_root.resolve()
        if dataset_root and dataset_root.is_absolute()
        else (config_root / dataset_root if dataset_root else config_root)
    )

    splits = {
        key: resolve_split(base, config.get(key)) for key in ("train", "val", "test")
    }
    names = config.get("names") or {}
    if isinstance(names, list):
        names = {idx: label for idx, label in enumerate(names)}
    return {k: v for k, v in splits.items() if v}, names


def count_labels(labels_dir: Path) -> Counter:
    counter: Counter = Counter()
    if not labels_dir.exists():
        return counter
    for label_file in labels_dir.rglob("*.txt"):
        try:
            with label_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    class_id = int(line.split()[0])
                    counter[class_id] += 1
        except Exception as exc:  # pragma: no cover - best-effort stats
            print(f"[warn] Skipping {label_file}: {exc}", file=sys.stderr)
    return counter


def format_table(results: Dict[str, Counter], names: Dict[int, str]) -> str:
    lines = [
        "Split\tTotal\t"
        + "\t".join(names.get(i, f"class_{i}") for i in sorted(names.keys()))
    ]
    for split, counts in results.items():
        total = sum(counts.values())
        row = [split, str(total)]
        for cls_id in sorted(names.keys()):
            row.append(str(counts.get(cls_id, 0)))
        lines.append("\t".join(row))
    return "\n".join(lines)


def emit_csv(results: Dict[str, Counter], names: Dict[int, str]) -> None:
    writer = csv.writer(sys.stdout)
    header = ["split", "total"] + [
        names.get(i, f"class_{i}") for i in sorted(names.keys())
    ]
    writer.writerow(header)
    for split, counts in results.items():
        total = sum(counts.values())
        row = [split, total] + [counts.get(i, 0) for i in sorted(names.keys())]
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    data_config = Path(args.data_config).resolve()
    splits, names = load_config(data_config)

    if not splits:
        raise SystemExit(
            "No splits found in config; ensure train/val/test are defined."
        )

    all_results: Dict[str, Counter] = {}
    for split_name, split_path in splits.items():
        labels_dir = split_path
        if labels_dir.name != args.labels_subdir:
            labels_dir = labels_dir / args.labels_subdir
        counter = count_labels(labels_dir)
        all_results[split_name] = counter

    if args.csv:
        emit_csv(all_results, names)
    else:
        print(format_table(all_results, names))


if __name__ == "__main__":
    main()
