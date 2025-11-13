#!/usr/bin/env python3
"""Remap YOLO class IDs in-place to fix label mix-ups."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable


def parse_mapping(pairs: Iterable[str]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for pair in pairs:
        src, dst = pair.split(":")
        mapping[int(src)] = int(dst)
    return mapping


def remap_file(label_path: Path, mapping: Dict[int, int]) -> None:
    lines = []
    changed = False
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls in mapping:
                parts[0] = str(mapping[cls])
                changed = True
            lines.append(" ".join(parts))
    if changed:
        with label_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap YOLO class IDs across a dataset (e.g., fix header/body mix-ups)."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing split folders with labels/ subdirectories.",
    )
    parser.add_argument(
        "--map",
        nargs="+",
        required=True,
        metavar="SRC:DST",
        help="Class ID remaps (e.g., 0:1 1:2 2:0).",
    )
    args = parser.parse_args()

    mapping = parse_mapping(args.map)
    root = Path(args.root).resolve()
    label_dirs = list(root.rglob("labels"))
    if not label_dirs:
        raise SystemExit(f"No labels directories found under {root}")

    for label_dir in label_dirs:
        for txt in label_dir.glob("*.txt"):
            remap_file(txt, mapping)

    print(f"Remapped classes under {root} using mapping {mapping}")


if __name__ == "__main__":
    main()
