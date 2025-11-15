#!/usr/bin/env python3
"""Visualize YOLO label files to sanity-check class assignments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml
from PIL import Image, ImageDraw, ImageFont

UM_BLUE = "#00274C"
UM_MAIZE = "#FFCB05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO label files for QC.")
    parser.add_argument(
        "--image", required=True, help="Path to the source image (JPG/PNG)."
    )
    parser.add_argument(
        "--labels", required=True, help="Path to the matching YOLO label .txt file."
    )
    parser.add_argument(
        "--names-yaml",
        default="src/yolo_v8/finance-image-parser.yaml",
        help="Dataset YAML containing `names` entries.",
    )
    parser.add_argument(
        "--output", default="preview.jpg", help="Where to save the annotated preview."
    )
    return parser.parse_args()


def load_class_names(names_yaml: Path) -> List[str]:
    with names_yaml.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return names
    return [names[k] for k in sorted(names.keys())]


def load_labels(label_path: Path):
    boxes = []
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((cls, xc, yc, w, h))
    return boxes


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, width: int, height: int):
    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    label_path = Path(args.labels)
    names_path = Path(args.names_yaml)

    class_names = load_class_names(names_path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    boxes = load_labels(label_path)
    width, height = image.size

    for cls, xc, yc, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, width, height)
        draw.rectangle((x1, y1, x2, y2), outline=UM_BLUE, width=3)
        label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        text = f"{label}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill=UM_MAIZE)
        draw.text((x1, y1), text, font=font, fill=UM_BLUE)

    output_path = Path(args.output)
    image.save(output_path)
    print(f"Saved preview -> {output_path}")


if __name__ == "__main__":
    main()
