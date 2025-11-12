#!/usr/bin/env python3
"""
CLI helper to fine-tune YOLOv8 on the finance form dataset.
Runs both locally and on Great Lakes without path edits.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml
from ultralytics import YOLO

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CONFIG = Path(__file__).with_name("finance-image-parser.yaml")
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 with reproducible defaults.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to starting weights (.pt).")
    parser.add_argument(
        "--data-config",
        type=str,
        default=str(DEFAULT_DATA_CONFIG),
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640, help="Square image size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction.")
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005, dest="weight_decay")
    parser.add_argument("--project", type=str, default="runs/detect", help="Relative or absolute project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name inside the project directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", action="store_true", help="Cache images to RAM/disk for faster epochs.")
    parser.add_argument("--cos-lr", action="store_true", dest="cos_lr")
    parser.add_argument("--clean-broken", action="store_true", help="Drop unreadable images before training.")
    parser.add_argument("--exist-ok", action="store_true", dest="exist_ok", help="Overwrite an existing run.")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent matching run.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability (0 disables).")
    return parser.parse_args()


def resolve_path(path_like: str | os.PathLike[str], base: Optional[Path] = None) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    base_path = base or REPO_ROOT
    return (base_path / candidate).resolve()


def load_dataset_splits(config_path: Path) -> Dict[str, Path]:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    config_dir = config_path.parent
    dataset_root = config.get("path")
    dataset_base = resolve_path(dataset_root, config_dir) if dataset_root else config_dir

    def _resolve(entry: Optional[str]) -> Optional[Path]:
        if not entry:
            return None
        candidate = Path(entry)
        if candidate.is_absolute():
            return candidate
        return (dataset_base / candidate).resolve()

    splits = {split: _resolve(config.get(split)) for split in ("train", "val", "test")}
    return {k: v for k, v in splits.items() if v is not None}


def clean_corrupt_images(image_dirs: Iterable[Path]) -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    removed = 0
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for img_path in img_dir.rglob("*"):
            if img_path.suffix.lower() not in exts:
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                try:
                    img_path.unlink()
                except FileNotFoundError:
                    pass
                else:
                    removed += 1
    if removed:
        print(f"Removed {removed} corrupt image(s).")


def main() -> None:
    args = parse_args()

    weights_path = resolve_path(args.weights)
    data_config_path = resolve_path(args.data_config)
    project_dir = resolve_path(args.project)

    if not data_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    splits = load_dataset_splits(data_config_path)
    if args.clean_broken:
        clean_corrupt_images(splits.values())

    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.name or datetime.now().strftime("finance-parser-%Y%m%d_%H%M%S")

    print(
        f"Starting training -> weights: {weights_path}, data: {data_config_path}, "
        f"project: {project_dir}, run: {run_name}"
    )

    model = YOLO(str(weights_path))
    results = model.train(
        data=str(data_config_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        patience=args.patience,
        seed=args.seed,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        cache=args.cache,
        resume=args.resume,
        exist_ok=args.exist_ok,
        mosaic=args.mosaic,
    )

    save_dir = Path(getattr(results, "save_dir", project_dir / run_name))
    metrics_csv = save_dir / "results.csv"
    results_png = save_dir / "results.png"
    confusion_png = save_dir / "confusion_matrix.png"

    print(f"Artifacts directory: {save_dir}")
    if metrics_csv.exists():
        print(f"- Metrics table: {metrics_csv}")
    if results_png.exists():
        print(f"- Training curves: {results_png}")
    if confusion_png.exists():
        print(f"- Confusion matrix: {confusion_png}")


if __name__ == "__main__":
    main()
