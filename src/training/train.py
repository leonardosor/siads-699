#!/usr/bin/env python3
"""
CLI helper to fine-tune YOLOv8 on the finance form dataset.
Supports Optuna hyperparameter optimization for best model performance.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import optuna
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from PIL import Image

# Fix for PyTorch 2.6+ weights_only=True default
# Monkey-patch ultralytics to use weights_only=False for torch.load
try:
    import ultralytics.nn.tasks as tasks_module
    original_torch_safe_load = tasks_module.torch_safe_load
    
    def patched_torch_safe_load(weight):
        """Patched version that uses weights_only=False for YOLOv8 compatibility."""
        file = weight
        # Check if file exists, if not let original function handle it (will download)
        if isinstance(file, (str, Path)) and not Path(file).exists():
            return original_torch_safe_load(weight)
        return torch.load(file, map_location='cpu', weights_only=False), file
    
    tasks_module.torch_safe_load = patched_torch_safe_load
except (ImportError, AttributeError):
    pass  # Older PyTorch/Ultralytics versions


# Determine repo root: traverse up until we find the directory containing 'src/', 'data/', 'models/'
def _find_repo_root() -> Path:
    """Find the repository root by looking for marker directories."""
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Limit traversal to 5 levels up
        if (
            (current / "src").exists()
            and (current / "data").exists()
            and (current / "models").exists()
        ):
            return current
        current = current.parent
    raise RuntimeError(
        "Could not find repository root. Expected to find 'src/', 'data/', and 'models/' directories."
    )


REPO_ROOT = _find_repo_root()
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_DATA_CONFIG = SRC_DIR / "training" / "finance-image-parser.yaml"
DEFAULT_WEIGHTS = MODELS_DIR / "pretrained" / "yolov8n.pt"
DEFAULT_PROJECT = MODELS_DIR / "experiments" / "active"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 with Optuna hyperparameter optimization."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to starting weights (.pt).",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=str(DEFAULT_DATA_CONFIG),
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id(s) or 'cpu'. Default: '0' (GPU)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name. If None, auto-generated.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        dest="n_trials",
        help="Number of Optuna trials. Default: 20",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs per trial (or full training). Default: 100",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (only for non-Optuna training). Default: 16",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training.",
    )
    parser.add_argument(
        "--clean-broken",
        action="store_true",
        help="Remove corrupt images before training.",
    )

    return parser.parse_args()


def resolve_path(
    path_like: str | os.PathLike[str], base: Optional[Path] = None
) -> Path:
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
    dataset_base = (
        resolve_path(dataset_root, config_dir) if dataset_root else config_dir
    )

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
    """Remove corrupt/unreadable images from dataset directories."""
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
            except Exception as e:
                try:
                    img_path.unlink()
                except FileNotFoundError:
                    pass
                else:
                    removed += 1
                    print(
                        f"  Removed corrupt image: {img_path.name} ({type(e).__name__})"
                    )
    if removed:
        print(f"Removed {removed} corrupt image(s) total.\n")


def validate_dataset(splits: Dict[str, Path], verbose: bool = True) -> int:
    """Validate dataset splits and count images before training."""
    total_images = 0
    if verbose:
        print("=" * 70)
        print("DATASET VALIDATION")
        print("=" * 70)

    for split_name, split_path in sorted(splits.items()):
        if split_path and split_path.exists():
            # Check both split_path directly and split_path/images subdirectory
            image_dirs = [split_path, split_path / "images"]
            image_count = 0

            for img_dir in image_dirs:
                if img_dir.exists():
                    image_count += len(list(img_dir.glob("*.[jJ][pP][gG]")))
                    image_count += len(list(img_dir.glob("*.[jJ][pP][eE][gG]")))
                    image_count += len(list(img_dir.glob("*.[pP][nN][gG]")))

            if verbose:
                print(f"  {split_name:12s}: {image_count:6d} images")
            total_images += image_count
        elif verbose:
            print(f"  {split_name:12s}: NOT FOUND")

    if verbose:
        print("-" * 70)
        print(f"  {'TOTAL':12s}: {total_images:6d} images")
        print("=" * 70 + "\n")

    if total_images == 0:
        raise ValueError("No images found in dataset splits.")

    return total_images


def create_objective(
    weights_path: Path,
    data_config_path: Path,
    project_dir: Path,
    device: str,
    epochs: int,
    cache: bool,
) -> Any:
    """Create Optuna objective function for hyperparameter tuning."""

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
        lrf = trial.suggest_float("lrf", 1e-5, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch = trial.suggest_categorical("batch", [8, 16, 32])
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

        # Augmentation params
        mosaic = trial.suggest_float("mosaic", 0.5, 1.0)
        fliplr = trial.suggest_float("fliplr", 0.0, 0.7)
        degrees = trial.suggest_float("degrees", 0.0, 20.0)
        hsv_h = trial.suggest_float("hsv_h", 0.0, 0.03)
        hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
        hsv_v = trial.suggest_float("hsv_v", 0.0, 0.6)
        mixup = trial.suggest_float("mixup", 0.0, 0.3)

        # Train model
        model = YOLO(str(weights_path))
        results = model.train(
            data=str(data_config_path),
            epochs=epochs,
            batch=batch,
            device=device,
            project=str(project_dir),
            name=f"trial_{trial.number}",
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer=optimizer,
            mosaic=mosaic,
            fliplr=fliplr,
            degrees=degrees,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            mixup=mixup,
            cache=cache,
            verbose=False,
            patience=20,
            save=False,
            plots=False,
        )

        # Return validation mAP50-95 as objective
        return float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    return objective


def train_with_params(
    weights_path: Path,
    data_config_path: Path,
    project_dir: Path,
    device: str,
    epochs: int,
    cache: bool,
    run_name: str,
    params: Optional[Dict[str, Any]] = None,
    batch_override: Optional[int] = None,
) -> Path:
    """Train model with given or default parameters."""

    # Default params if not provided
    if params is None:
        params = {
            "lr0": 0.001,
            "lrf": 0.0001,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "batch": batch_override or 16,
            "optimizer": "SGD",
            "mosaic": 1.0,
            "fliplr": 0.5,
            "degrees": 10.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "mixup": 0.1,
        }
    elif batch_override:
        params["batch"] = batch_override

    print("\n" + "=" * 70)
    print("TRAINING WITH PARAMETERS:")
    for k, v in params.items():
        print(f"  {k:15s}: {v}")
    print("=" * 70 + "\n")

    # Train model
    model = YOLO(str(weights_path))
    results = model.train(
        data=str(data_config_path),
        epochs=epochs,
        batch=params["batch"],
        device=device,
        project=str(project_dir),
        name=run_name,
        lr0=params["lr0"],
        lrf=params["lrf"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
        optimizer=params["optimizer"],
        mosaic=params["mosaic"],
        fliplr=params["fliplr"],
        degrees=params["degrees"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        mixup=params["mixup"],
        cache=cache,
        verbose=True,
        patience=50,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    return save_dir


def main() -> None:
    args = parse_args()

    # Resolve paths
    weights_path = resolve_path(args.weights)
    data_config_path = resolve_path(args.data_config)
    project_dir = MODELS_DIR / "experiments" / "active"

    # Validate paths
    if not data_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Load and validate dataset
    splits = load_dataset_splits(data_config_path)
    validate_dataset(splits)

    if args.clean_broken:
        print("Cleaning corrupt images...")
        clean_corrupt_images(splits.values())

    project_dir.mkdir(parents=True, exist_ok=True)

    if args.optimize:
        # Run Optuna optimization
        print("\n" + "=" * 70)
        print(f"OPTUNA HYPERPARAMETER OPTIMIZATION ({args.n_trials} trials)")
        print("=" * 70 + "\n")

        study = optuna.create_study(
            direction="maximize",
            study_name="yolov8_optimization",
            storage=f"sqlite:///{project_dir}/optuna_study.db",
            load_if_exists=True,
        )

        objective = create_objective(
            weights_path,
            data_config_path,
            project_dir,
            args.device,
            args.epochs,
            args.cache,
        )

        study.optimize(objective, n_trials=args.n_trials)

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Best mAP50-95: {study.best_value:.4f}")
        print("\nBest Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k:15s}: {v}")
        print("=" * 70 + "\n")

        # Train final model with best params
        run_name = args.name or datetime.now().strftime("finance-parser-%Y%m%d_%H%M%S")
        save_dir = train_with_params(
            weights_path,
            data_config_path,
            project_dir,
            args.device,
            args.epochs * 3,
            args.cache,
            run_name,
            study.best_params,
        )
    else:
        # Standard training
        run_name = args.name or datetime.now().strftime("finance-parser-%Y%m%d_%H%M%S")
        save_dir = train_with_params(
            weights_path,
            data_config_path,
            project_dir,
            args.device,
            args.epochs,
            args.cache,
            run_name,
            batch_override=args.batch,
        )

    # Print results
    best_pt = save_dir / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results Directory: {save_dir}")
    print(f"Best Weights     : {best_pt}")
    print("=" * 70)
    print("\nNext steps:")
    print(f"  cp {best_pt} {MODELS_DIR}/production/best.pt")
    print(f"  echo {save_dir.name} > {MODELS_DIR}/production/active_run.txt")


if __name__ == "__main__":
    main()
