#!/usr/bin/env python3
"""
Final Training Script - Uses Best Parameters from Optuna Study

This script loads the best hyperparameters from the Optuna optimization study
and performs a final training run with more epochs for production model.

Features:
- Automatically loads best parameters from optuna_study.db
- Supports manual parameter override via JSON file
- Trains with extended epochs for better convergence
- Saves production-ready model with metadata
- Creates comprehensive training report
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import find_repo_root

# Fix for PyTorch 2.6+ weights_only=True default
try:
    import ultralytics.nn.tasks as tasks_module

    original_torch_safe_load = tasks_module.torch_safe_load

    def patched_torch_safe_load(weight):
        """Patched version that uses weights_only=False for YOLOv8 compatibility."""
        file = weight
        if isinstance(file, (str, Path)) and not Path(file).exists():
            return original_torch_safe_load(weight)
        return torch.load(file, map_location="cpu", weights_only=False), file

    tasks_module.torch_safe_load = patched_torch_safe_load
except (ImportError, AttributeError):
    pass


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_DATA_CONFIG = SRC_DIR / "training" / "finance-image-parser.yaml"
DEFAULT_WEIGHTS = MODELS_DIR / "pretrained" / "yolov8n.pt"
DEFAULT_STUDY_DB = MODELS_DIR / "experiments" / "active" / "optuna_study.db"
DEFAULT_PROJECT = MODELS_DIR / "experiments" / "final"
PRODUCTION_DIR = MODELS_DIR / "production"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final training with best Optuna parameters."
    )
    parser.add_argument(
        "--study-db",
        type=str,
        default=str(DEFAULT_STUDY_DB),
        help="Path to Optuna study database.",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Optional: JSON file with parameters (overrides Optuna study).",
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
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs for final training. Default: 300",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name. If None, auto-generated with timestamp.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience. Default: 100",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy best model to production directory after training.",
    )

    return parser.parse_args()


def resolve_path(
    path_like: str | os.PathLike[str], base: Optional[Path] = None
) -> Path:
    """Resolve a path relative to base or REPO_ROOT."""
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    base_path = base or REPO_ROOT
    return (base_path / candidate).resolve()


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_best_params_from_db(db_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract best parameters directly from Optuna SQLite database.
    
    Returns:
        Dictionary of best hyperparameters or None if study not found
    """
    if not db_path.exists():
        print(f"Warning: Study database not found at {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study info
        cursor.execute("SELECT study_id, study_name FROM studies")
        study_info = cursor.fetchone()

        if not study_info:
            print("No studies found in database!")
            conn.close()
            return None

        study_id, study_name = study_info

        # Get direction
        cursor.execute(
            "SELECT direction FROM study_directions WHERE study_id = ?",
            (study_id,),
        )
        direction_info = cursor.fetchone()
        direction = direction_info[0] if direction_info else "MAXIMIZE"
        maximize = direction == "MAXIMIZE"

        # Get all completed trials
        cursor.execute(
            """
            SELECT t.trial_id, t.number, tv.value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY t.number
            """,
            (study_id,),
        )
        completed_trials = cursor.fetchall()

        if not completed_trials:
            print("No completed trials found!")
            conn.close()
            return None

        # Get best trial
        best_trial = (
            max(completed_trials, key=lambda t: t[2])
            if maximize
            else min(completed_trials, key=lambda t: t[2])
        )
        best_trial_id, best_number, best_value = best_trial

        # Get parameters for best trial
        cursor.execute(
            """
            SELECT param_name, param_value, distribution_json
            FROM trial_params
            WHERE trial_id = ?
            ORDER BY param_name
            """,
            (best_trial_id,),
        )
        best_params_raw = cursor.fetchall()

        params_dict = {}
        for param_name, param_value, distribution_json in best_params_raw:
            dist_info = json.loads(distribution_json) if distribution_json else {}

            # Handle categorical parameters (stored as indices)
            if dist_info.get("name") == "CategoricalDistribution":
                choices = dist_info.get("attributes", {}).get("choices", [])
                value = (
                    choices[int(param_value)]
                    if int(param_value) < len(choices)
                    else param_value
                )
            else:
                # Numeric parameter
                value = param_value

            params_dict[param_name] = value

        conn.close()

        # Add metadata
        params_dict["_metadata"] = {
            "study_name": study_name,
            "trial_number": best_number,
            "best_map50_95": best_value,
            "direction": direction,
            "loaded_from": str(db_path),
            "loaded_at": datetime.now().isoformat(),
        }

        return params_dict

    except Exception as e:
        print(f"Error loading parameters from database: {e}")
        return None


def load_params_from_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load parameters from JSON file."""
    if not json_path.exists():
        print(f"Warning: JSON file not found at {json_path}")
        return None

    try:
        with open(json_path, "r") as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"Error loading parameters from JSON: {e}")
        return None


def get_default_params() -> Dict[str, Any]:
    """Return default parameters as fallback."""
    return {
        "lr0": 0.001,
        "lrf": 0.0001,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "batch": 16,
        "optimizer": "SGD",
        "mosaic": 1.0,
        "fliplr": 0.5,
        "degrees": 10.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mixup": 0.1,
        "_metadata": {
            "source": "default",
            "note": "Using default YOLOv8 parameters",
        },
    }


def print_parameters(params: Dict[str, Any]) -> None:
    """Print parameters in a formatted way."""
    print("\n" + "=" * 80)
    print("TRAINING PARAMETERS")
    print("=" * 80)

    # Print metadata first if available
    if "_metadata" in params:
        print("\nParameter Source:")
        for k, v in params["_metadata"].items():
            print(f"  {k:20s}: {v}")
        print()

    # Print hyperparameters
    print("Hyperparameters:")
    for k, v in params.items():
        if k != "_metadata":
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.6f}")
            else:
                print(f"  {k:20s}: {v}")
    print("=" * 80 + "\n")


def save_training_metadata(
    save_dir: Path,
    params: Dict[str, Any],
    epochs: int,
    data_config: Path,
    weights: Path,
) -> None:
    """Save comprehensive training metadata to JSON file."""
    metadata = {
        "training_info": {
            "start_time": datetime.now().isoformat(),
            "epochs": epochs,
            "data_config": str(data_config),
            "initial_weights": str(weights),
            "save_directory": str(save_dir),
        },
        "hyperparameters": {k: v for k, v in params.items() if k != "_metadata"},
        "parameter_source": params.get("_metadata", {}),
    }

    metadata_file = save_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training metadata saved to: {metadata_file}")


def train_final_model(
    weights_path: Path,
    data_config_path: Path,
    project_dir: Path,
    run_name: str,
    device: str,
    epochs: int,
    params: Dict[str, Any],
    cache: bool,
    patience: int,
) -> Path:
    """
    Train the final model with best parameters.
    
    Returns:
        Path to the saved model directory
    """
    print("\n" + "=" * 80)
    print("STARTING FINAL TRAINING")
    print("=" * 80)
    print(f"Run Name     : {run_name}")
    print(f"Project Dir  : {project_dir}")
    print(f"Epochs       : {epochs}")
    print(f"Patience     : {patience}")
    print(f"Device       : {device}")
    print(f"Cache Images : {cache}")
    print("=" * 80 + "\n")

    # Clear GPU memory before starting
    clear_gpu_memory()

    # Initialize model
    model = YOLO(str(weights_path))

    # Train with best parameters
    results = model.train(
        data=str(data_config_path),
        epochs=epochs,
        batch=int(params.get("batch", 16)),
        device=device,
        project=str(project_dir),
        name=run_name,
        lr0=float(params.get("lr0", 0.001)),
        lrf=float(params.get("lrf", 0.0001)),
        momentum=float(params.get("momentum", 0.937)),
        weight_decay=float(params.get("weight_decay", 0.0005)),
        optimizer=params.get("optimizer", "SGD"),
        mosaic=float(params.get("mosaic", 1.0)),
        fliplr=float(params.get("fliplr", 0.5)),
        degrees=float(params.get("degrees", 10.0)),
        hsv_h=float(params.get("hsv_h", 0.015)),
        hsv_s=float(params.get("hsv_s", 0.7)),
        hsv_v=float(params.get("hsv_v", 0.4)),
        mixup=float(params.get("mixup", 0.1)),
        cache=cache,
        verbose=True,
        patience=patience,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
    )

    save_dir = Path(results.save_dir)

    # Save training metadata
    save_training_metadata(save_dir, params, epochs, data_config_path, weights_path)

    # Print final metrics
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("\nFinal Metrics:")
        print(f"  mAP50-95   : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  mAP50      : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  Precision  : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall     : {metrics.get('metrics/recall(B)', 0):.4f}")

    print(f"\nResults saved to: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
    print("=" * 80 + "\n")

    return save_dir


def deploy_to_production(save_dir: Path, run_name: str) -> None:
    """
    Deploy the trained model to production directory.
    
    Copies best weights and creates metadata for tracking.
    """
    print("\n" + "=" * 80)
    print("DEPLOYING TO PRODUCTION")
    print("=" * 80)

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    # Copy best weights
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Error: Best weights not found at {best_pt}")
        return

    production_model = PRODUCTION_DIR / "best.pt"
    
    import shutil
    shutil.copy2(best_pt, production_model)
    print(f"✓ Copied best weights to: {production_model}")

    # Save active run metadata
    active_run_file = PRODUCTION_DIR / "active_run.txt"
    with open(active_run_file, "w") as f:
        f.write(f"{run_name}\n")
    print(f"✓ Updated active run: {active_run_file}")

    # Copy training metadata
    metadata_src = save_dir / "training_metadata.json"
    if metadata_src.exists():
        metadata_dst = PRODUCTION_DIR / "training_metadata.json"
        shutil.copy2(metadata_src, metadata_dst)
        print(f"✓ Copied training metadata: {metadata_dst}")

    # Create deployment record
    deployment_record = {
        "deployed_at": datetime.now().isoformat(),
        "run_name": run_name,
        "source_directory": str(save_dir),
        "weights_path": str(production_model),
    }

    deployment_file = PRODUCTION_DIR / "deployment_history.json"
    
    # Load existing history
    history = []
    if deployment_file.exists():
        try:
            with open(deployment_file, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    history.append(deployment_record)
    
    with open(deployment_file, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Updated deployment history: {deployment_file}")
    print("=" * 80)
    print("✓ DEPLOYMENT COMPLETE")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()

    # Resolve paths
    weights_path = resolve_path(args.weights)
    data_config_path = resolve_path(args.data_config)
    study_db_path = resolve_path(args.study_db)
    project_dir = DEFAULT_PROJECT

    # Validate required files
    if not data_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Load parameters (priority: JSON > Optuna DB > Defaults)
    params = None

    if args.params_json:
        json_path = resolve_path(args.params_json)
        params = load_params_from_json(json_path)
        if params:
            print(f"✓ Loaded parameters from JSON: {json_path}")

    if params is None:
        params = load_best_params_from_db(study_db_path)
        if params:
            print(f"✓ Loaded best parameters from Optuna study: {study_db_path}")

    if params is None:
        print("⚠ Using default parameters (no Optuna study or JSON found)")
        params = get_default_params()

    # Print parameters
    print_parameters(params)

    # Generate run name
    run_name = args.name or f"final-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create project directory
    project_dir.mkdir(parents=True, exist_ok=True)

    # Train the model
    save_dir = train_final_model(
        weights_path=weights_path,
        data_config_path=data_config_path,
        project_dir=project_dir,
        run_name=run_name,
        device=args.device,
        epochs=args.epochs,
        params=params,
        cache=args.cache,
        patience=args.patience,
    )

    # Deploy to production if requested
    if args.deploy:
        deploy_to_production(save_dir, run_name)
    else:
        print("\nTo deploy this model to production, run:")
        print(f"  python src/training/train_final.py --deploy")
        print("Or manually copy the weights:")
        print(f"  cp {save_dir}/weights/best.pt {PRODUCTION_DIR}/best.pt")

    print("\n✓ All done!\n")


if __name__ == "__main__":
    main()
