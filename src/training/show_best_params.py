#!/usr/bin/env python3
"""
Show Best Parameters from Optuna Study

Quick utility to display the best hyperparameters from the Optuna optimization study.
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import find_repo_root

REPO_ROOT = find_repo_root()
DEFAULT_STUDY_DB = REPO_ROOT / "models" / "experiments" / "active" / "optuna_study.db"


def load_best_params(db_path: Path) -> dict:
    """Load best parameters from Optuna database."""
    if not db_path.exists():
        print(f"❌ Study database not found: {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study info
        cursor.execute("SELECT study_id, study_name FROM studies")
        study_info = cursor.fetchone()
        if not study_info:
            print("❌ No studies found in database")
            conn.close()
            return None

        study_id, study_name = study_info

        # Get direction
        cursor.execute(
            "SELECT direction FROM study_directions WHERE study_id = ?", (study_id,)
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
            print("❌ No completed trials found")
            conn.close()
            return None

        # Get best trial
        best_trial = (
            max(completed_trials, key=lambda t: t[2])
            if maximize
            else min(completed_trials, key=lambda t: t[2])
        )
        best_trial_id, best_number, best_value = best_trial

        # Get parameters
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

            if dist_info.get("name") == "CategoricalDistribution":
                choices = dist_info.get("attributes", {}).get("choices", [])
                value = (
                    choices[int(param_value)]
                    if int(param_value) < len(choices)
                    else param_value
                )
            else:
                value = param_value

            params_dict[param_name] = value

        conn.close()

        return {
            "study_name": study_name,
            "trial_number": best_number,
            "best_map50_95": best_value,
            "direction": direction,
            "parameters": params_dict,
        }

    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return None


def print_best_params(result: dict) -> None:
    """Print best parameters in a formatted way."""
    print("\n" + "=" * 80)
    print("BEST OPTUNA PARAMETERS")
    print("=" * 80)
    
    print(f"\nStudy Name    : {result['study_name']}")
    print(f"Best Trial    : #{result['trial_number']}")
    print(f"Best mAP50-95 : {result['best_map50_95']:.4f}")
    print(f"Optimization  : {result['direction']}")
    
    print("\nHyperparameters:")
    print("-" * 80)
    
    params = result['parameters']
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, float):
            print(f"  {key:20s} : {value:.6f}")
        else:
            print(f"  {key:20s} : {value}")
    
    print("=" * 80)


def save_to_json(result: dict, output_path: Path) -> None:
    """Save parameters to JSON file."""
    output = {
        "metadata": {
            "study_name": result["study_name"],
            "trial_number": result["trial_number"],
            "best_map50_95": result["best_map50_95"],
            "direction": result["direction"],
        },
        "parameters": result["parameters"],
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Parameters saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Show best parameters from Optuna study"
    )
    parser.add_argument(
        "--study-db",
        type=str,
        default=str(DEFAULT_STUDY_DB),
        help="Path to Optuna study database",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save parameters to JSON file",
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.study_db)
    result = load_best_params(db_path)
    
    if result:
        print_best_params(result)
        
        if args.save_json:
            save_to_json(result, Path(args.save_json))
        
        # Print suggested training command
        print("\n" + "=" * 80)
        print("SUGGESTED TRAINING COMMAND")
        print("=" * 80)
        print("\nTo train with these parameters:")
        print(f"\n  python src/training/train_final.py --epochs 300 --deploy\n")
        print("Or to customize:")
        print(f"  python src/training/train_final.py --epochs 500 --patience 150 --cache --deploy\n")
        print("=" * 80 + "\n")
    else:
        print("\n❌ Failed to load best parameters from Optuna study\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
