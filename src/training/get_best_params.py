#!/usr/bin/env python3
"""
Simple script to extract best parameters from Optuna study database.
Uses only sqlite3 (part of standard library) so no extra dependencies needed.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find the repository root by looking for .git directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent


REPO_ROOT = find_repo_root()
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_STUDY_DB = MODELS_DIR / "experiments" / "active" / "optuna_study.db"


def get_best_params_from_db(db_path: Path) -> None:
    """Query SQLite database directly to get best trial parameters."""

    if not db_path.exists():
        print(f"Error: Study database not found at {db_path}")
        return

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get study info
    cursor.execute("SELECT study_id, study_name FROM studies")
    study_info = cursor.fetchone()

    if not study_info:
        print("No studies found in database!")
        conn.close()
        return

    study_id, study_name = study_info

    # Get direction from study_directions table
    cursor.execute(
        "SELECT direction FROM study_directions WHERE study_id = ?",
        (study_id,),
    )
    direction_info = cursor.fetchone()
    direction = direction_info[0] if direction_info else "MAXIMIZE"
    maximize = direction == "MAXIMIZE"

    print("=" * 80)
    print("OPTUNA STUDY ANALYSIS")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Direction: {direction}")

    # Get all trials with their values
    cursor.execute(
        """
        SELECT t.trial_id, t.number, t.state, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ?
        ORDER BY t.number
        """,
        (study_id,),
    )
    trials = cursor.fetchall()

    print(f"Total trials: {len(trials)}")

    # Count states
    completed_trials = [t for t in trials if t[2] == "COMPLETE"]
    failed_trials = [t for t in trials if t[2] == "FAIL"]

    print(f"Completed trials: {len(completed_trials)}")
    print(f"Failed trials: {len(failed_trials)}")
    print()

    if not completed_trials:
        print("No completed trials found!")
        conn.close()
        return

    # Get best trial (max or min depending on direction)
    best_trial = (
        max(completed_trials, key=lambda t: t[3])
        if maximize
        else min(completed_trials, key=lambda t: t[3])
    )
    best_trial_id, best_number, _, best_value = best_trial

    print("=" * 80)
    print("BEST TRIAL")
    print("=" * 80)
    print(f"Trial number: {best_number}")
    print(f"Best mAP50-95: {best_value:.4f}")
    print()

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
    best_params = cursor.fetchall()

    print("Best Hyperparameters:")
    params_dict = {}
    for param_name, param_value, distribution_json in best_params:
        # param_value is stored as FLOAT in DB, but may represent different types
        # Check distribution to determine actual type
        dist_info = json.loads(distribution_json) if distribution_json else {}

        # Handle categorical parameters (stored as indices)
        if dist_info.get("name") == "CategoricalDistribution":
            choices = dist_info.get("attributes", {}).get("choices", [])
            # param_value is the index
            value = (
                choices[int(param_value)]
                if int(param_value) < len(choices)
                else param_value
            )
        else:
            # Numeric parameter
            value = param_value

        params_dict[param_name] = value
        print(f"  {param_name:15s}: {value}")
    print()

    # Show all completed trials sorted by value
    print("=" * 80)
    print("ALL COMPLETED TRIALS (sorted by performance)")
    print("=" * 80)
    print(f"{'Trial':<10} {'mAP50-95':<15} {'Parameters'}")
    print("-" * 80)

    sorted_trials = sorted(completed_trials, key=lambda t: t[3], reverse=maximize)
    for trial_id, number, _, value in sorted_trials[:10]:  # Show top 10
        cursor.execute(
            """
            SELECT param_name, param_value, distribution_json
            FROM trial_params
            WHERE trial_id = ?
            LIMIT 3
            """,
            (trial_id,),
        )
        params = cursor.fetchall()
        params_list = []
        for pname, pval, pdist in params:
            dist_info = json.loads(pdist) if pdist else {}
            if dist_info.get("name") == "CategoricalDistribution":
                choices = dist_info.get("attributes", {}).get("choices", [])
                val = choices[int(pval)] if int(pval) < len(choices) else pval
            else:
                val = pval
            params_list.append(f"{pname}={val}")
        params_str = ", ".join(params_list)
        print(f"{number:<10} {value:<15.4f} {params_str[:60]}...")
    print()

    # Save best params to JSON
    json_path = db_path.parent / "best_params.json"
    with open(json_path, "w") as f:
        json.dump(params_dict, f, indent=2)
    print(f"Best parameters saved to: {json_path}")
    print()

    print("=" * 80)
    print("RECOMMENDED COMMAND FOR FINAL TRAINING")
    print("=" * 80)
    print("\nUse these parameters with train.py:")
    print()
    cmd_parts = ["python src/training/train.py"]
    for k, v in params_dict.items():
        if isinstance(v, str):
            cmd_parts.append(f"--{k} '{v}'")
        else:
            cmd_parts.append(f"--{k} {v}")
    print(" \\\n  ".join(cmd_parts))
    print()

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract best parameters from Optuna study"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_STUDY_DB),
        help="Path to Optuna study database",
    )

    args = parser.parse_args()
    db_path = Path(args.db)

    get_best_params_from_db(db_path)


if __name__ == "__main__":
    main()
