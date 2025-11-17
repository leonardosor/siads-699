#!/usr/bin/env python3
"""
Analyze Optuna study results and extract best parameters from completed trials.
"""

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common import find_repo_root

REPO_ROOT = find_repo_root()
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_STUDY_DB = MODELS_DIR / "experiments" / "active" / "optuna_study.db"


def analyze_study(db_path: Path) -> None:
    """Analyze Optuna study and print results."""

    if not db_path.exists():
        print(f"Error: Study database not found at {db_path}")
        return

    # Load the study
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(
        study_name="yolov8_optimization",
        storage=storage,
    )

    print("=" * 80)
    print("OPTUNA STUDY ANALYSIS")
    print("=" * 80)
    print(f"Study name: {study.study_name}")
    print(f"Direction: {study.direction.name}")
    print(f"Total trials: {len(study.trials)}")

    # Get completed trials
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"Completed trials: {len(completed_trials)}")
    print(f"Failed trials: {len(failed_trials)}")
    print()

    if not completed_trials:
        print("No completed trials found!")
        return

    # Show best trial
    best_trial = study.best_trial
    print("=" * 80)
    print("BEST TRIAL")
    print("=" * 80)
    print(f"Trial number: {best_trial.number}")
    print(f"Best mAP50-95: {best_trial.value:.4f}")
    print()
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k:15s}: {v}")
    print()

    # Show all completed trials
    print("=" * 80)
    print("ALL COMPLETED TRIALS")
    print("=" * 80)

    results = []
    for trial in completed_trials:
        result = {"trial": trial.number, "mAP50-95": trial.value}
        result.update(trial.params)
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("mAP50-95", ascending=False)

    # Print top 5 trials
    print("\nTop 5 Trials:")
    print(df.head(5).to_string(index=False))
    print()

    # Save full results to CSV
    csv_path = db_path.parent / "optuna_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to: {csv_path}")
    print()

    # Show failed trials
    if failed_trials:
        print("=" * 80)
        print("FAILED TRIALS")
        print("=" * 80)
        for trial in failed_trials:
            print(f"Trial {trial.number}: {trial.state.name}")
            if trial.user_attrs:
                print(f"  User attributes: {trial.user_attrs}")
        print()

    print("=" * 80)
    print("RECOMMENDED PARAMETERS FOR FINAL TRAINING")
    print("=" * 80)
    print("\nUse these parameters with train.py:")
    print()
    print("Best parameters (Trial {}):".format(best_trial.number))
    for k, v in best_trial.params.items():
        print(f"  --{k} {v}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna study results")
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_STUDY_DB),
        help="Path to Optuna study database",
    )

    args = parser.parse_args()
    db_path = Path(args.db)

    analyze_study(db_path)


if __name__ == "__main__":
    main()
