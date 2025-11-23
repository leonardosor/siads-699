#!/usr/bin/env python3
"""
Quick Helper - Show Best Optuna Parameters and Training Command

This is a simplified script that quickly shows:
1. Best parameters from the Optuna study
2. Ready-to-use command for final training
"""

import sys
from pathlib import Path

# Find repo root
def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import the function from get_best_params
from training.get_best_params import get_best_params_from_db

MODELS_DIR = REPO_ROOT / "models"
STUDY_DB = MODELS_DIR / "experiments" / "active" / "optuna_study.db"


def main():
    print("\n" + "=" * 80)
    print("BEST OPTUNA PARAMETERS - QUICK VIEW")
    print("=" * 80 + "\n")

    if not STUDY_DB.exists():
        print(f"❌ Optuna study database not found at:")
        print(f"   {STUDY_DB}")
        print("\nYou need to run hyperparameter optimization first:")
        print("   python src/training/train.py --optimize --n-trials 20")
        sys.exit(1)

    # Get best parameters
    get_best_params_from_db(STUDY_DB)

    print("\n" + "=" * 80)
    print("RECOMMENDED FINAL TRAINING COMMANDS")
    print("=" * 80)

    print("\n1. BASIC FINAL TRAINING (300 epochs):")
    print("   " + "-" * 76)
    print("   python src/training/train_final.py")

    print("\n2. TRAIN WITH AUTO-DEPLOYMENT:")
    print("   " + "-" * 76)
    print("   python src/training/train_final.py --deploy")

    print("\n3. EXTENDED TRAINING (500 epochs):")
    print("   " + "-" * 76)
    print("   python src/training/train_final.py --epochs 500 --patience 150")

    print("\n4. FAST TRAINING WITH CACHING:")
    print("   " + "-" * 76)
    print("   python src/training/train_final.py --cache --epochs 300")

    print("\n5. MAXIMUM PERFORMANCE:")
    print("   " + "-" * 76)
    print("   python src/training/train_final.py \\")
    print("     --epochs 500 \\")
    print("     --patience 150 \\")
    print("     --cache \\")
    print("     --deploy")

    print("\n" + "=" * 80)
    print("ADDITIONAL OPTIONS")
    print("=" * 80)

    print("\nFor more options, see:")
    print("   python src/training/train_final.py --help")

    print("\nFor detailed documentation:")
    print("   cat docs/FINAL_TRAINING.md")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
