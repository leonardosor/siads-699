#!/usr/bin/env python3
"""
Split ground-truth annotated images into train/val/test sets.

Takes the 100 annotated ground-truth images and creates proper YOLO dataset splits.
"""

import shutil
from pathlib import Path
import random


def prepare_groundtruth_splits(
    ground_truth_dir: Path,
    output_base_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Split ground-truth data into train/val/test sets.

    Args:
        ground_truth_dir: Directory containing ground-truth images and labels/
        output_base_dir: Base directory for splits (e.g., data/input/)
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    ), "Ratios must sum to 1.0"

    print("=" * 70)
    print("Preparing ground-truth dataset splits")
    print("=" * 70)
    print(f"Source: {ground_truth_dir}")
    print(f"Output: {output_base_dir}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Random seed: {seed}")
    print()

    # Get all images and labels
    images_dir = ground_truth_dir
    labels_dir = ground_truth_dir / "labels"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Find all image-label pairs
    image_files = sorted(images_dir.glob("*.jpg"))

    pairs = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"⚠️  No label for: {img_path.name}")

    print(f"Found {len(pairs)} image-label pairs")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # n_test gets the remainder to handle rounding

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    print(f"\nSplit sizes:")
    print(f"  Training:   {len(train_pairs):3d} ({len(train_pairs)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_pairs):3d} ({len(val_pairs)/n_total*100:.1f}%)")
    print(f"  Testing:    {len(test_pairs):3d} ({len(test_pairs)/n_total*100:.1f}%)")
    print()

    # Create output directories
    splits = {"training": train_pairs, "validation": val_pairs, "testing": test_pairs}

    for split_name, split_pairs in splits.items():
        split_img_dir = output_base_dir / split_name / "images"
        split_lbl_dir = output_base_dir / split_name / "labels"

        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"Copying {split_name} split...")
        for img_path, lbl_path in split_pairs:
            # Copy image
            shutil.copy2(img_path, split_img_dir / img_path.name)
            # Copy label
            shutil.copy2(lbl_path, split_lbl_dir / lbl_path.name)

        print(f"  ✓ {len(split_pairs)} images and labels copied to {split_name}/")

    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print("=" * 70)
    print(f"Training:   {output_base_dir / 'training'}")
    print(f"Validation: {output_base_dir / 'validation'}")
    print(f"Testing:    {output_base_dir / 'testing'}")
    print("=" * 70)


def main():
    # Determine repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent

    # Paths
    ground_truth_dir = repo_root / "data" / "input" / "ground-truth"
    output_base_dir = repo_root / "data" / "input"

    # Prepare splits
    prepare_groundtruth_splits(
        ground_truth_dir,
        output_base_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print("\nNext steps:")
    print("1. Verify dataset structure:")
    print("   python src/training/train.py --help")
    print("2. Start training:")
    print("   python src/training/train.py --epochs 100 --batch 16 --cache")
    print("3. Or run hyperparameter optimization:")
    print("   python src/training/train.py --optimize --n-trials 20 --epochs 50")


if __name__ == "__main__":
    main()
