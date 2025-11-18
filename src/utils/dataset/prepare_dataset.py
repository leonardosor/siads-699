#!/usr/bin/env python3
"""
Unified dataset preparation script with multiple modes.

Modes:
  groundtruth - Split ground-truth images into train/val/test (no augmentation)
  augmented   - Generate augmented dataset from ground-truth images
  parquet     - Extract dataset from parquet files

Examples:
  # Simple split
  python prepare_dataset.py groundtruth

  # With augmentation (recommended)
  python prepare_dataset.py augmented --augmentations-per-image 50

  # From parquet files
  python prepare_dataset.py parquet --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Try importing pandas for parquet mode
try:
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BoundingBox,
    add_gaussian_noise,
    adjust_background_color,
    adjust_brightness_contrast,
    adjust_hsv,
    find_repo_root,
    horizontal_flip,
    load_yolo_labels,
    save_yolo_labels,
)

# ============================================================================
# Common Utilities
# ============================================================================


def validate_ratios(train: float, val: float, test: float) -> None:
    """Validate that split ratios sum to 1.0"""
    total = train + val + test
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def split_samples(
    samples: List,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    """Split samples into train/val/test sets"""
    random.seed(seed)
    np.random.seed(seed)

    # Shuffle
    samples_copy = samples.copy()
    random.shuffle(samples_copy)

    # Split
    n_total = len(samples_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = samples_copy[:n_train]
    val_samples = samples_copy[n_train : n_train + n_val]
    test_samples = samples_copy[n_train + n_val :]

    return train_samples, val_samples, test_samples


def backup_existing_data(output_dir: Path, seed: int) -> None:
    """Backup existing training/validation/testing directories"""
    for split in ["training", "validation", "testing"]:
        split_dir = output_dir / split
        if split_dir.exists():
            backup_dir = output_dir / f"{split}_backup_{seed}"
            print(f"Backing up {split_dir} -> {backup_dir}")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(split_dir, backup_dir)


def save_split(
    split_name: str,
    samples: List,
    output_dir: Path,
    clear_existing: bool = True,
) -> None:
    """Save a split (train/val/test) to disk"""
    split_img_dir = output_dir / split_name / "images"
    split_lbl_dir = output_dir / split_name / "labels"

    # Clear existing directories if requested
    if clear_existing:
        if split_img_dir.exists():
            shutil.rmtree(split_img_dir)
        if split_lbl_dir.exists():
            shutil.rmtree(split_lbl_dir)

    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    return split_img_dir, split_lbl_dir


# ============================================================================
# Mode 1: Ground-Truth (Simple Split)
# ============================================================================


def prepare_groundtruth(
    ground_truth_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    backup_existing: bool = False,
    seed: int = 42,
) -> None:
    """Split ground-truth images into train/val/test (no augmentation)"""
    validate_ratios(train_ratio, val_ratio, test_ratio)

    print("=" * 70)
    print("PREPARING GROUND-TRUTH DATASET (NO AUGMENTATION)")
    print("=" * 70)
    print(f"Ground-truth source: {ground_truth_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Random seed: {seed}")
    print()

    # Backup if requested
    if backup_existing:
        backup_existing_data(output_dir, seed)

    # Find image-label pairs
    images_dir = ground_truth_dir
    labels_dir = ground_truth_dir / "labels"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_files = sorted(images_dir.glob("*.jpg"))
    pairs = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"⚠️  No label for: {img_path.name}")

    print(f"Found {len(pairs)} image-label pairs")

    # Split
    train_pairs, val_pairs, test_pairs = split_samples(
        pairs, train_ratio, val_ratio, test_ratio, seed
    )

    n_total = len(pairs)
    print(f"\nSplit sizes:")
    print(f"  Training:   {len(train_pairs):3d} ({len(train_pairs)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_pairs):3d} ({len(val_pairs)/n_total*100:.1f}%)")
    print(f"  Testing:    {len(test_pairs):3d} ({len(test_pairs)/n_total*100:.1f}%)")
    print()

    # Save splits
    splits = {"training": train_pairs, "validation": val_pairs, "testing": test_pairs}

    for split_name, split_pairs in splits.items():
        split_img_dir, split_lbl_dir = save_split(split_name, split_pairs, output_dir)

        print(f"Copying {split_name} split...")
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, split_img_dir / img_path.name)
            shutil.copy2(lbl_path, split_lbl_dir / lbl_path.name)

        print(f"  ✓ {len(split_pairs)} images and labels copied to {split_name}/")

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)


# ============================================================================
# Mode 2: Augmented Ground-Truth
# ============================================================================


def apply_random_augmentation(
    img: np.ndarray, boxes: List[BoundingBox]
) -> Tuple[np.ndarray, List[BoundingBox], str]:
    """Apply a random augmentation to the image and boxes"""
    aug_type = random.choice(
        [
            "horizontal_flip",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "noise",
            "hsv_shift",
            "combined",
            "bg_white",
            "bg_cream",
            "bg_gray",
            "bg_normalize",
        ]
    )

    if aug_type == "horizontal_flip":
        return *horizontal_flip(img, boxes), "hflip"

    elif aug_type == "brightness_up":
        return adjust_brightness_contrast(img, alpha=1.0, beta=30), boxes, "bright"

    elif aug_type == "brightness_down":
        return adjust_brightness_contrast(img, alpha=1.0, beta=-30), boxes, "dark"

    elif aug_type == "contrast_up":
        return adjust_brightness_contrast(img, alpha=1.3, beta=0), boxes, "contrast"

    elif aug_type == "contrast_down":
        return adjust_brightness_contrast(img, alpha=0.7, beta=0), boxes, "lowcontrast"

    elif aug_type == "noise":
        return add_gaussian_noise(img, std=15), boxes, "noise"

    elif aug_type == "hsv_shift":
        h_shift = random.randint(-10, 10)
        s_scale = random.uniform(0.8, 1.2)
        v_scale = random.uniform(0.8, 1.2)
        return adjust_hsv(img, h_shift, s_scale, v_scale), boxes, "hsv"

    elif aug_type == "combined":
        aug_img = img.copy()
        aug_boxes = boxes.copy()

        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            aug_img = adjust_brightness_contrast(aug_img, alpha=alpha, beta=beta)

        if random.random() > 0.5:
            h_shift = random.randint(-5, 5)
            s_scale = random.uniform(0.9, 1.1)
            v_scale = random.uniform(0.9, 1.1)
            aug_img = adjust_hsv(aug_img, h_shift, s_scale, v_scale)

        return aug_img, aug_boxes, "combined"

    elif aug_type == "bg_white":
        return adjust_background_color(img, "white"), boxes, "bgwhite"

    elif aug_type == "bg_cream":
        return adjust_background_color(img, "cream"), boxes, "bgcream"

    elif aug_type == "bg_gray":
        return adjust_background_color(img, "gray"), boxes, "bggray"

    elif aug_type == "bg_normalize":
        return adjust_background_color(img, "normalize"), boxes, "bgnorm"

    return img, boxes, "none"


def prepare_augmented(
    ground_truth_dir: Path,
    output_dir: Path,
    augmentations_per_image: int = 50,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    keep_originals: bool = False,
    backup_existing: bool = False,
    seed: int = 42,
) -> None:
    """Generate augmented dataset from ground-truth images"""
    validate_ratios(train_ratio, val_ratio, test_ratio)

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("PREPARING AUGMENTED GROUND-TRUTH DATASET")
    print("=" * 70)
    print(f"Ground-truth source: {ground_truth_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentations per image: {augmentations_per_image}")
    print(f"Keep originals: {keep_originals}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Random seed: {seed}")
    print()

    # Backup if requested
    if backup_existing:
        backup_existing_data(output_dir, seed)

    # Find ground-truth images and labels
    images_dir = ground_truth_dir
    labels_dir = ground_truth_dir / "labels"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_files = sorted(images_dir.glob("*.jpg"))

    # Filter to only images with labels
    image_label_pairs = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            image_label_pairs.append((img_path, label_path))
        else:
            print(f"⚠️  No label for: {img_path.name}, skipping")

    print(f"Found {len(image_label_pairs)} ground-truth image-label pairs")
    print()

    # Generate all augmented samples
    print("Generating augmented samples...")
    all_samples = []

    for img_path, label_path in image_label_pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load {img_path.name}, skipping")
            continue

        boxes = load_yolo_labels(label_path)
        stem = img_path.stem

        # Keep original if requested
        if keep_originals:
            all_samples.append((img.copy(), boxes.copy(), stem, "original"))

        # Generate augmentations
        for i in range(augmentations_per_image):
            aug_img, aug_boxes, aug_name = apply_random_augmentation(img, boxes)
            all_samples.append((aug_img, aug_boxes, stem, f"aug{i:03d}_{aug_name}"))

    print(f"Generated {len(all_samples)} total samples (originals + augmentations)")
    print()

    # Split into train/val/test
    train_samples, val_samples, test_samples = split_samples(
        all_samples, train_ratio, val_ratio, test_ratio, seed
    )

    n_total = len(all_samples)
    print(f"Split sizes:")
    print(
        f"  Training:   {len(train_samples):6d} ({len(train_samples)/n_total*100:.1f}%)"
    )
    print(f"  Validation: {len(val_samples):6d} ({len(val_samples)/n_total*100:.1f}%)")
    print(
        f"  Testing:    {len(test_samples):6d} ({len(test_samples)/n_total*100:.1f}%)"
    )
    print()

    # Save splits
    splits = {
        "training": train_samples,
        "validation": val_samples,
        "testing": test_samples,
    }

    print("Saving augmented dataset...")
    for split_name, samples in splits.items():
        split_img_dir, split_lbl_dir = save_split(split_name, samples, output_dir)

        print(f"  Writing {split_name} split ({len(samples)} samples)...")
        for idx, (img, boxes, stem, aug_suffix) in enumerate(samples):
            # Save image
            img_filename = f"{stem}_{aug_suffix}.jpg"
            img_path = split_img_dir / img_filename
            cv2.imwrite(str(img_path), img)

            # Save labels
            label_filename = f"{stem}_{aug_suffix}.txt"
            label_path = split_lbl_dir / label_filename
            save_yolo_labels(boxes, label_path)

        print(f"  ✓ {len(samples)} samples written to {split_name}/")

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Ground-truth images used: {len(image_label_pairs)}")
    print(f"Total augmented samples:  {len(all_samples)}")
    print()
    print("Dataset splits:")
    print(f"  Training:   {output_dir / 'training'}")
    print(f"  Validation: {output_dir / 'validation'}")
    print(f"  Testing:    {output_dir / 'testing'}")
    print("=" * 70)


# ============================================================================
# Mode 3: Parquet Extraction
# ============================================================================


def extract_image_from_parquet(image_data) -> "Image.Image":
    """Extract PIL Image from parquet row data"""
    if isinstance(image_data, dict):
        image_bytes = image_data.get("bytes")
    else:
        image_bytes = image_data

    return Image.open(io.BytesIO(image_bytes))


def create_yolo_label(label, image_width: int, image_height: int) -> Optional[str]:
    """Create YOLO format label for document classification"""
    # Default to body class (1) for all images
    class_id = 1

    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
    # All values normalized 0-1
    x_center = 0.5
    y_center = 0.5
    width = 1.0
    height = 1.0

    return f"{class_id} {x_center} {y_center} {width} {height}"


def process_parquet_file(
    parquet_path: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    split_name: str,
    limit: Optional[int] = None,
) -> int:
    """Process a single parquet file and extract images/labels"""
    print(f"\nProcessing: {parquet_path.name}")

    df = pd.read_parquet(parquet_path)

    if limit and limit < len(df):
        df = df.head(limit)
        print(f"  Limited to {limit} images")

    print(f"  Processing {len(df)} images...")

    processed = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        try:
            image = extract_image_from_parquet(row["image"])
            image_filename = f"{parquet_path.stem}_{idx:06d}.jpg"
            image_path = output_images_dir / image_filename

            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(image_path, "JPEG", quality=95)

            label = row.get("label", "invoice")
            yolo_label = create_yolo_label(label, image.width, image.height)

            if yolo_label:
                label_filename = f"{parquet_path.stem}_{idx:06d}.txt"
                label_path = output_labels_dir / label_filename
                label_path.write_text(yolo_label)

            processed += 1

        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            continue

    return processed


def prepare_parquet(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    limit_per_file: Optional[int] = None,
    backup_existing: bool = False,
    seed: int = 42,
) -> None:
    """Prepare YOLO dataset from parquet files"""
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "Pandas and PIL are required for parquet mode. "
            "Install with: pip install pandas pillow tqdm"
        )

    validate_ratios(train_ratio, val_ratio, test_ratio)

    # Find all parquet files
    parquet_files = list(raw_dir.rglob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {raw_dir}")
        return

    print("=" * 70)
    print("DATASET PREPARATION FROM PARQUET FILES")
    print("=" * 70)
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(parquet_files)} parquet file(s)")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print()

    # Backup if requested
    if backup_existing:
        backup_existing_data(output_dir, seed)

    # Create output directories
    splits = {
        "training": train_ratio,
        "validation": val_ratio,
        "testing": test_ratio,
    }

    for split_name in splits.keys():
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # Process each parquet file
    total_processed = 0
    split_counts = {split: 0 for split in splits.keys()}

    for i, parquet_path in enumerate(parquet_files):
        # Determine split based on file index
        cumulative_ratio = 0
        current_split = "training"

        for split_name, ratio in splits.items():
            cumulative_ratio += ratio
            if (i / len(parquet_files)) < cumulative_ratio:
                current_split = split_name
                break

        output_images_dir = output_dir / current_split / "images"
        output_labels_dir = output_dir / current_split / "labels"

        processed = process_parquet_file(
            parquet_path,
            output_images_dir,
            output_labels_dir,
            current_split,
            limit_per_file,
        )

        total_processed += processed
        split_counts[current_split] += processed

    # Print summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Total images processed: {total_processed}")
    print("\nSplit distribution:")
    for split_name, count in split_counts.items():
        print(f"  {split_name:12s}: {count:6d} images")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified dataset preparation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split ground-truth without augmentation
  python prepare_dataset.py groundtruth

  # Generate augmented dataset (recommended)
  python prepare_dataset.py augmented --augmentations-per-image 50

  # Extract from parquet files
  python prepare_dataset.py parquet --raw-dir data/raw
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Dataset preparation mode")

    # ========================================================================
    # Groundtruth subcommand
    # ========================================================================
    groundtruth_parser = subparsers.add_parser(
        "groundtruth",
        help="Split ground-truth images into train/val/test (no augmentation)",
    )
    groundtruth_parser.add_argument(
        "--ground-truth-dir",
        type=str,
        default="data/input/ground-truth",
        help="Directory containing ground-truth images and labels/",
    )
    groundtruth_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/input",
        help="Output base directory",
    )
    groundtruth_parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio"
    )
    groundtruth_parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    groundtruth_parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio"
    )
    groundtruth_parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Backup existing training data",
    )
    groundtruth_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ========================================================================
    # Augmented subcommand
    # ========================================================================
    augmented_parser = subparsers.add_parser(
        "augmented",
        help="Generate augmented dataset from ground-truth images",
    )
    augmented_parser.add_argument(
        "--ground-truth-dir",
        type=str,
        default="data/input/ground-truth",
        help="Directory containing ground-truth images and labels/",
    )
    augmented_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/input",
        help="Output base directory",
    )
    augmented_parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=50,
        help="Number of augmented versions per ground-truth image",
    )
    augmented_parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio"
    )
    augmented_parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    augmented_parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio"
    )
    augmented_parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Include original images in addition to augmented versions",
    )
    augmented_parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Backup existing training data",
    )
    augmented_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ========================================================================
    # Parquet subcommand
    # ========================================================================
    parquet_parser = subparsers.add_parser(
        "parquet",
        help="Extract dataset from parquet files",
    )
    parquet_parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing parquet files",
    )
    parquet_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/input",
        help="Output base directory",
    )
    parquet_parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio"
    )
    parquet_parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    parquet_parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio"
    )
    parquet_parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Limit number of images per parquet file",
    )
    parquet_parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Backup existing training data",
    )
    parquet_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent

    # Execute appropriate mode
    if args.mode == "groundtruth":
        ground_truth_dir = repo_root / args.ground_truth_dir
        output_dir = repo_root / args.output_dir

        if not ground_truth_dir.exists():
            raise FileNotFoundError(
                f"Ground-truth directory not found: {ground_truth_dir}"
            )

        prepare_groundtruth(
            ground_truth_dir=ground_truth_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            backup_existing=args.backup_existing,
            seed=args.seed,
        )

    elif args.mode == "augmented":
        ground_truth_dir = repo_root / args.ground_truth_dir
        output_dir = repo_root / args.output_dir

        if not ground_truth_dir.exists():
            raise FileNotFoundError(
                f"Ground-truth directory not found: {ground_truth_dir}"
            )

        prepare_augmented(
            ground_truth_dir=ground_truth_dir,
            output_dir=output_dir,
            augmentations_per_image=args.augmentations_per_image,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            keep_originals=args.keep_originals,
            backup_existing=args.backup_existing,
            seed=args.seed,
        )

    elif args.mode == "parquet":
        raw_dir = repo_root / args.raw_dir
        output_dir = repo_root / args.output_dir

        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

        prepare_parquet(
            raw_dir=raw_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            limit_per_file=args.limit_per_file,
            backup_existing=args.backup_existing,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
