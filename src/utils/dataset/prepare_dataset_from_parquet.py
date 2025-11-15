#!/usr/bin/env python3
"""
Prepare YOLO dataset from parquet files.
Extracts images and creates YOLO format labels from parquet data.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _find_repo_root() -> Path:
    """Find the repository root by looking for marker directories."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (
            (current / "src").exists()
            and (current / "data").exists()
            and (current / "models").exists()
        ):
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root.")


REPO_ROOT = _find_repo_root()
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INPUT_DIR = DATA_DIR / "input"


def extract_image_from_parquet(image_data) -> Image.Image:
    """
    Extract PIL Image from parquet row data.

    Args:
        image_data: Image data from parquet (dict or bytes)

    Returns:
        PIL Image object
    """
    if isinstance(image_data, dict):
        image_bytes = image_data.get("bytes")
    else:
        image_bytes = image_data

    return Image.open(io.BytesIO(image_bytes))


def create_yolo_label(label, image_width: int, image_height: int) -> Optional[str]:
    """
    Create YOLO format label for document classification.
    For now, creates a single bounding box covering the entire image.

    Args:
        label: Label (int or str)
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        YOLO format label string or None
    """
    # Default to body class (1) for all images since parquet doesn't have bbox annotations
    # TODO: Replace this with actual bounding box annotations from Label Studio
    # The parquet label is document type (invoice=6), not region type (header/body/footer)
    class_id = 1  # Body class - default for now

    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
    # All values normalized 0-1
    # For document-level classification, use full image bbox
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
    """
    Process a single parquet file and extract images/labels.

    Args:
        parquet_path: Path to parquet file
        output_images_dir: Directory to save images
        output_labels_dir: Directory to save YOLO labels
        split_name: Name of the split (train/val/test)
        limit: Optional limit on number of images to process

    Returns:
        Number of images processed
    """
    print(f"\nProcessing: {parquet_path.name}")

    # Read parquet file
    df = pd.read_parquet(parquet_path)

    if limit and limit < len(df):
        df = df.head(limit)
        print(f"  Limited to {limit} images")

    print(f"  Processing {len(df)} images...")

    processed = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        try:
            # Extract image
            image = extract_image_from_parquet(row["image"])

            # Save image
            image_filename = f"{parquet_path.stem}_{idx:06d}.jpg"
            image_path = output_images_dir / image_filename

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(image_path, "JPEG", quality=95)

            # Create and save YOLO label
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


def prepare_dataset(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = INPUT_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    limit_per_file: Optional[int] = None,
) -> None:
    """
    Prepare YOLO dataset from all parquet files.

    Args:
        raw_dir: Directory containing parquet files
        output_dir: Base output directory
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        limit_per_file: Optional limit on images per parquet file
    """
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
    print("\nNext steps:")
    print("  1. Verify images and labels in data/input/")
    print("  2. Run training: python src/training/train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from parquet files."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DIR),
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(INPUT_DIR),
        help="Base output directory for dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of data for training. Default: 0.7",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of data for validation. Default: 0.15",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of data for testing. Default: 0.15",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Limit number of images per parquet file (for testing).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_dataset(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        limit_per_file=args.limit_per_file,
    )


if __name__ == "__main__":
    main()
