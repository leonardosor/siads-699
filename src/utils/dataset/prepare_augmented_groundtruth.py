#!/usr/bin/env python3
"""
Prepare training dataset using ONLY augmented ground-truth images.
This script:
1. Takes the 100 ground-truth images with proper bounding box annotations
2. Generates augmented versions (with preserved bounding boxes)
3. Splits augmented data into train/val/test sets
4. Replaces the current training dataset
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare augmented ground-truth dataset for training"
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=str,
        default="data/input/ground-truth",
        help="Directory containing ground-truth images and labels/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/input",
        help="Output base directory (will create train/val/test subdirs)",
    )
    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=50,
        help="Number of augmented versions per ground-truth image (default: 50)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Include original images in addition to augmented versions",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Backup existing training data before replacing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


class BoundingBox:
    """YOLO format bounding box (class, x_center, y_center, width, height) - normalized [0, 1]"""

    def __init__(
        self, class_id: int, x_center: float, y_center: float, width: float, height: float
    ):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def to_corners(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert YOLO format to corner coordinates (x_min, y_min, x_max, y_max) in pixels"""
        x_center_px = self.x_center * img_width
        y_center_px = self.y_center * img_height
        width_px = self.width * img_width
        height_px = self.height * img_height

        x_min = x_center_px - width_px / 2
        y_min = y_center_px - height_px / 2
        x_max = x_center_px + width_px / 2
        y_max = y_center_px + height_px / 2

        return x_min, y_min, x_max, y_max

    @staticmethod
    def from_corners(
        class_id: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        img_width: int,
        img_height: int,
    ) -> "BoundingBox":
        """Create BoundingBox from corner coordinates (in pixels)"""
        # Clamp coordinates to image boundaries
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))

        # Calculate center and dimensions in normalized format
        width_px = x_max - x_min
        height_px = y_max - y_min
        x_center_px = x_min + width_px / 2
        y_center_px = y_min + height_px / 2

        x_center = x_center_px / img_width if img_width > 0 else 0.5
        y_center = y_center_px / img_height if img_height > 0 else 0.5
        width = width_px / img_width if img_width > 0 else 0
        height = height_px / img_height if img_height > 0 else 0

        return BoundingBox(class_id, x_center, y_center, width, height)

    def to_yolo_string(self) -> str:
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def is_valid(self) -> bool:
        """Check if bounding box is valid (has positive area)"""
        return self.width > 0.001 and self.height > 0.001


def load_yolo_labels(label_path: Path) -> List[BoundingBox]:
    """Load YOLO format labels from file"""
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append(BoundingBox(class_id, x_center, y_center, width, height))
    return boxes


def save_yolo_labels(boxes: List[BoundingBox], output_path: Path) -> None:
    """Save bounding boxes to YOLO format file"""
    with open(output_path, "w") as f:
        for box in boxes:
            if box.is_valid():
                f.write(box.to_yolo_string() + "\n")


def horizontal_flip(
    img: np.ndarray, boxes: List[BoundingBox]
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Flip image and boxes horizontally"""
    flipped_img = cv2.flip(img, 1)
    flipped_boxes = []

    for box in boxes:
        new_box = BoundingBox(
            box.class_id, 1.0 - box.x_center, box.y_center, box.width, box.height
        )
        flipped_boxes.append(new_box)

    return flipped_img, flipped_boxes


def adjust_brightness_contrast(
    img: np.ndarray, alpha: float = 1.0, beta: int = 0
) -> np.ndarray:
    """Adjust image brightness and contrast"""
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def add_gaussian_noise(img: np.ndarray, mean: float = 0, std: float = 10) -> np.ndarray:
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img


def adjust_hsv(
    img: np.ndarray, h_shift: int = 0, s_scale: float = 1.0, v_scale: float = 1.0
) -> np.ndarray:
    """Adjust hue, saturation, and value"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180  # Hue
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)  # Value

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


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
        # Apply multiple augmentations
        aug_img = img.copy()
        aug_boxes = boxes.copy()

        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            aug_img = adjust_brightness_contrast(aug_img, alpha=alpha, beta=beta)

        # Random HSV shift
        if random.random() > 0.5:
            h_shift = random.randint(-5, 5)
            s_scale = random.uniform(0.9, 1.1)
            v_scale = random.uniform(0.9, 1.1)
            aug_img = adjust_hsv(aug_img, h_shift, s_scale, v_scale)

        return aug_img, aug_boxes, "combined"

    return img, boxes, "none"


def prepare_augmented_groundtruth(
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
    """Generate augmented dataset from ground-truth images and split into train/val/test"""

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

    # Backup existing data if requested
    if backup_existing:
        for split in ["training", "validation", "testing"]:
            split_dir = output_dir / split
            if split_dir.exists():
                backup_dir = output_dir / f"{split}_backup_{seed}"
                print(f"Backing up {split_dir} -> {backup_dir}")
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(split_dir, backup_dir)

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

    # Generate all augmented samples first
    print("Generating augmented samples...")
    all_samples = []

    for img_path, label_path in image_label_pairs:
        # Load image and labels
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

    # Shuffle all samples
    random.shuffle(all_samples)

    # Split into train/val/test
    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train : n_train + n_val]
    test_samples = all_samples[n_train + n_val :]

    print(f"Split sizes:")
    print(f"  Training:   {len(train_samples):6d} ({len(train_samples)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_samples):6d} ({len(val_samples)/n_total*100:.1f}%)")
    print(f"  Testing:    {len(test_samples):6d} ({len(test_samples)/n_total*100:.1f}%)")
    print()

    # Save splits
    splits = {
        "training": train_samples,
        "validation": val_samples,
        "testing": test_samples,
    }

    print("Saving augmented dataset...")
    for split_name, samples in splits.items():
        split_img_dir = output_dir / split_name / "images"
        split_lbl_dir = output_dir / split_name / "labels"

        # Clear existing directories
        if split_img_dir.exists():
            shutil.rmtree(split_img_dir)
        if split_lbl_dir.exists():
            shutil.rmtree(split_lbl_dir)

        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

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
    print()
    print("Next steps:")
    print("  1. Verify dataset with:")
    print("     python src/training/train.py --data-config src/training/finance-image-parser.yaml")
    print("  2. Start training:")
    print("     python src/training/train.py --epochs 100 --batch 16 --cache")


def main() -> None:
    args = parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent

    ground_truth_dir = repo_root / args.ground_truth_dir
    output_dir = repo_root / args.output_dir

    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {ground_truth_dir}")

    prepare_augmented_groundtruth(
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


if __name__ == "__main__":
    main()
