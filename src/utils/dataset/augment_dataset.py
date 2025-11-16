#!/usr/bin/env python3
"""
Data augmentation script for YOLO format datasets.
Applies various transformations to images and automatically adjusts bounding boxes.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BoundingBox,
    load_yolo_labels,
    save_yolo_labels,
    horizontal_flip,
    adjust_brightness_contrast,
    add_gaussian_noise,
    adjust_hsv,
    adjust_background_color,
    rotate_image_90,
    random_crop_with_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment YOLO dataset with transformations (rotation, flip, brightness, etc.)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing images and .txt label files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for augmented dataset",
    )
    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=5,
        help="Number of augmented versions to create per image (default: 5)",
    )
    parser.add_argument(
        "--copy-originals",
        action="store_true",
        help="Copy original images to output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def apply_random_augmentation(img: np.ndarray, boxes: List[BoundingBox]) -> Tuple[np.ndarray, List[BoundingBox], str]:
    """Apply a random augmentation to the image and boxes"""
    aug_type = random.choice([
        "horizontal_flip",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "brightness_up",
        "brightness_down",
        "contrast_up",
        "contrast_down",
        "noise",
        "hsv_shift",
        "crop",
        "bg_white",
        "bg_cream",
        "bg_gray",
        "bg_normalize",
    ])

    if aug_type == "horizontal_flip":
        return *horizontal_flip(img, boxes), "hflip"

    elif aug_type == "rotate_90":
        return *rotate_image_90(img, boxes, k=1), "rot90"

    elif aug_type == "rotate_180":
        return *rotate_image_90(img, boxes, k=2), "rot180"

    elif aug_type == "rotate_270":
        return *rotate_image_90(img, boxes, k=3), "rot270"

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

    elif aug_type == "crop":
        crop_ratio = random.uniform(0.85, 0.95)
        aug_img, aug_boxes = random_crop_with_boxes(img, boxes, crop_ratio)
        return aug_img, aug_boxes, "crop"

    elif aug_type == "bg_white":
        return adjust_background_color(img, "white"), boxes, "bgwhite"

    elif aug_type == "bg_cream":
        return adjust_background_color(img, "cream"), boxes, "bgcream"

    elif aug_type == "bg_gray":
        return adjust_background_color(img, "gray"), boxes, "bggray"

    elif aug_type == "bg_normalize":
        return adjust_background_color(img, "normalize"), boxes, "bgnorm"

    return img, boxes, "none"


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    augmentations_per_image: int = 5,
    copy_originals: bool = False,
    seed: int = 42,
) -> None:
    """Augment all images in input directory"""
    random.seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Generating {augmentations_per_image} augmentations per image...")
    print(f"Output directory: {output_dir}")
    print()

    total_generated = 0

    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path.name}, skipping...")
            continue

        # Load labels
        label_path = img_path.with_suffix(".txt")
        boxes = load_yolo_labels(label_path)

        stem = img_path.stem
        ext = img_path.suffix

        # Copy originals if requested
        if copy_originals:
            shutil.copy(img_path, output_dir / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, output_dir / label_path.name)

        # Generate augmentations
        for i in range(augmentations_per_image):
            aug_img, aug_boxes, aug_name = apply_random_augmentation(img, boxes)

            # Save augmented image and labels
            output_img_name = f"{stem}_aug{i}_{aug_name}{ext}"
            output_label_name = f"{stem}_aug{i}_{aug_name}.txt"

            cv2.imwrite(str(output_dir / output_img_name), aug_img)
            save_yolo_labels(aug_boxes, output_dir / output_label_name)

            total_generated += 1

        if (total_generated) % 50 == 0:
            print(f"  Generated {total_generated} augmented images...")

    print()
    print("=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Original images       : {len(image_files)}")
    print(f"Augmented images      : {total_generated}")
    print(f"Total images created  : {total_generated + (len(image_files) if copy_originals else 0)}")
    print(f"Output directory      : {output_dir}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augmentations_per_image=args.augmentations_per_image,
        copy_originals=args.copy_originals,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()