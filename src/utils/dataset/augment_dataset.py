#!/usr/bin/env python3
"""
Data augmentation script for YOLO format datasets.
Applies various transformations to images and automatically adjusts bounding boxes.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance


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


class BoundingBox:
    """YOLO format bounding box (class, x_center, y_center, width, height) - normalized [0, 1]"""

    def __init__(self, class_id: int, x_center: float, y_center: float, width: float, height: float):
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
        class_id: int, x_min: float, y_min: float, x_max: float, y_max: float,
        img_width: int, img_height: int
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

        x_center = x_center_px / img_width
        y_center = y_center_px / img_height
        width = width_px / img_width
        height = height_px / img_height

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


def horizontal_flip(img: np.ndarray, boxes: List[BoundingBox]) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Flip image and boxes horizontally"""
    flipped_img = cv2.flip(img, 1)
    flipped_boxes = []

    for box in boxes:
        # Flip x_center: new_x = 1 - old_x
        new_box = BoundingBox(
            box.class_id,
            1.0 - box.x_center,
            box.y_center,
            box.width,
            box.height
        )
        flipped_boxes.append(new_box)

    return flipped_img, flipped_boxes


def rotate_image_90(img: np.ndarray, boxes: List[BoundingBox], k: int = 1) -> Tuple[np.ndarray, List[BoundingBox]]:
    """
    Rotate image by 90 degrees k times clockwise
    k=1: 90° CW, k=2: 180°, k=3: 270° CW (90° CCW)
    """
    rotated_img = np.rot90(img, k=-k)  # OpenCV uses different convention
    rotated_boxes = []

    h, w = img.shape[:2]

    for box in boxes:
        # Convert to pixel coordinates
        x_min, y_min, x_max, y_max = box.to_corners(w, h)

        # Apply rotation transformation
        if k == 1:  # 90° CW
            new_x_min = y_min
            new_y_min = w - x_max
            new_x_max = y_max
            new_y_max = w - x_min
            new_w, new_h = h, w
        elif k == 2:  # 180°
            new_x_min = w - x_max
            new_y_min = h - y_max
            new_x_max = w - x_min
            new_y_max = h - y_min
            new_w, new_h = w, h
        elif k == 3:  # 270° CW (90° CCW)
            new_x_min = h - y_max
            new_y_min = x_min
            new_x_max = h - y_min
            new_y_max = x_max
            new_w, new_h = h, w
        else:
            continue

        new_box = BoundingBox.from_corners(
            box.class_id, new_x_min, new_y_min, new_x_max, new_y_max, new_w, new_h
        )
        rotated_boxes.append(new_box)

    return rotated_img, rotated_boxes


def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Adjust image brightness and contrast
    alpha: contrast (1.0 = no change, <1.0 = decrease, >1.0 = increase)
    beta: brightness (0 = no change, negative = darker, positive = brighter)
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def add_gaussian_noise(img: np.ndarray, mean: float = 0, std: float = 10) -> np.ndarray:
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img


def adjust_hsv(img: np.ndarray, h_shift: int = 0, s_scale: float = 1.0, v_scale: float = 1.0) -> np.ndarray:
    """Adjust hue, saturation, and value"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180  # Hue
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)  # Value

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_crop_with_boxes(
    img: np.ndarray, boxes: List[BoundingBox], crop_ratio: float = 0.9
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Randomly crop image while keeping all bounding boxes"""
    h, w = img.shape[:2]
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)

    # Calculate crop coordinates to keep all boxes
    max_x_start = 0
    max_y_start = 0
    min_x_end = w
    min_y_end = h

    for box in boxes:
        x_min, y_min, x_max, y_max = box.to_corners(w, h)
        max_x_start = max(max_x_start, x_max - new_w)
        max_y_start = max(max_y_start, y_max - new_h)
        min_x_end = min(min_x_end, x_min + new_w)
        min_y_end = min(min_y_end, y_min + new_h)

    # Ensure valid crop region
    if max_x_start >= min_x_end - new_w or max_y_start >= min_y_end - new_h:
        return img, boxes  # Can't crop without losing boxes

    x_start = random.randint(int(max_x_start), int(min_x_end - new_w))
    y_start = random.randint(int(max_y_start), int(min_y_end - new_h))

    cropped_img = img[y_start:y_start + new_h, x_start:x_start + new_w]

    # Adjust bounding boxes
    cropped_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box.to_corners(w, h)
        new_x_min = x_min - x_start
        new_y_min = y_min - y_start
        new_x_max = x_max - x_start
        new_y_max = y_max - y_start

        new_box = BoundingBox.from_corners(
            box.class_id, new_x_min, new_y_min, new_x_max, new_y_max, new_w, new_h
        )
        if new_box.is_valid():
            cropped_boxes.append(new_box)

    return cropped_img, cropped_boxes


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