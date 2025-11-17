#!/usr/bin/env python3
"""
Standalone script to convert COCO to YOLO and augment the dataset.
Uses only standard library + cv2 and numpy (which are widely available).
"""

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ========================================================================
# COCO TO YOLO CONVERSION
# ========================================================================


def coco_to_yolo(bbox, img_width, img_height):
    """Convert COCO bbox to YOLO format"""
    x_min, y_min, width, height = bbox

    # Calculate center and normalize
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    return x_center, y_center, width_norm, height_norm


def convert_coco_to_yolo_labels(coco_json_path, images_dir):
    """Convert COCO annotations and create YOLO .txt files"""
    print("=" * 70)
    print("STEP 1: Converting COCO to YOLO format")
    print("=" * 70)

    # Class mapping: COCO category_id -> YOLO class_id
    # COCO: 0=body, 1=footer, 2=header, 3=vertical_num
    # YOLO: 0=header, 1=body, 2=footer
    class_mapping = {0: 1, 1: 2, 2: 0}  # Skip vertical_num

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    converted = 0
    for img_id, img in images.items():
        file_name = Path(img["file_name"]).name
        img_width = img["width"]
        img_height = img["height"]

        # Check if image exists
        image_path = images_dir / file_name
        if not image_path.exists():
            continue

        # Get annotations
        img_annotations = annotations_by_image.get(img_id, [])
        if not img_annotations:
            continue

        # Convert to YOLO format
        yolo_labels = []
        for ann in img_annotations:
            category_id = ann["category_id"]
            if category_id not in class_mapping:
                continue

            yolo_class_id = class_mapping[category_id]
            bbox = ann["bbox"]
            x_center, y_center, width, height = coco_to_yolo(
                bbox, img_width, img_height
            )
            yolo_labels.append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if not yolo_labels:
            continue

        # Write YOLO label file
        label_file = image_path.with_suffix(".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

        converted += 1

    print(f"✓ Converted {converted} images to YOLO format")
    print()
    return converted


# ========================================================================
# DATA AUGMENTATION
# ========================================================================


class BoundingBox:
    def __init__(self, class_id, x_center, y_center, width, height):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def to_corners(self, img_width, img_height):
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
    def from_corners(class_id, x_min, y_min, x_max, y_max, img_width, img_height):
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))

        width_px = x_max - x_min
        height_px = y_max - y_min
        x_center_px = x_min + width_px / 2
        y_center_px = y_min + height_px / 2

        x_center = x_center_px / img_width
        y_center = y_center_px / img_height
        width = width_px / img_width
        height = height_px / img_height

        return BoundingBox(class_id, x_center, y_center, width, height)

    def to_yolo_string(self):
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def is_valid(self):
        return self.width > 0.001 and self.height > 0.001


def load_yolo_labels(label_path):
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


def save_yolo_labels(boxes, output_path):
    with open(output_path, "w") as f:
        for box in boxes:
            if box.is_valid():
                f.write(box.to_yolo_string() + "\n")


def horizontal_flip(img, boxes):
    flipped_img = cv2.flip(img, 1)
    flipped_boxes = []
    for box in boxes:
        new_box = BoundingBox(
            box.class_id, 1.0 - box.x_center, box.y_center, box.width, box.height
        )
        flipped_boxes.append(new_box)
    return flipped_img, flipped_boxes


def rotate_image_90(img, boxes, k=1):
    rotated_img = np.rot90(img, k=-k)
    rotated_boxes = []
    h, w = img.shape[:2]

    for box in boxes:
        x_min, y_min, x_max, y_max = box.to_corners(w, h)

        if k == 1:  # 90° CW
            new_x_min, new_y_min, new_x_max, new_y_max = (
                y_min,
                w - x_max,
                y_max,
                w - x_min,
            )
            new_w, new_h = h, w
        elif k == 2:  # 180°
            new_x_min, new_y_min, new_x_max, new_y_max = (
                w - x_max,
                h - y_max,
                w - x_min,
                h - y_min,
            )
            new_w, new_h = w, h
        elif k == 3:  # 270° CW
            new_x_min, new_y_min, new_x_max, new_y_max = (
                h - y_max,
                x_min,
                h - y_min,
                x_max,
            )
            new_w, new_h = h, w
        else:
            continue

        new_box = BoundingBox.from_corners(
            box.class_id, new_x_min, new_y_min, new_x_max, new_y_max, new_w, new_h
        )
        rotated_boxes.append(new_box)

    return rotated_img, rotated_boxes


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_random_augmentation(img, boxes):
    aug_type = random.choice(
        ["hflip", "rot90", "rot180", "rot270", "bright", "dark", "contrast"]
    )

    if aug_type == "hflip":
        return *horizontal_flip(img, boxes), "hflip"
    elif aug_type == "rot90":
        return *rotate_image_90(img, boxes, k=1), "rot90"
    elif aug_type == "rot180":
        return *rotate_image_90(img, boxes, k=2), "rot180"
    elif aug_type == "rot270":
        return *rotate_image_90(img, boxes, k=3), "rot270"
    elif aug_type == "bright":
        return adjust_brightness_contrast(img, alpha=1.0, beta=30), boxes, "bright"
    elif aug_type == "dark":
        return adjust_brightness_contrast(img, alpha=1.0, beta=-30), boxes, "dark"
    elif aug_type == "contrast":
        return adjust_brightness_contrast(img, alpha=1.3, beta=0), boxes, "contrast"

    return img, boxes, "none"


def augment_dataset(
    input_dir, output_dir, augmentations_per_image=5, copy_originals=True, seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 2: Augmenting dataset")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per image: {augmentations_per_image}")
    print()

    # Find all images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    image_files = [f for f in image_files if f.stem != "labels"]  # Skip labels.txt

    if not image_files:
        print("No images found!")
        return

    print(f"Found {len(image_files)} images")
    print()

    total_generated = 0
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load {img_path.name}")
            continue

        # Load labels
        label_path = img_path.with_suffix(".txt")
        boxes = load_yolo_labels(label_path)

        if not boxes:
            print(f"⚠️  No labels for {img_path.name}")
            continue

        stem = img_path.stem
        ext = img_path.suffix

        # Copy originals if requested
        if copy_originals:
            shutil.copy(img_path, output_dir / img_path.name)
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

        if total_generated % 50 == 0:
            print(f"  Generated {total_generated} augmented images...")

    print()
    print("=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Original images       : {len(image_files)}")
    print(f"Augmented images      : {total_generated}")
    if copy_originals:
        print(f"Total images created  : {total_generated + len(image_files)}")
    else:
        print(f"Total images created  : {total_generated}")
    print(f"Output directory      : {output_dir}")
    print("=" * 70)


# ========================================================================
# MAIN
# ========================================================================


def main():
    print("\n" + "=" * 70)
    print("DATASET AUGMENTATION PIPELINE")
    print("=" * 70)
    print()

    # Paths
    ground_truth_dir = Path("data/input/ground-truth")
    coco_json = ground_truth_dir / "labels.txt"
    output_dir = Path("data/input/ground-truth-augmented")

    # Step 1: Convert COCO to YOLO
    converted = convert_coco_to_yolo_labels(coco_json, ground_truth_dir)

    if converted == 0:
        print("❌ No images converted. Exiting.")
        sys.exit(1)

    # Step 2: Augment dataset
    augment_dataset(
        ground_truth_dir,
        output_dir,
        augmentations_per_image=5,
        copy_originals=True,
        seed=42,
    )

    print("\n✓ All done! Your augmented dataset is ready for training.")
    print(f"\nNext steps:")
    print(f"  1. Verify the output: ls {output_dir}")
    print(f"  2. Use this augmented data for training")


if __name__ == "__main__":
    try:
        import cv2
        import numpy
    except ImportError:
        print("ERROR: This script requires cv2 and numpy")
        print("Install with: pip install opencv-python numpy")
        sys.exit(1)

    main()
