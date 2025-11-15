#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format for training.

Reads labels.txt (COCO format) and creates corresponding YOLO .txt files
for each image in the ground-truth directory.

COCO format: {x_min, y_min, width, height} in pixels
YOLO format: {class_id x_center y_center width height} normalized [0, 1]
"""

import json
import os
from pathlib import Path
from typing import Dict, List


def coco_to_yolo(bbox: List[float], img_width: int, img_height: int) -> tuple:
    """
    Convert COCO bbox format to YOLO format.

    Args:
        bbox: [x_min, y_min, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    x_min, y_min, width, height = bbox

    # Calculate center point
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)

    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # Clamp to [0, 1] to handle any edge cases
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_coco_to_yolo_labels(
    coco_json_path: Path,
    images_dir: Path,
    labels_dir: Path,
    class_mapping: Dict[int, int] = None,
) -> None:
    """
    Convert COCO annotations to YOLO format labels.

    Args:
        coco_json_path: Path to labels.txt (COCO format JSON)
        images_dir: Directory containing the images
        labels_dir: Output directory for YOLO labels
        class_mapping: Optional mapping from COCO category_id to YOLO class_id
                      If None, uses: {0: 1, 1: 2, 2: 0, 3: 3} (body->1, footer->2, header->0)
    """
    # Default class mapping: COCO categories to YOLO classes
    # COCO: 0=body, 1=footer, 2=header, 3=vertical_num
    # YOLO: 0=header, 1=body, 2=footer (we'll ignore vertical_num for now)
    if class_mapping is None:
        class_mapping = {
            0: 1,  # body -> class 1
            1: 2,  # footer -> class 2
            2: 0,  # header -> class 0
            # 3: skip vertical_num (not in our 3-class model)
        }

    print("=" * 70)
    print("Converting COCO annotations to YOLO format")
    print("=" * 70)
    print(f"Input JSON: {coco_json_path}")
    print(f"Images dir: {images_dir}")
    print(f"Labels dir: {labels_dir}")
    print(f"Class mapping: {class_mapping}")
    print()

    # Load COCO annotations
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    print(
        f"Loaded {len(images)} images, {len(annotations)} annotations, {len(categories)} categories"
    )
    print(f"Categories: {[(c['id'], c['name']) for c in categories]}")
    print()

    # Create image_id to image info mapping
    image_info = {img["id"]: img for img in images}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Create labels directory
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Convert each image's annotations
    converted_count = 0
    skipped_count = 0
    total_labels = 0

    for img_id, img in image_info.items():
        # Extract image filename from path
        file_name = Path(img["file_name"]).name
        img_width = img["width"]
        img_height = img["height"]

        # Check if image exists
        image_path = images_dir / file_name
        if not image_path.exists():
            print(f"⚠️  Image not found: {file_name}")
            skipped_count += 1
            continue

        # Get annotations for this image
        img_annotations = annotations_by_image.get(img_id, [])

        if not img_annotations:
            print(f"⚠️  No annotations for: {file_name}")
            skipped_count += 1
            continue

        # Convert annotations to YOLO format
        yolo_labels = []
        for ann in img_annotations:
            category_id = ann["category_id"]

            # Skip categories not in our mapping (e.g., vertical_num)
            if category_id not in class_mapping:
                continue

            yolo_class_id = class_mapping[category_id]
            bbox = ann["bbox"]

            # Convert to YOLO format
            x_center, y_center, width, height = coco_to_yolo(
                bbox, img_width, img_height
            )

            yolo_labels.append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if not yolo_labels:
            print(f"⚠️  No valid labels after filtering: {file_name}")
            skipped_count += 1
            continue

        # Write YOLO label file
        label_file = labels_dir / file_name.replace(".jpg", ".txt").replace(
            ".png", ".txt"
        )
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

        converted_count += 1
        total_labels += len(yolo_labels)

    print("=" * 70)
    print("Conversion complete!")
    print("=" * 70)
    print(f"✓ Converted: {converted_count} images")
    print(f"✓ Total labels: {total_labels}")
    print(f"⚠️  Skipped: {skipped_count} images")
    print(f"✓ Average labels per image: {total_labels / converted_count:.1f}")
    print(f"\nYOLO labels saved to: {labels_dir}")
    print("=" * 70)


def main():
    # Determine repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent

    # Paths
    ground_truth_dir = repo_root / "data" / "input" / "ground-truth"
    coco_json = ground_truth_dir / "labels.txt"
    images_dir = ground_truth_dir
    labels_dir = ground_truth_dir / "labels"

    # Convert
    convert_coco_to_yolo_labels(coco_json, images_dir, labels_dir)

    print("\nNext steps:")
    print("1. Verify a few label files:")
    print(f"   python src/utils/dataset/preview_yolo_labels.py {images_dir} --limit 5")
    print("2. Copy ground-truth data to training/validation splits")
    print("3. Run training with actual annotations!")


if __name__ == "__main__":
    main()
