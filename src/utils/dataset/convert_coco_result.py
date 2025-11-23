#!/usr/bin/env python3
"""
Convert result.json (COCO format) to YOLO format labels.
Extracts annotations from the COCO JSON and creates matching .txt files for each image.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def coco_to_yolo(bbox: List[float], img_width: int, img_height: int) -> tuple:
    """
    Convert COCO bbox format to YOLO format.

    COCO: [x_min, y_min, width, height] in pixels
    YOLO: [x_center, y_center, width, height] normalized to [0, 1]
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

    # Clamp to [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    return x_center_norm, y_center_norm, width_norm, height_norm


def main():
    # Paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    ground_truth_dir = repo_root / "data" / "input" / "ground-truth"
    result_json = ground_truth_dir / "results_c_fixed.json"  # Complete annotations for all 165 images
    labels_dir = ground_truth_dir / "labels"

    if not result_json.exists():
        print(f"Error: {result_json} not found")
        sys.exit(1)

    # Load COCO format annotations
    print(f"Loading {result_json}")
    with open(result_json, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    print(f"Found {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")

    # Category mapping: COCO category_id -> YOLO class_id
    # COCO: 0=body, 1=footer, 2=header, 3=vertical_num
    # YOLO: 0=header, 1=body, 2=footer (skip vertical_num)
    class_mapping = {
        0: 1,  # body -> class 1
        1: 2,  # footer -> class 2
        2: 0,  # header -> class 0
        # 3: skip vertical_num (not in our 3-class model)
    }

    # Create image_id to image info mapping
    image_info: Dict[int, dict] = {img["id"]: img for img in images}

    # Group annotations by image_id
    annotations_by_image: Dict[int, List] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Create labels directory
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing labels
    print(f"\nClearing existing labels in {labels_dir}")
    for old_label in labels_dir.glob("*.txt"):
        old_label.unlink()

    # Convert each image's annotations
    print("\nConverting annotations to YOLO format...")
    converted_count = 0
    skipped_count = 0

    for img_id, img in image_info.items():
        # Extract just the filename from the path
        file_name = Path(img["file_name"]).name
        img_width = img["width"]
        img_height = img["height"]

        # Check if image exists
        image_path = ground_truth_dir / file_name
        if not image_path.exists():
            print(f"[WARN] Image not found: {file_name}")
            skipped_count += 1
            continue

        # Get annotations for this image
        img_annotations = annotations_by_image.get(img_id, [])

        if not img_annotations:
            print(f"[WARN] No annotations for: {file_name}")
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
            x_center, y_center, width, height = coco_to_yolo(bbox, img_width, img_height)

            yolo_labels.append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if not yolo_labels:
            print(f"[WARN] No valid labels after filtering: {file_name}")
            skipped_count += 1
            continue

        # Write YOLO label file
        label_filename = Path(file_name).stem + ".txt"
        label_file = labels_dir / label_filename

        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

        converted_count += 1
        print(f"[OK] {file_name} -> {label_filename} ({len(yolo_labels)} labels)")

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"[OK] Converted: {converted_count} images")
    print(f"[WARN] Skipped: {skipped_count} images")
    print(f"\nYOLO labels saved to: {labels_dir}")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Verify the labels were created correctly:")
    print(f"   ls {labels_dir} | wc -l")
    print("2. Prepare augmented dataset:")
    print("   python src/utils/dataset/prepare_dataset.py augmented --augmentations-per-image 50")


if __name__ == "__main__":
    main()
