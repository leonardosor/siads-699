# ================================================================
# COCO Image Visualization Script
# ------------------------------------------------
# This script was collaboratively generated with assistance from
# ChatGPT (OpenAI, 2025), an AI language model providing
# code synthesis, explanation, and documentation support.
# ================================================================

import json
import os
import ntpath
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ======================
# PATH CONFIGURATION
# ======================
BASE_DIR = "./data/input/main-annotated"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
ANNOTATION_PATH = os.path.join(BASE_DIR, "result.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "annotated")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# LOAD ANNOTATIONS
# ======================
with open(ANNOTATION_PATH, "r") as f:
    coco_data = json.load(f)

# Normalize file paths from Label Studio or Windows
for img in coco_data["images"]:
    img["file_name"] = ntpath.basename(img["file_name"])

# Build lookup for images
images = {img["id"]: img for img in coco_data["images"]}

# ======================
# VISUALIZATION LOOP
# ======================
for img_id, img_info in images.items():
    img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
    if not os.path.exists(img_path):
        print(f"Skipping missing file: {img_info['file_name']}")
        continue

    # Load image
    img = Image.open(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()

    # Draw bounding boxes
    anns_for_image = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == img_id
    ]
    for ann in anns_for_image:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)

        # Draw category label if available
        if "categories" in coco_data:
            category = next(
                (
                    c["name"]
                    for c in coco_data["categories"]
                    if c["id"] == ann["category_id"]
                ),
                "object",
            )
            plt.text(x, y - 5, category, color="lime", fontsize=10, weight="bold")

    plt.axis("off")
    plt.title(img_info["file_name"])

    # Save annotated image
    output_path = os.path.join(OUTPUT_DIR, img_info["file_name"])
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved annotated image: {output_path}")

print("✅ Annotation visualization complete.")
