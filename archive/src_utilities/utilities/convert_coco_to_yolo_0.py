# ================================================================
# COCO → YOLOv8 Minimal Conversion for All Splits
# ------------------------------------------------
# This script was collaboratively generated with assistance from
# ChatGPT (OpenAI, 2025) for academic demonstration purposes.
# ================================================================

import json, os

BASE = "/workspace/data/input"
SPLITS = ["training", "validation", "testing"]

for split in SPLITS:
    ann_path = f"{BASE}/{split}/annotations.json"
    lbl_path = f"{BASE}/{split}/labels"

    if not os.path.exists(ann_path):
        print(f"Skipping {split} — no annotations.json found.")
        continue

    os.makedirs(lbl_path, exist_ok=True)
    coco = json.load(open(ann_path))

    print(f"Converting {split} annotations...")
    for ann in coco["annotations"]:
        img = next(i for i in coco["images"] if i["id"] == ann["image_id"])
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]
        xc, yc = (x + bw / 2) / w, (y + bh / 2) / h
        bw, bh = bw / w, bh / h
        cls = ann["category_id"]
        name = os.path.splitext(img["file_name"])[0] + ".txt"

        with open(f"{lbl_path}/{name}", "a") as f:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ {split} done → {len(coco['annotations'])} boxes converted.\n")

