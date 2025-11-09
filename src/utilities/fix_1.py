#!/usr/bin/env python3
import os, glob

for split in ["training", "validation"]:
    img_dir = f"/home/joehiggi/siads-699/data/input/{split}/images"
    lbl_dir = f"/home/joehiggi/siads-699/data/input/{split}/labels"
    imgs = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(img_dir, "*.jpg"))}
    lbls = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(lbl_dir, "*.txt"))}

    extra_lbls = lbls - imgs
    extra_imgs = imgs - lbls

    print(f"\n{split.upper()}:")
    if extra_lbls:
        print(f"  Labels without images ({len(extra_lbls)}):")
        for x in sorted(extra_lbls):
            print("    ", x + ".txt")
    if extra_imgs:
        print(f"  Images without labels ({len(extra_imgs)}):")
        for x in sorted(extra_imgs):
            print("    ", x + ".jpg")
    if not extra_lbls and not extra_imgs:
        print("  ✅  Perfect match between images and labels")

