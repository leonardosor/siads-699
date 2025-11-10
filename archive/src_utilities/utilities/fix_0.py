#!/usr/bin/env python3
import cv2, os, glob
from PIL import Image

def bad_images_in(path):
    bad = []
    for f in glob.glob(os.path.join(path, "*.jpg")):
        try:
            img_cv = cv2.imread(f)
            if img_cv is None or img_cv.size == 0:
                raise ValueError("Unreadable by OpenCV")
            Image.open(f).verify()
        except Exception as e:
            bad.append((f, str(e)))
    return bad

for split in ["training", "validation"]:
    path = f"/home/joehiggi/siads-699/data/input/{split}/images"
    bad = bad_images_in(path)
    if bad:
        print(f"⚠️  {len(bad)} bad {split} images found — deleting:")
        for b, msg in bad:
            print("   ", b, "→", msg)
            os.remove(b)
    else:
        print(f"✅  All {split} images passed both OpenCV and PIL checks")

