#!/usr/bin/env python3
from PIL import Image
import glob, os

for split in ["training", "validation"]:
    for f in glob.glob(f"/home/joehiggi/siads-699/data/input/{split}/images/*.jpg"):
        try:
            img = Image.open(f).convert("RGB")
            img.save(f, "JPEG", quality=95)
        except Exception as e:
            print("Failed to reencode:", f, e)
print("✅  Re-encoding complete")

