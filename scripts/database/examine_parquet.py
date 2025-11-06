"""Quick script to examine parquet file structure"""
import pandas as pd

# Read first parquet file
df = pd.read_parquet("/workspace/data/raw/train-00000-of-00005.parquet")

print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nColumn types:")
print(df.dtypes)

if "label" in df.columns:
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

# Check image column structure
if "image" in df.columns:
    first_image = df["image"].iloc[0]
    if isinstance(first_image, dict):
        print(f"\nImage is stored as dict with keys: {list(first_image.keys())}")
        if "bytes" in first_image:
            print(f'First image size: {len(first_image["bytes"])} bytes')
    else:
        print(f"\nImage type: {type(first_image)}")
        print(f"First image size: {len(first_image)} bytes")
