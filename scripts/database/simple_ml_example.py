"""
SIMPLE EXAMPLE:

1. Metadata in PostgreSQL (fast queries) → tells you WHERE images are
2. Images in Parquet files (efficient storage) → loaded on-demand during training
3. Best of both worlds: SQL filtering + efficient image loading
"""
import pandas as pd
from PIL import Image
from io import BytesIO
from sqlalchemy import create_engine

# PART 1: Setup - Load metadata only (FAST!)

print("=" * 70)
print("PART 1: Query metadata from PostgreSQL")
print("=" * 70)

# Connect to database
engine = create_engine("postgresql://postgres:123@localhost:5432/postgres")

# Query metadata - this is VERY fast (no image bytes!)
query = """
SELECT id, parquet_file, row_index, label
FROM document_metadata
LIMIT 10
"""

metadata = pd.read_sql(query, engine)
print("\nMetadata (first 10 rows):")
print(metadata)
print(f"\nQuery took milliseconds (no images loaded yet!)")

# PART 2: Load a specific image when you need it

print("\n" + "=" * 70)
print("PART 2: Load specific image from parquet file on-demand")
print("=" * 70)

# Get info about first image
first_row = metadata.iloc[0]
print(f"\nLoading image:")
print(f"  File: {first_row['parquet_file']}")
print(f"  Row:  {first_row['row_index']}")
print(f"  Label: {first_row['label']}")

# Load the parquet file
parquet_path = f"/workspace/data/raw/{first_row['parquet_file']}"
df = pd.read_parquet(parquet_path)

# Get the specific image
image_data = df.iloc[first_row["row_index"]]["image"]
image_bytes = image_data["bytes"] if isinstance(image_data, dict) else image_data

# Convert to PIL Image
image = Image.open(BytesIO(image_bytes))
print(f"\n✓ Loaded image: {image.size} pixels, mode: {image.mode}")


# PART 4: Common ML queries

print("=" * 70)
print("PART 4: Useful SQL queries for ML")
print("=" * 70)

# Count samples per label
print("\n1. Class distribution:")
query = "SELECT label, COUNT(*) as count FROM document_metadata GROUP BY label ORDER BY label"
print(pd.read_sql(query, engine))

# Random sample for quick testing
print("\n2. Random sample of 5 images:")
query = "SELECT * FROM document_metadata ORDER BY RANDOM() LIMIT 5"
print(pd.read_sql(query, engine)[["id", "parquet_file", "row_index", "label"]])

# Filter by label
print("\n3. Get 3 samples from label 6:")
query = "SELECT * FROM document_metadata WHERE label = 6 LIMIT 3"
print(pd.read_sql(query, engine)[["id", "parquet_file", "row_index", "label"]])

print("\n" + "=" * 70)
print("✓ Option 2 is perfect for ML workflows!")
print("=" * 70)
