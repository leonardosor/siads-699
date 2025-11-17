# Dataset Setup - Quick Reference

## Current Status
- ✅ 100 ground-truth images with proper bounding box annotations in `data/input/ground-truth/`
- ⚠️  Old training dataset (~16k images from parquet) needs to be replaced

## Setup Augmented Ground-Truth Dataset

### Step 1: Prepare Dataset (Choose One)

**Option A: Standard (5,000 samples)** - RECOMMENDED
```bash
python src/utils/dataset/prepare_dataset.py augmented
```

**Option B: Large Dataset (10,000 samples)**
```bash
python src/utils/dataset/prepare_dataset.py augmented --augmentations-per-image 100
```

**Option C: With Backup**
```bash
python src/utils/dataset/prepare_dataset.py augmented \
  --augmentations-per-image 100 \
  --backup-existing
```

> **Note**: The old scripts (`prepare_augmented_groundtruth.py`, `prepare_groundtruth_splits.py`, `prepare_dataset_from_parquet.py`) are now wrappers for backward compatibility. Use `prepare_dataset.py` for new workflows.

### Step 2: Verify Dataset
```bash
# Count images
echo "Training:   $(ls data/input/training/images | wc -l) images"
echo "Validation: $(ls data/input/validation/images | wc -l) images"
echo "Testing:    $(ls data/input/testing/images | wc -l) images"

# Verify labels match
echo "Training labels:   $(ls data/input/training/labels | wc -l) labels"
echo "Validation labels: $(ls data/input/validation/labels | wc -l) labels"
echo "Testing labels:    $(ls data/input/testing/labels | wc -l) labels"
```

### Step 3: Train Model
```bash
# Quick test (10 epochs)
python src/training/train.py --epochs 10 --batch 16

# Full training (100 epochs)
python src/training/train.py --epochs 100 --batch 16 --cache

# With hyperparameter optimization
python src/training/train.py --optimize --n-trials 20 --epochs 50
```

## Dataset Sizes

| Augmentations | Total Samples | Train | Val | Test |
|--------------|---------------|-------|-----|------|
| 10 | 1,000 | 700 | 150 | 150 |
| 50 | 5,000 | 3,500 | 750 | 750 |
| 100 | 10,000 | 7,000 | 1,500 | 1,500 |
| 200 | 20,000 | 14,000 | 3,000 | 3,000 |

## Key Points

✅ **Use ground-truth only** - The 100 manually annotated images have proper bounding boxes
✅ **Augmentation preserves labels** - All transformations correctly adjust bounding box coordinates
✅ **Reproducible** - Uses fixed random seed (42) for consistent splits
✅ **Git-safe** - Training data is in `.gitignore`, only scripts are committed

## Troubleshooting

**"Python not found"**
- Use your Python environment: `conda activate yolov8-env` or similar
- Or use Docker: `docker exec -it <container> python src/utils/dataset/...`

**"Labels directory not found"**
- Ensure `data/input/ground-truth/labels/` exists
- Check that .txt label files match .jpg image files

**Need more/less data**
- Adjust `--augmentations-per-image` parameter
- Start with 50, increase to 100-200 if needed

## Documentation

📚 [Full Augmentation Guide](src/utils/dataset/README_AUGMENTATION.md)
📚 [Training Guide](src/training/README.md)
📚 [Quick Reference](src/training/QUICK_REFERENCE.md)

## Unified Dataset Preparation Script

The new `prepare_dataset.py` consolidates all dataset preparation workflows:

```bash
# Simple split (no augmentation)
python src/utils/dataset/prepare_dataset.py groundtruth

# Augmented dataset (recommended)
python src/utils/dataset/prepare_dataset.py augmented --augmentations-per-image 50

# Extract from parquet files
python src/utils/dataset/prepare_dataset.py parquet --raw-dir data/raw
```

## Old Workflow (Deprecated but Still Works)

The old scripts are now wrappers for backward compatibility:
- `prepare_augmented_groundtruth.py` → wraps `prepare_dataset.py augmented`
- `prepare_groundtruth_splits.py` → wraps `prepare_dataset.py groundtruth`
- `prepare_dataset_from_parquet.py` → wraps `prepare_dataset.py parquet`

✅ **Use**: `prepare_dataset.py` with subcommands for new workflows
⚠️  **Legacy**: Old scripts still work but show compatibility notice
