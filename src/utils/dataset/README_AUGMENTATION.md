# Ground-Truth Dataset Augmentation Guide

This guide explains how to prepare your training dataset using ONLY the 100 ground-truth images with proper bounding box annotations.

## Overview

The current training dataset contains ~16,000 images extracted from parquet files with generic full-image bounding boxes. This workflow replaces that dataset with augmented versions of your 100 high-quality ground-truth annotated images.

## Quick Start

### 1. Prepare Augmented Dataset

Run the augmentation script to generate enhanced versions of your ground-truth images:

```bash
# Basic usage (50 augmentations per image, ~5,000 total samples)
python src/utils/dataset/prepare_augmented_groundtruth.py

# More augmentations for larger dataset (100 augmentations = ~10,000 samples)
python src/utils/dataset/prepare_augmented_groundtruth.py --augmentations-per-image 100

# Include original images + augmentations
python src/utils/dataset/prepare_augmented_groundtruth.py --keep-originals

# Backup existing data before replacing
python src/utils/dataset/prepare_augmented_groundtruth.py --backup-existing
```

### 2. Verify the Dataset

Check that the augmented dataset was created correctly:

```bash
# Count images in each split
ls data/input/training/images | wc -l
ls data/input/validation/images | wc -l
ls data/input/testing/images | wc -l

# Verify labels exist
ls data/input/training/labels | wc -l
```

### 3. Start Training

Train your model using the augmented ground-truth dataset:

```bash
# Standard training
python src/training/train.py --epochs 100 --batch 16 --cache

# With hyperparameter optimization
python src/training/train.py --optimize --n-trials 20 --epochs 50
```

## Script Options

### `prepare_augmented_groundtruth.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--ground-truth-dir` | `data/input/ground-truth` | Directory with ground-truth images and labels |
| `--output-dir` | `data/input` | Output directory (creates train/val/test subdirs) |
| `--augmentations-per-image` | `50` | Number of augmented versions per image |
| `--train-ratio` | `0.7` | Training set ratio |
| `--val-ratio` | `0.15` | Validation set ratio |
| `--test-ratio` | `0.15` | Test set ratio |
| `--keep-originals` | False | Include original images in addition to augmentations |
| `--backup-existing` | False | Backup existing training data before replacing |
| `--seed` | `42` | Random seed for reproducibility |

## Augmentation Types

The script applies the following augmentations while preserving bounding box coordinates:

### Geometric Transformations
- **Horizontal flip**: Mirror image horizontally (adjusts x-coordinates)

### Color/Brightness Adjustments
- **Brightness up/down**: Adjust image brightness (±30)
- **Contrast up/down**: Adjust image contrast (0.7x - 1.3x)
- **HSV shift**: Modify hue, saturation, and value
- **Gaussian noise**: Add random noise for robustness

### Combined Augmentations
- Multiple transformations applied together for variety

## Dataset Size Examples

| Augmentations per Image | Ground-Truth Images | Total Samples | Train (70%) | Val (15%) | Test (15%) |
|-------------------------|---------------------|---------------|-------------|-----------|------------|
| 10 | 100 | 1,000 | 700 | 150 | 150 |
| 50 | 100 | 5,000 | 3,500 | 750 | 750 |
| 100 | 100 | 10,000 | 7,000 | 1,500 | 1,500 |
| 200 | 100 | 20,000 | 14,000 | 3,000 | 3,000 |

*Add `--keep-originals` to include the original 100 images in addition to augmentations*

## Why Use Ground-Truth Only?

### Before (Parquet Data)
- ❌ ~16,000 images with generic full-image bounding boxes
- ❌ Document-level labels (not region detection)
- ❌ No actual header/body/footer annotations
- ❌ Lower quality training signal

### After (Augmented Ground-Truth)
- ✅ High-quality bounding box annotations
- ✅ Proper header/body/footer region labels
- ✅ Augmented for dataset size and robustness
- ✅ Better training signal from fewer, better-labeled samples

## Workflow Integration

### Local Training
```bash
# 1. Prepare augmented dataset
python src/utils/dataset/prepare_augmented_groundtruth.py --augmentations-per-image 100

# 2. Train locally
python src/training/train.py --epochs 100 --batch 16 --cache
```

### Great Lakes (HPC) Training
```bash
# 1. Prepare dataset locally (or on Great Lakes)
python src/utils/dataset/prepare_augmented_groundtruth.py --augmentations-per-image 100

# 2. Submit batch job
sbatch src/training/batch_job.sh

# 3. Sync results back
bash src/utils/deployment/sync_yolov8_run.sh finance-parser-20251115_120000
```

## Troubleshooting

### Issue: "Labels directory not found"
**Solution**: Ensure `data/input/ground-truth/labels/` exists with .txt label files

### Issue: "No ground-truth image-label pairs found"
**Solution**: Check that each .jpg image has a corresponding .txt label file

### Issue: Training dataset too small
**Solution**: Increase `--augmentations-per-image` (try 100-200 for larger datasets)

### Issue: Training dataset too large / out of memory
**Solution**: Decrease `--augmentations-per-image` or use smaller batch size in training

## Backup and Recovery

### Backup Current Dataset
```bash
# Manual backup
cp -r data/input/training data/input/training_backup
cp -r data/input/validation data/input/validation_backup
cp -r data/input/testing data/input/testing_backup

# Or use the --backup-existing flag
python src/utils/dataset/prepare_augmented_groundtruth.py --backup-existing
```

### Restore from Backup
```bash
# Restore manually backed up data
rm -rf data/input/training data/input/validation data/input/testing
mv data/input/training_backup data/input/training
mv data/input/validation_backup data/input/validation
mv data/input/testing_backup data/input/testing

# Or restore from script-created backup
rm -rf data/input/training data/input/validation data/input/testing
mv data/input/training_backup_42 data/input/training
mv data/input/validation_backup_42 data/input/validation
mv data/input/testing_backup_42 data/input/testing
```

## Next Steps

After preparing your augmented dataset:

1. **Verify data quality**: Visually inspect some augmented images and labels
2. **Run training**: Start with a shorter training run to validate the dataset
3. **Monitor metrics**: Watch for improvements in mAP50 and mAP50-95
4. **Iterate**: Adjust augmentation parameters based on results

## Related Scripts

- [prepare_groundtruth_splits.py](prepare_groundtruth_splits.py) - Split ground-truth without augmentation
- [augment_dataset.py](augment_dataset.py) - General-purpose augmentation tool
- [prepare_dataset_from_parquet.py](prepare_dataset_from_parquet.py) - Extract from parquet files (old workflow)
