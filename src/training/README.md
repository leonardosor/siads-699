# Model Training

This directory contains scripts and configurations for training YOLOv8 models on financial document datasets.

## Files

- **train.py** - Production training script with full CLI argument support
- **batch_job.sh** - SLURM batch job for Great Lakes HPC cluster
- **finance-image-parser.yaml** - Dataset configuration for YOLOv8

## Quick Start

### Local Training (CPU/GPU)

```bash
# Basic training
python src/training/train.py \
  --weights models/pretrained/yolov8n.pt \
  --data-config src/training/finance-image-parser.yaml \
  --epochs 50 \
  --batch 8 \
  --imgsz 640 \
  --project models/runs \
  --name my-experiment

# With GPU
python src/training/train.py \
  --weights models/pretrained/yolov8n.pt \
  --data-config src/training/finance-image-parser.yaml \
  --epochs 150 \
  --batch 16 \
  --imgsz 1024 \
  --device 0 \
  --project models/runs \
  --name finance-high-res
```

### Great Lakes HPC Training

```bash
# Default configuration (150 epochs, batch=4, imgsz=1024)
sbatch src/training/batch_job.sh

# Custom configuration
EPOCHS=200 BATCH=8 IMGSZ=640 sbatch src/training/batch_job.sh

# With custom run name
RUN_NAME=my-custom-run sbatch src/training/batch_job.sh
```

## Training Script Arguments

### Required Arguments
- `--weights` - Path to pretrained weights (e.g., models/pretrained/yolov8n.pt)
- `--data-config` - Path to dataset YAML (e.g., src/training/finance-image-parser.yaml)

### Common Arguments
- `--epochs` - Number of training epochs (default: 100)
- `--batch` - Batch size (default: 16)
- `--imgsz` - Input image size (default: 640)
- `--device` - Device to use (cpu, 0, 0,1,2,3) (default: 0 if GPU available)
- `--project` - Output directory for runs (default: models/runs)
- `--name` - Name of this training run (default: auto-generated)

### Advanced Arguments
- `--lr0` - Initial learning rate (default: 0.01)
- `--lrf` - Final learning rate factor (default: 0.01)
- `--momentum` - SGD momentum (default: 0.937)
- `--weight-decay` - Optimizer weight decay (default: 0.0005)
- `--patience` - Early stopping patience in epochs (default: 50)
- `--workers` - DataLoader workers (default: 8)
- `--cos-lr` - Use cosine learning rate schedule
- `--resume` - Resume from last checkpoint
- `--clean-broken` - Check and remove corrupt images before training

## Dataset Configuration

The `finance-image-parser.yaml` file defines the dataset structure:

```yaml
path: ../data/input/training  # Dataset root
train: images                  # Training images subdirectory
val: ../validation/images      # Validation images
test: ../testing/images        # Test images

names:
  0: header
  1: body
  2: footer
```

Dataset should be organized as:
```
data/input/
├── training/
│   ├── images/     # .jpg/.png files
│   └── labels/     # .txt files with same names as images
├── validation/
│   ├── images/
│   └── labels/
└── testing/
    ├── images/
    └── labels/
```

## SLURM Batch Job (Great Lakes)

### Environment Variables

Configure the batch job using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | `/home/$USER/siads-699` | Repository root on Great Lakes |
| `RUN_NAME` | `finance-parser-YYYYMMDD_HHMMSS` | Training run name |
| `EPOCHS` | `150` | Number of epochs |
| `BATCH` | `4` | Batch size |
| `IMGSZ` | `1024` | Image size |
| `PATIENCE` | `60` | Early stopping patience |
| `HYPERPARAMS` | `--mosaic 0 --cache` | Additional hyperparameters |

### Example Configurations

**Quick test run:**
```bash
EPOCHS=10 BATCH=8 IMGSZ=640 RUN_NAME=quick-test sbatch src/training/batch_job.sh
```

**High resolution training:**
```bash
EPOCHS=200 BATCH=4 IMGSZ=1024 PATIENCE=80 sbatch src/training/batch_job.sh
```

**Resume previous run:**
```bash
HYPERPARAMS="--resume" RUN_NAME=my-previous-run sbatch src/training/batch_job.sh
```

### Monitoring Training

```bash
# View latest log
src/utils/monitoring/tail_latest_training_log.sh

# Follow log in real-time
src/utils/monitoring/tail_latest_training_log.sh 100

# Check job status
squeue -u $USER

# View full log
cat ~/team12_yolov8_main-<job-id>.log
```

### Downloading Results

After training completes:

```bash
# Download and deploy in one step
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826

# Download only (no deployment)
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826 --no-restart

# Download without updating active model
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826 --no-best
```

## Training Output

Each training run creates a directory: `models/runs/<run-name>/`

Contents:
```
<run-name>/
├── weights/
│   ├── best.pt              # Best checkpoint (highest mAP)
│   └── last.pt              # Last epoch checkpoint
├── results.csv              # Per-epoch metrics
├── args.yaml                # Training configuration
├── confusion_matrix.png     # Classification confusion matrix
├── confusion_matrix_normalized.png
├── results.png              # Training curves (loss, mAP, precision, recall)
├── BoxF1_curve.png          # F1 score vs confidence
├── BoxP_curve.png           # Precision vs confidence
├── BoxR_curve.png           # Recall vs confidence
├── BoxPR_curve.png          # Precision-Recall curve
├── labels.jpg               # Label distribution visualization
├── train_batch*.jpg         # Sample training batches
└── val_batch*.jpg           # Validation predictions
```

## Interpreting Results

### Key Metrics (from results.csv)

- **train/box_loss** - Bounding box localization loss (lower is better)
- **train/cls_loss** - Classification loss (lower is better)
- **metrics/precision** - True Positives / (TP + False Positives)
- **metrics/recall** - True Positives / (TP + False Negatives)
- **metrics/mAP50** - Mean Average Precision at IoU=0.5
- **metrics/mAP50-95** - Mean Average Precision averaged over IoU 0.5-0.95

### What to Look For

**Good Training:**
- Smooth decrease in training losses
- Validation mAP increases over time
- Low gap between train and validation metrics
- Precision and recall both high (>0.7)

**Overfitting Signs:**
- Training loss decreases but validation loss increases
- Large gap between train/validation metrics
- Validation mAP plateaus while training continues

**Underfitting Signs:**
- Both training and validation losses remain high
- Low precision and recall
- Erratic training curves

## Hyperparameter Tuning

### Image Size
- **640**: Fast training, good for simple layouts
- **1024**: Better for small text/details, slower training
- **1280+**: Highest quality, requires powerful GPU

### Batch Size
- **Larger (16+)**: More stable gradients, faster training
- **Smaller (4-8)**: Required for large image sizes or limited GPU memory

### Learning Rate
- **Default (0.01)**: Good starting point
- **Lower (0.001)**: If training is unstable
- **Higher (0.05)**: If convergence is too slow

### Data Augmentation
- **--mosaic 1.0**: Enabled (default) - good for varied layouts
- **--mosaic 0.0**: Disabled - use if training images are very uniform
- **--cache**: Cache images in memory (faster but uses more RAM)

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch` size
- Reduce `--imgsz`
- Reduce `--workers`

### Training Too Slow
- Increase `--batch` if GPU memory allows
- Use `--cache` to cache images in memory
- Reduce `--workers` if CPU is bottleneck

### Poor Convergence
- Increase `--patience` for more training time
- Try `--cos-lr` for cosine learning rate schedule
- Adjust `--lr0` (try 0.001 or 0.05)

### Corrupt Images Error
- Use `--clean-broken` flag to auto-remove corrupt images
- Or manually check with: `src/utils/dataset/count_yolo_labels.py`

## Best Practices

1. **Start small** - Test with --epochs 10 before full training
2. **Monitor early** - Check first few epochs for issues
3. **Use early stopping** - Set reasonable --patience to avoid wasted compute
4. **Save runs** - Don't delete failed runs, archive them for comparison
5. **Document changes** - Note hyperparameter changes in run names
6. **Version control** - Commit dataset changes before training

## Advanced: Multi-GPU Training

```bash
# Use GPUs 0 and 1
python src/training/train.py \
  --device 0,1 \
  --batch 32 \
  --workers 16 \
  ...
```

Note: Multi-GPU not supported in current Great Lakes configuration (requests only 1 GPU).
