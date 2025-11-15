# Models Directory

This directory contains all YOLOv8 model weights, training runs, and archived experiments for the SIADS 699 Capstone project.

## Directory Structure

```
models/
├── pretrained/          # Base model weights from Ultralytics
├── trained/             # Production-ready trained models
├── runs/                # Successful training experiments
└── archive/             # Historical/incomplete experiments
```

## Subdirectories

### `pretrained/`
Contains base YOLOv8 model weights downloaded from Ultralytics:
- **yolov8n.pt** - YOLOv8 Nano (baseline model for fine-tuning)

These are starting checkpoints used for transfer learning on our financial document dataset.

### `trained/`
Production-ready model weights used by the Streamlit application:
- **best.pt** - Symlink or copy of the current active model
- **active_run.txt** - Name of the training run currently deployed

To switch the active model:
```bash
src/utils/models/set_active_run.sh <run-name>
```

### `runs/`
Successfully completed training experiments with full metrics and artifacts:
- Each subdirectory represents one training run
- Naming convention: `finance-parser-YYYYMMDD_HHMMSS`
- Contains: weights/, results.csv, performance plots, validation images

**Current runs:**
- `finance-parser-20251112_114247/` - 170 epochs, mAP50: 0.642
- `finance-parser-20251112_143826/` - Completed run with batch samples

### `archive/`
Historical experiments and incomplete runs kept for reference:
- `finance-image-parser4/` - Failed early (stopped at epoch 4)
- `body-focus2/` - Configured but never executed

## Training Run Structure

Each training run directory contains:
```
<run-name>/
├── weights/
│   ├── best.pt          # Best model checkpoint (highest mAP)
│   └── last.pt          # Final epoch checkpoint
├── results.csv          # Per-epoch training metrics
├── args.yaml            # Hyperparameters used
├── confusion_matrix.png # Model performance visualization
├── results.png          # Training curves (loss, mAP, precision, recall)
└── val_batch*.jpg       # Validation predictions with ground truth
```

## Model Performance Comparison

### finance-parser-20251112_114247
- **Epochs Completed**: 170 / 200 (early stopping)
- **Final Metrics** (epoch 169):
  - Precision: 0.835
  - Recall: 0.490
  - mAP50: 0.642
  - mAP50-95: 0.425
- **Training Config**:
  - Batch size: 8
  - Image size: 640
  - Optimizer: SGD with momentum
  - Learning rate schedule: Default
- **Notes**: Good precision but lower recall suggests the model is conservative

### finance-parser-20251112_143826
- **Status**: Completed run
- **Notes**: Same configuration as 114247, possibly a rerun
- Contains additional intermediate batch visualizations

## Switching Models

To change the active model used by Streamlit:

```bash
# Switch to a specific run
src/utils/models/set_active_run.sh finance-parser-20251112_143826

# Restart Streamlit to load new weights
docker compose -f src/environments/docker/compose.yml restart app
```

## Model Evaluation Guidelines

When evaluating which model to deploy:

1. **Precision vs Recall Tradeoff**
   - High precision: Fewer false positives (safer for production)
   - High recall: Catches more actual sections (more complete extraction)

2. **mAP (Mean Average Precision)**
   - mAP50: Performance at 50% IoU threshold
   - mAP50-95: Average across multiple IoU thresholds (more strict)
   - Higher values = better overall detection quality

3. **Class-Specific Performance**
   - Check confusion matrices in run directories
   - Ensure all classes (header, body, footer) are well-detected

4. **Validation Visualizations**
   - Review `val_batch*.jpg` images in run directories
   - Verify predictions align with ground truth

## Baseline Model

The baseline rollback model is stored in the archive:
- Location: `models/archive/finance-image-parser4/`
- Used by: `src/utils/deployment/reset_streamlit_model.sh`

To revert to baseline:
```bash
src/utils/deployment/reset_streamlit_model.sh
```

## Best Practices

1. Always test new models locally before deploying
2. Keep `active_run.txt` updated when changing models
3. Document significant model changes in this file
4. Archive old production models before replacing
5. Monitor Streamlit logs after model updates

## Syncing Models from Great Lakes

To download a trained model from the Great Lakes HPC cluster:
```bash
src/utils/deployment/sync_yolov8_run.sh <run-name>
```

This will:
1. Rsync the run artifacts to `models/runs/<run-name>/`
2. Copy `best.pt` to `models/trained/best.pt`
3. Update `active_run.txt`
4. Rebuild and restart Streamlit containers

## Artifacts

Large runs can be compressed for easier transfer:
```bash
tar -czf <run-name>.tar.gz -C models/runs <run-name>
```

Store compressed runs in `models/artifacts/` (gitignored for size).
