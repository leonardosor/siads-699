# Final Training with Best Optuna Parameters

This guide explains how to perform final model training using the best hyperparameters found during Optuna optimization.

## Overview

After running hyperparameter optimization with `train.py --optimize`, you can use `train_final.py` to train a production-ready model with:

- **Best hyperparameters** from Optuna study
- **Extended training** (default 300 epochs vs 100 for optimization)
- **Automatic deployment** to production directory
- **Comprehensive metadata** tracking

## Quick Start

### 1. Basic Final Training

Train with best parameters from Optuna study:

```bash
python src/training/train_final.py
```

This will:
- Load best parameters from `models/experiments/active/optuna_study.db`
- Train for 300 epochs (default)
- Save results to `models/experiments/final/final-training-{timestamp}/`

### 2. Train and Deploy to Production

Automatically deploy the best model after training:

```bash
python src/training/train_final.py --deploy
```

This will:
- Train the model
- Copy `best.pt` to `models/production/best.pt`
- Update `models/production/active_run.txt`
- Create deployment history record

### 3. Custom Training Duration

Train for a specific number of epochs:

```bash
python src/training/train_final.py --epochs 500 --patience 150
```

- `--epochs 500`: Train for 500 epochs
- `--patience 150`: Early stopping after 150 epochs without improvement

## Usage Options

### Command-Line Arguments

```bash
python src/training/train_final.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--study-db PATH` | `models/experiments/active/optuna_study.db` | Path to Optuna study database |
| `--params-json PATH` | None | Load parameters from JSON file (overrides Optuna) |
| `--weights PATH` | `models/pretrained/yolov8n.pt` | Initial weights to start from |
| `--data-config PATH` | `src/training/finance-image-parser.yaml` | Dataset configuration YAML |
| `--device` | `0` | CUDA device (`0`, `1`, `cpu`, etc.) |
| `--epochs` | `300` | Number of training epochs |
| `--name` | Auto-generated | Custom run name |
| `--cache` | False | Cache images in RAM for faster training |
| `--patience` | `100` | Early stopping patience (epochs) |
| `--deploy` | False | Deploy to production after training |

### Parameter Loading Priority

The script loads parameters in this order (first found wins):

1. **Custom JSON file** (`--params-json`)
2. **Optuna study database** (`--study-db`)
3. **Default parameters** (YOLOv8 defaults)

## Advanced Usage

### 1. Use Custom Parameters from JSON

Create a JSON file with your parameters:

```json
{
  "lr0": 0.00056,
  "lrf": 0.000038,
  "momentum": 0.887,
  "weight_decay": 0.000066,
  "batch": 32,
  "optimizer": "Adam",
  "mosaic": 0.589,
  "fliplr": 0.235,
  "degrees": 3.68,
  "hsv_h": 0.0086,
  "hsv_s": 0.592,
  "hsv_v": 0.477,
  "mixup": 0.162
}
```

Train with these parameters:

```bash
python src/training/train_final.py --params-json my_params.json
```

### 2. Extract Best Parameters from Study

View and export best parameters:

```bash
# Analyze Optuna study
python src/training/analyze_study.py

# Extract to JSON (using get_best_params.py)
python src/training/get_best_params.py
```

This creates `models/experiments/active/best_params.json`.

### 3. Resume Training from Checkpoint

To continue training from a previous run:

```bash
python src/training/train_final.py \
  --weights models/experiments/final/my-previous-run/weights/last.pt \
  --epochs 200
```

### 4. Multi-GPU Training

Train on multiple GPUs:

```bash
python src/training/train_final.py --device 0,1,2,3 --batch 64
```

**Note:** Batch size may need adjustment for multiple GPUs.

### 5. Training with Image Caching

Enable image caching for faster training (requires sufficient RAM):

```bash
python src/training/train_final.py --cache --epochs 300
```

**Memory Requirements:**
- Small dataset (~100 images): ~2-4 GB RAM
- Medium dataset (~1000 images): ~20-40 GB RAM
- Large dataset (10000+ images): 200+ GB RAM

## Output Structure

After training, you'll get:

```
models/experiments/final/final-training-20251123_120000/
├── weights/
│   ├── best.pt              # Best model weights
│   └── last.pt              # Last epoch weights
├── results.csv              # Training metrics per epoch
├── results.png              # Training curves visualization
├── confusion_matrix.png     # Confusion matrix
├── training_metadata.json   # Complete training parameters
└── args.yaml               # YOLO training arguments
```

### Training Metadata File

`training_metadata.json` contains:

```json
{
  "training_info": {
    "start_time": "2025-11-23T12:00:00",
    "epochs": 300,
    "data_config": "/path/to/finance-image-parser.yaml",
    "initial_weights": "/path/to/yolov8n.pt",
    "save_directory": "/path/to/results"
  },
  "hyperparameters": {
    "lr0": 0.00056,
    "lrf": 0.000038,
    ...
  },
  "parameter_source": {
    "study_name": "yolov8_optimization",
    "trial_number": 1,
    "best_map50_95": 0.9660,
    "loaded_from": "/path/to/optuna_study.db"
  }
}
```

## Production Deployment

### Manual Deployment

After training, manually deploy the model:

```bash
# Copy best weights
cp models/experiments/final/final-training-*/weights/best.pt \
   models/production/best.pt

# Update active run marker
echo "final-training-20251123_120000" > models/production/active_run.txt
```

### Automatic Deployment

Use the `--deploy` flag:

```bash
python src/training/train_final.py --deploy
```

This creates:
- `models/production/best.pt` - Production model weights
- `models/production/active_run.txt` - Active run identifier
- `models/production/training_metadata.json` - Training parameters
- `models/production/deployment_history.json` - Deployment records

### Deployment History

`deployment_history.json` tracks all deployments:

```json
[
  {
    "deployed_at": "2025-11-23T12:30:00",
    "run_name": "final-training-20251123_120000",
    "source_directory": "/path/to/results",
    "weights_path": "/path/to/production/best.pt"
  }
]
```

## Best Practices

### 1. Training Duration

| Scenario | Recommended Epochs | Patience |
|----------|-------------------|----------|
| Quick validation | 100 | 30 |
| Standard training | 300 | 100 |
| Maximum performance | 500-1000 | 150-200 |
| Fine-tuning | 50-100 | 20-30 |

### 2. Monitoring Training

Watch training progress in real-time:

```bash
# In another terminal
tail -f models/experiments/final/final-training-*/results.csv
```

Or use TensorBoard (if configured):

```bash
tensorboard --logdir models/experiments/final/
```

### 3. Model Validation

After training, validate the model:

```bash
# Run validation metrics
python notebooks/validate_ci.py

# Test on sample images
python src/training/debug_predictions.py \
  --model models/production/best.pt \
  --images data/input/ground-truth/
```

### 4. Parameter Tuning Tips

If the model from Optuna isn't performing well:

1. **Check trial results:** `python src/training/analyze_study.py`
2. **Try top-3 parameter sets:** Export and test manually
3. **Run more Optuna trials:** Increase `--n-trials` in `train.py`
4. **Adjust epoch count:** Some models need more time to converge

## Troubleshooting

### "No studies found in database"

The Optuna study database doesn't exist. Run optimization first:

```bash
python src/training/train.py --optimize --n-trials 20
```

### CUDA Out of Memory

Reduce batch size or enable gradient accumulation:

```bash
# Reduce batch size in best_params.json
{
  "batch": 8,  # Reduced from 32
  ...
}

python src/training/train_final.py --params-json best_params.json
```

### Training Plateaus Early

Increase patience or adjust learning rate schedule:

```bash
python src/training/train_final.py --patience 150 --epochs 500
```

### Best Model Not Improving

Try:
1. Longer warmup: Modify `warmup_epochs` in code
2. Different optimizer: Edit parameters JSON
3. Learning rate adjustment: Test lower/higher `lr0`

## Integration with Pipeline

### Complete Training Workflow

```bash
# 1. Prepare data
python scripts/prepare_data.py

# 2. Run hyperparameter optimization
python src/training/train.py --optimize --n-trials 30 --epochs 100

# 3. Analyze results
python src/training/analyze_study.py

# 4. Final training with best parameters
python src/training/train_final.py --epochs 300 --deploy

# 5. Validate model
python notebooks/validate_ci.py

# 6. Test predictions
python src/training/debug_predictions.py \
  --model models/production/best.pt \
  --images data/input/validation/
```

## Performance Benchmarks

Typical training times (NVIDIA A100):

| Configuration | Time per Epoch | Total Time (300 epochs) |
|---------------|----------------|-------------------------|
| Batch=8, no cache | 45s | ~3.75 hours |
| Batch=16, no cache | 30s | ~2.5 hours |
| Batch=32, cached | 15s | ~1.25 hours |

**Note:** Times vary based on:
- GPU model
- Dataset size
- Image resolution
- Augmentation intensity

## Further Reading

- [Optuna Optimization Guide](../src/training/TRAINING_IMPROVEMENTS.md)
- [Model Validation](../notebooks/validate_ci.py)
- [Dataset Variability Analysis](../notebooks/calculate_dataset_variability.py)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

## Support

For issues or questions:
1. Check existing documentation in `docs/`
2. Review Optuna results: `src/training/analyze_study.py`
3. Validate dataset: `notebooks/validate_ci.py`
4. Check training logs in `models/experiments/final/*/`
