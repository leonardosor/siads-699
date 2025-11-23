# Final Training Guide

This guide explains how to perform the final training step using the best hyperparameters found by Optuna.

## Quick Start

### 1. View Best Parameters

First, check what parameters Optuna found:

```bash
python src/training/show_best_params.py
```

This will display:
- The best trial number and mAP50-95 score
- All optimized hyperparameters
- A suggested training command

### 2. Run Final Training

Train with the best parameters:

```bash
python src/training/train_final.py --epochs 300 --deploy
```

This will:
- Load best parameters from `models/experiments/active/optuna_study.db`
- Train for 300 epochs (or until early stopping)
- Save results to `models/experiments/final/`
- Deploy the best model to `models/production/`

## Command Line Options

### Basic Options

```bash
python src/training/train_final.py \
  --epochs 300 \              # Number of training epochs (default: 300)
  --patience 100 \            # Early stopping patience (default: 100)
  --device 0 \                # GPU device ('0', '1', or 'cpu')
  --cache \                   # Cache images in RAM for faster training
  --deploy                    # Deploy to production after training
```

### Advanced Options

```bash
python src/training/train_final.py \
  --study-db path/to/optuna_study.db \     # Custom Optuna database
  --params-json path/to/params.json \      # Override with JSON params
  --weights models/pretrained/yolov8s.pt \ # Start from different weights
  --name my-final-run \                    # Custom run name
  --data-config path/to/data.yaml          # Custom dataset config
```

## Parameter Sources (Priority Order)

The script loads parameters in this order:

1. **JSON file** (if `--params-json` specified)
2. **Optuna database** (default: `models/experiments/active/optuna_study.db`)
3. **Default values** (if no study found)

### Using Custom Parameters

Export Optuna parameters to JSON:

```bash
python src/training/show_best_params.py --save-json best_params.json
```

Edit `best_params.json` if needed, then train with it:

```bash
python src/training/train_final.py --params-json best_params.json --epochs 300 --deploy
```

## Understanding the Output

After training completes, you'll find:

```
models/experiments/final/<run-name>/
├── weights/
│   ├── best.pt              # Best model weights (highest mAP50-95)
│   └── last.pt              # Last epoch weights
├── results.png              # Training curves
├── confusion_matrix.png     # Confusion matrix
├── training_metadata.json   # Complete training configuration
└── args.yaml                # YOLO training arguments
```

If `--deploy` was used:

```
models/production/
├── best.pt                  # Deployed production model
├── training_metadata.json   # Training info
├── active_run.txt           # Current run name
└── deployment_history.json  # Deployment log
```

## Recommended Training Configurations

### Quick Test (Verify Parameters)
```bash
python src/training/train_final.py --epochs 50 --patience 20
```

### Standard Training
```bash
python src/training/train_final.py --epochs 300 --patience 100 --cache --deploy
```

### Extended Training (Best Results)
```bash
python src/training/train_final.py --epochs 500 --patience 150 --cache --deploy
```

### GPU Memory Issues
```bash
# Use smaller model or reduce batch size in Optuna params
python src/training/train_final.py --epochs 300 --device 0
```

## Monitoring Training

### During Training

Watch the terminal output for:
- Epoch progress
- Loss values (box_loss, cls_loss, dfl_loss)
- Metrics (precision, recall, mAP50, mAP50-95)
- Early stopping countdown

### After Training

1. Check training curves:
   ```bash
   open models/experiments/final/<run-name>/results.png
   ```

2. View final metrics:
   ```bash
   cat models/experiments/final/<run-name>/training_metadata.json
   ```

3. Compare with Optuna trials:
   ```bash
   python src/training/show_best_params.py
   ```

## Deployment

### Automatic Deployment

Use `--deploy` flag to automatically deploy after training:

```bash
python src/training/train_final.py --epochs 300 --deploy
```

### Manual Deployment

Copy weights manually:

```bash
cp models/experiments/final/<run-name>/weights/best.pt models/production/best.pt
```

### Check Active Production Model

```bash
cat models/production/active_run.txt
cat models/production/deployment_history.json
```

## Troubleshooting

### "No studies found in database"

**Problem:** Optuna study database doesn't exist or is empty.

**Solution:**
1. Check if you've run Optuna tuning:
   ```bash
   ls -lh models/experiments/active/optuna_study.db
   ```
2. If missing, run hyperparameter tuning first:
   ```bash
   python src/training/optuna_tuning.py --trials 50
   ```

### "CUDA out of memory"

**Problem:** GPU doesn't have enough memory.

**Solutions:**
1. Reduce batch size (edit parameters or JSON)
2. Use CPU training: `--device cpu`
3. Use smaller model: `--weights models/pretrained/yolov8n.pt`
4. Don't use cache: remove `--cache` flag

### Early Stopping Too Soon

**Problem:** Training stops before convergence.

**Solution:** Increase patience:
```bash
python src/training/train_final.py --epochs 500 --patience 200
```

### Poor Performance

**Problem:** Final model performs worse than expected.

**Solutions:**
1. Verify parameters: `python src/training/show_best_params.py`
2. Check data quality in `data/input/`
3. Run more Optuna trials before final training
4. Try longer training: `--epochs 500`

## Best Practices

1. **Always check best parameters first:**
   ```bash
   python src/training/show_best_params.py
   ```

2. **Use cache for faster training:**
   ```bash
   # Only if you have enough RAM
   python src/training/train_final.py --cache --epochs 300
   ```

3. **Monitor training actively:**
   - Watch for overfitting (validation loss increasing)
   - Check if early stopping triggers appropriately
   - Verify metrics improve over time

4. **Deploy only after validation:**
   ```bash
   # Train without deploy first
   python src/training/train_final.py --epochs 300
   
   # Validate results, then deploy manually
   cp models/experiments/final/<run>/weights/best.pt models/production/best.pt
   ```

5. **Keep deployment history:**
   - `deployment_history.json` tracks all deployments
   - Useful for A/B testing or rollbacks

## Example Workflow

Complete workflow from Optuna to production:

```bash
# 1. Run hyperparameter optimization (if not done)
python src/training/optuna_tuning.py --trials 50

# 2. View best parameters
python src/training/show_best_params.py

# 3. (Optional) Export and customize parameters
python src/training/show_best_params.py --save-json custom_params.json
# Edit custom_params.json if needed

# 4. Final training
python src/training/train_final.py --epochs 300 --cache --patience 100

# 5. Validate results
python notebooks/validate_ci.py

# 6. Deploy to production
python src/training/train_final.py --deploy
# OR manually copy: cp models/experiments/final/<run>/weights/best.pt models/production/best.pt
```

## Additional Resources

- **Optuna Study Dashboard:** Use `optuna-dashboard` to visualize trials
- **TensorBoard:** YOLO logs to TensorBoard automatically
- **Validation Notebook:** `notebooks/validate_ci.py` for confidence intervals

## Questions?

- Check Optuna study: `models/experiments/active/optuna_study.db`
- Review training logs: `models/experiments/final/<run-name>/`
- Validate dataset: `python notebooks/validate_ci.py`

## Batch / Slurm Execution

To launch multiple final-training runs on the Great Lakes cluster (varying seeds or epochs) use the batch script:

```bash
sbatch src/training/batch_final_training.sh
```

Environment overrides (export before `sbatch`):

```bash
export SEEDS="42 1337 2025"          # Seeds to iterate
export EPOCHS_LIST="300 400"         # Epoch counts per seed
export EXTRA_FLAGS="--cache"         # Additional flags to pass through
export DEPLOY_MODE=best               # one of: first|best|all|none
export DEVICE=0                       # GPU device spec
```

Optional: Pre-export parameters once (avoids repeated DB reads):

```bash
python src/training/show_best_params.py --save-json models/experiments/active/best_params.json
export PARAMS_JSON=models/experiments/active/best_params.json
sbatch src/training/batch_final_training.sh
```

Deployment modes:

- `best`: deploy run with highest `best_map50_95` (default)
- `first`: deploy only the first run in the sequence
- `all`: deploy every run (overwrites `production/best.pt` each time)
- `none`: skip deployment entirely

Summary report is written to `models/experiments/final/batch_summary_<timestamp>.txt` listing each run and its mAP score.

Note: The batch script assumes the Optuna study DB at `models/experiments/active/optuna_study.db`. If missing, runs will fall back to default parameters.

