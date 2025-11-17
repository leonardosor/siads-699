# Optuna Hyperparameter Optimization Results

## Summary

Out of 12 trials, **11 completed successfully** and 1 failed due to CUDA errors. The best performing trial achieved a **mAP50-95 of 0.9068** (Trial 6).

## Best Parameters (Trial 6)

```json
{
  "batch": 16,
  "degrees": 0.01336678075573472,
  "fliplr": 0.25223348475253676,
  "hsv_h": 0.025166804231601173,
  "hsv_s": 0.670122948392864,
  "hsv_v": 0.10974989167768132,
  "lr0": 0.0028875899337697402,
  "lrf": 1.4026707974095094e-05,
  "mixup": 0.16460308761846165,
  "momentum": 0.879355229753037,
  "mosaic": 0.8312654351989561,
  "optimizer": "Adam",
  "weight_decay": 2.7832175103615595e-05
}
```

## Top 5 Trials by Performance

1. **Trial 6**: mAP50-95 = 0.9068 (Best)
2. **Trial 10**: mAP50-95 = 0.8954
3. **Trial 0**: mAP50-95 = 0.7511
4. **Trial 1**: mAP50-95 = 0.7503
5. **Trial 9**: mAP50-95 = 0.7376

## Failed Trials

- **Trial 11**: Failed with CUDA error (`cudaErrorUnknown`)

## Fixes Applied to train.py

### 1. GPU Memory Management

Added `clear_gpu_memory()` function that:
- Clears CUDA cache with `torch.cuda.empty_cache()`
- Synchronizes CUDA operations with `torch.cuda.synchronize()`
- Runs garbage collection with `gc.collect()`

This function is now called:
- Before each training trial starts
- After each trial completes (successful or failed)
- After cleaning up model and results objects

### 2. Error Handling in Objective Function

The objective function now includes comprehensive error handling:

```python
try:
    # Clear GPU memory before starting
    clear_gpu_memory()

    # Train model
    # ...

    # Clear GPU memory after training
    del model
    del results
    clear_gpu_memory()

except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    # Handle CUDA errors gracefully
    # Store error info in trial user attributes
    # Clear GPU memory
    # Re-raise to mark trial as failed
    raise
```

### 3. Optuna Study Configuration

Updated `study.optimize()` call with:
- `catch=(Exception,)`: Catches all exceptions and continues with next trial instead of crashing
- `gc_after_trial=True`: Runs garbage collection after each trial

### 4. Improved Result Reporting

Added checks for completed vs. failed trials:
- Displays count of completed and failed trials
- Warns if all trials failed
- Provides troubleshooting suggestions

## How to Use the Best Parameters

### Option 1: Run the analysis script

```bash
python src/training/get_best_params.py
```

This will display the best parameters and save them to `best_params.json`.

### Option 2: Continue Optuna optimization

If you want to run more trials with the fixes applied:

```bash
python src/training/train.py --optimize --n-trials 10 --epochs 100
```

The study will load from the existing database and continue optimization.

### Option 3: Train with best parameters directly

Use the parameters from Trial 6 for final training:

```bash
python src/training/train.py \
  --epochs 300 \
  --batch 16 \
  --lr0 0.002888 \
  --lrf 0.000014 \
  --momentum 0.879 \
  --weight_decay 0.000028 \
  --optimizer Adam \
  --mosaic 0.831 \
  --fliplr 0.252 \
  --degrees 0.013 \
  --hsv_h 0.025 \
  --hsv_s 0.670 \
  --hsv_v 0.110 \
  --mixup 0.165
```

## Common CUDA Error Solutions

If you continue to experience CUDA errors:

1. **Reduce batch size**: Try `--batch 8` instead of 16 or 32
2. **Set environment variable**: `export CUDA_LAUNCH_BLOCKING=1` for better error messages
3. **Restart GPU**: Reset CUDA context or restart Docker container
4. **Update GPU drivers**: Ensure you have the latest NVIDIA drivers
5. **Check GPU memory**: Use `nvidia-smi` to monitor GPU memory usage

## Files Generated

- `models/experiments/active/optuna_study.db`: SQLite database with all trial data
- `models/experiments/active/best_params.json`: Best hyperparameters from completed trials
- `src/training/get_best_params.py`: Script to analyze study results
- `src/training/analyze_study.py`: Alternative analysis script (requires optuna and pandas)

## Next Steps

1. Train final model with best parameters for 300 epochs
2. Evaluate on test set
3. Compare with baseline model
4. Deploy to production if performance is satisfactory
