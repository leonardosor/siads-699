# Windows Training Troubleshooting Guide

## Common Issues and Solutions

### 1. PyTorch CUDA DLL Loading Error

**Error Message:**
```
OSError: [WinError 1455] The paging file is too small for this operation to complete.
Error loading "C:\Users\...\torch\lib\cufft64_11.dll" or one of its dependencies.
```

**Root Cause:**
Windows virtual memory (page file) is insufficient to load PyTorch's CUDA libraries into RAM.

**Solutions (in order of preference):**

#### Option A: Increase Windows Page File (Recommended)

1. **Open Virtual Memory Settings:**
   - Press `Win + R`, type `SystemPropertiesAdvanced`, press Enter
   - OR: Right-click "This PC" → Properties → Advanced system settings
   - Click "Settings" under Performance
   - Go to "Advanced" tab → Click "Change" under Virtual Memory

2. **Configure Page File:**
   - Uncheck "Automatically manage paging file size for all drives"
   - Select your system drive (usually C:)
   - Choose "Custom size"
   - Set **Initial size: 8192 MB** (8 GB)
   - Set **Maximum size: 16384 MB** (16 GB)
   - Click "Set" then "OK"

3. **Restart your computer** for changes to take effect

4. **After restart, test GPU training:**
   ```pwsh
   python src/training/train_final.py --epochs 10 --device 0 --name test-gpu
   ```

#### Option B: Use CPU Training (Temporary Workaround)

If you can't restart immediately or need to train right away:

```pwsh
# Basic CPU training
python src/training/train_final.py --epochs 50 --device cpu --name final-cpu-run

# CPU training with caching (faster)
python src/training/train_final.py --epochs 50 --device cpu --name final-cpu-run --cache
```

**Note:** CPU training is 10-50x slower than GPU but works around the DLL loading issue.

#### Option C: Free Up System Resources

Before starting training:

1. **Close unnecessary applications:**
   - Web browsers (Chrome/Edge consume lots of RAM)
   - IDEs (VSCode, PyCharm)
   - Other memory-intensive programs

2. **Check available RAM:**
   ```pwsh
   Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory, TotalVisibleMemorySize
   ```

3. **Restart Python environment:**
   ```pwsh
   # Deactivate and reactivate conda environment
   conda deactivate
   conda activate capstone
   ```

4. **Try GPU training again:**
   ```pwsh
   python src/training/train_final.py --epochs 10 --device 0 --name production-run
   ```

---

### 2. Dataset Path Not Found

**Error Message:**
```
RuntimeError: Dataset 'finance-image-parser.yaml' error
images not found, missing path 'D:\workspace\data\input\...'
```

**Solution:**

The dataset YAML must use absolute paths on Windows. Check `src/training/finance-image-parser.yaml`:

```yaml
# CORRECT (absolute path):
path: D:/docs/MADS/699/data/input
train: training/images
val: validation/images
test: testing/images

# INCORRECT (relative paths don't resolve correctly):
path: ../data/input
path: /workspace/data/input
```

**Verify dataset structure:**
```pwsh
Get-ChildItem -Path .\data\input\training\images | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem -Path .\data\input\validation\images | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem -Path .\data\input\testing\images | Measure-Object | Select-Object -ExpandProperty Count
```

---

### 3. Out of Memory (OOM) During Training

**Error Message:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions:**

#### Reduce Batch Size
```pwsh
# Default batch from Optuna: 32
# Try smaller batch:
python src/training/train_final.py --epochs 100 --device 0 --params-json custom_params.json
```

Create `custom_params.json`:
```json
{
  "batch": 16,
  "lr0": 0.001407,
  "lrf": 0.000022,
  "momentum": 0.863025,
  "weight_decay": 0.000094,
  "optimizer": "AdamW",
  "mosaic": 0.770396,
  "fliplr": 0.182890,
  "degrees": 0.052213,
  "hsv_h": 0.025551,
  "hsv_s": 0.730214,
  "hsv_v": 0.139144,
  "mixup": 0.124436
}
```

#### Clear GPU Cache Before Training
```pwsh
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
python src/training/train_final.py --epochs 100 --device 0
```

#### Monitor GPU Usage
```pwsh
# In separate terminal, monitor GPU:
nvidia-smi -l 1
```

---

### 4. Precision/Recall Showing as 0.0 in Validation Report

**Issue:**
Running `python notebooks/validate_ci.py` shows precision and recall as 0.0 despite high mAP50.

**Root Cause:**
The Optuna study database doesn't store precision/recall as user attributes; only the objective value (mAP50-95) is saved.

**Solution:**

The validation script now scans per-trial `results.csv` files to backfill these metrics. To verify:

```pwsh
# Check if trial_26 (best trial) has metrics:
python -c "import pandas as pd; df=pd.read_csv('models/experiments/active/trial_26/results.csv'); print(df.columns.tolist()); print(df[['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']].max())"
```

If you still see zeros, reorder the metric retrieval by processing epoch-level results before the Optuna DB (see code comment in `notebooks/validate_ci.py` line ~198).

**Alternative:** Run a fresh final training and the metrics will populate from the new run's `results.csv`:

```pwsh
python src/training/train_final.py --epochs 100 --device 0 --name final-production --deploy
python notebooks/validate_ci.py
```

---

### 5. Training Interrupted or Stuck

**Issue:**
Training starts but hangs at "0%|          | 0/70 [00:00<?, ?it/s]" or exits with code 1.

**Common Causes:**

1. **Manual Ctrl+C interruption** - Expected behavior
2. **System resources exhausted** - See Option A (increase page file)
3. **Batch size too large** - See "Out of Memory" section

**Recovery:**

```pwsh
# Resume from last checkpoint (if training was partially complete):
python src/training/train_final.py --epochs 100 --device 0 --name production-run

# Start fresh with smaller batch:
python src/training/train_final.py --epochs 100 --device 0 --params-json custom_params.json
```

---

## Quick Reference Commands

### Training Commands

```pwsh
# Full production training with best Optuna params (GPU):
python src/training/train_final.py --epochs 300 --device 0 --name final-production --deploy --patience 100

# Quick test run (1 epoch, CPU):
python src/training/train_final.py --epochs 1 --device cpu --name quick-test

# Training with custom parameters:
python src/training/train_final.py --epochs 100 --device 0 --params-json custom_params.json --name custom-run

# Training with image caching (faster but uses more RAM):
python src/training/train_final.py --epochs 100 --device 0 --cache --name cached-run
```

### Validation Commands

```pwsh
# Run statistical validation:
python notebooks/validate_ci.py

# Calculate dataset variability (Omega):
python notebooks/calculate_dataset_variability.py

# Check dataset composition:
python -c "from pathlib import Path; print('Training:', len(list(Path('data/input/training/images').glob('*.jpg')))); print('Validation:', len(list(Path('data/input/validation/images').glob('*.jpg')))); print('Testing:', len(list(Path('data/input/testing/images').glob('*.jpg'))))"
```

### System Diagnostics

```pwsh
# Check Python environment:
python --version
conda list | Select-String "torch|ultralytics"

# Check GPU availability:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check available RAM:
Get-CimInstance Win32_OperatingSystem | Select-Object @{Name="FreeRAM(GB)";Expression={[math]::Round($_.FreePhysicalMemory/1MB,2)}}, @{Name="TotalRAM(GB)";Expression={[math]::Round($_.TotalVisibleMemorySize/1MB,2)}}

# Check page file configuration:
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage
```

---

## Best Practices

### Before Training

1. **Close unnecessary applications** to free RAM
2. **Verify dataset paths** in YAML configuration
3. **Check GPU memory availability** with `nvidia-smi`
4. **Ensure page file is ≥8GB** (see Option A above)

### During Training

1. **Monitor GPU usage** with `nvidia-smi -l 1` in separate terminal
2. **Watch for OOM errors** - reduce batch size if needed
3. **Check disk space** - training saves checkpoints and plots
4. **Don't interrupt during checkpoint saves** (wait for epoch completion)

### After Training

1. **Review metrics** in `models/experiments/final/<run_name>/results.csv`
2. **Run validation script** to confirm statistical significance
3. **Deploy to production** if metrics meet requirements
4. **Archive training metadata** for reproducibility

---

## Contact & Support

For issues not covered here:
- Check the main [README.md](../README.md)
- Review training logs in `models/experiments/final/<run_name>/`
- Inspect SLURM-specific guidance in [SLURM_QUICKSTART.md](../SLURM_QUICKSTART.md)

---

*Last updated: November 23, 2025*
