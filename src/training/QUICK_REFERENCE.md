# Quick Reference: Training the Best Model

## Installation

First, install Optuna:
```bash
pip install optuna>=3.5.0
```

## Usage

### Standard Training (Good Defaults)
```bash
python src/training/train.py
```

### Hyperparameter Optimization with Optuna
```bash
python src/training/train.py --optimize --n-trials 20
```

This will:
1. Run 20 trials testing different hyperparameter combinations
2. Track results in SQLite database (`models/runs/optuna_study.db`)
3. Train final model with best parameters (3x epochs)

### Custom Training Options
```bash
# Longer training
python src/training/train.py --epochs 300

# Enable caching for faster epochs
python src/training/train.py --cache --epochs 200

# Clean broken images first
python src/training/train.py --clean-broken --cache

# Custom device (CPU or multi-GPU)
python src/training/train.py --device cpu
python src/training/train.py --device 0,1,2,3

# Named run
python src/training/train.py --name my-experiment-v1
```

## Optuna Optimization

### Quick Optimization (Fast)
```bash
python src/training/train.py --optimize --n-trials 10 --epochs 50
```

### Thorough Optimization (Best Results)
```bash
python src/training/train.py --optimize --n-trials 50 --epochs 100 --cache
```

### Continue Previous Optimization
```bash
# Optuna automatically resumes from the database
python src/training/train.py --optimize --n-trials 30
```

## Parameters Being Optimized

Optuna automatically tunes:
- **Learning rate** (lr0, lrf)
- **Optimizer** (SGD, Adam, AdamW)
- **Batch size** (8, 16, 32)
- **Momentum & weight decay**
- **Augmentation** (mosaic, fliplr, rotation, HSV, mixup)

## Deploy Best Model

After training:
```bash
# Copy to production
cp models/runs/finance-parser-*/weights/best.pt models/trained/best.pt

# Update active run
echo "finance-parser-20251115_143030" > models/trained/active_run.txt

# Restart Streamlit
docker compose -f src/environments/docker/compose.yml restart app
```

## View Optimization Results

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="yolov8_optimization",
    storage="sqlite:///models/runs/optuna_study.db"
)

# Best trial
print(f"Best mAP: {study.best_value}")
print(f"Best params: {study.best_params}")

# Optimization history
import plotly
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
```

## Output Structure

```
models/runs/
├── optuna_study.db              # Optimization database
├── trial_0/                     # First trial
├── trial_1/                     # Second trial
├── ...
└── finance-parser-20251115_*/   # Final model with best params
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    └── results.csv
```

## Tips

✅ Start with `--optimize --n-trials 20` for good hyperparameters  
✅ Use `--cache` to speed up repeated epochs  
✅ Use `--clean-broken` on first run  
✅ Increase `--n-trials` for better optimization (20-50 recommended)  
✅ Study database persists - you can continue optimization later  

## Troubleshooting

**Out of Memory?**
```bash
# Reduce epochs per trial
python src/training/train.py --optimize --epochs 30
```

**Want faster trials?**
```bash
# Use smaller epoch count for trials
python src/training/train.py --optimize --n-trials 30 --epochs 50
```
