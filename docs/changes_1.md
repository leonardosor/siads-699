# Dev Container Setup

Uses the root `../Dockerfile` with PostgreSQL and VS Code integration.

## Quick Start

**Prerequisites:** Docker Desktop + VS Code "Dev Containers" extension

**Start:**
```bash
docker-compose build ocr-cnn
code .
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

**Verify:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
tesseract --version
psql -h localhost -U postgres  # password: postgres
```

## What's Included

- PyTorch 2.0+ with CUDA, Tesseract OCR, ML dependencies
- PostgreSQL database
- VS Code extensions (Python, Jupyter, Docker, SQLTools)
- Ports: 5432 (PostgreSQL), 6006 (TensorBoard), 8888 (Jupyter)

## Common Commands

```bash
# Train model
python src/train.py --config config/train_config.yaml

# TensorBoard
tensorboard --logdir outputs/experiment_1/logs --host 0.0.0.0

# Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Database
psql -h localhost -U postgres
```

## Configuration

**No GPU?** Comment out `deploy` section in `docker-compose.yml` (lines 20-27)

**Add packages:** Edit `../requirements.txt` and rebuild

**Add extensions:** Edit `devcontainer.json` and rebuild

## Troubleshooting

- Build fails: Check `docker version`
- Need rebuild: Ctrl+Shift+P → "Rebuild Container Without Cache"
- Can't connect to DB: Wait 30s for healthcheck
