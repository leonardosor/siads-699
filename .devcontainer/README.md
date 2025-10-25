# Dev Container Setup

This dev container uses the same Docker image as the standalone setup (`../Dockerfile`) but adds PostgreSQL and VS Code integration.

## Quick Start

**Prerequisites:**
- Docker Desktop running
- VS Code with "Dev Containers" extension

**Two ways to start:**

1. **Build first (recommended):**
   ```bash
   docker-compose build ocr-cnn
   code .
   # Then: Ctrl+Shift+P → "Dev Containers: Reopen in Container"
   ```

2. **Let VS Code build:**
   ```
   Ctrl+Shift+P → "Dev Containers: Reopen in Container"
   ```

**Verify it's working:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
tesseract --version
psql -h localhost -U postgres  # password: postgres
```

## What You Get

From the root Dockerfile:
- PyTorch 2.0+ with CUDA
- Tesseract OCR
- All ML dependencies

Added by dev container:
- PostgreSQL database
- VS Code extensions (Python, Jupyter, Docker, SQLTools)
- Automatic port forwarding

## Common Commands

```bash
# Train a model
python src/train.py --config config/train_config.yaml

# Start TensorBoard
tensorboard --logdir outputs/experiment_1/logs --host 0.0.0.0

# Start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access database
psql -h localhost -U postgres
```

**Ports:** 5432 (PostgreSQL), 6006 (TensorBoard), 8888 (Jupyter)

## No GPU?

Edit `.devcontainer/docker-compose.yml` and comment out the `deploy` section (lines 20-27), then rebuild.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails | Check Docker is running: `docker version` |
| Need to rebuild | Ctrl+Shift+P → "Rebuild Container Without Cache" |
| Can't connect to DB | Wait 30 seconds for healthcheck |
| Extensions missing | Install manually from Extensions panel |

## Customization

**Add Python packages:**
1. Add to `../requirements.txt`
2. Rebuild container

**Add VS Code extensions:**
1. Edit `devcontainer.json` extensions list
2. Rebuild container

**Add system packages:**
1. Edit `../Dockerfile`
2. Rebuild container

## How It Works

The dev container uses the same image as `docker-compose build ocr-cnn` but adds:
- PostgreSQL container
- VS Code configuration
- Port forwarding

This means you can build the image once and use it for both development (VS Code) and production (standalone Docker).

## File Structure

```
.devcontainer/
├── docker-compose.yml    # Uses ../Dockerfile + adds PostgreSQL
├── devcontainer.json     # VS Code config
└── README.md            # This file

Root:
├── Dockerfile           # Shared ML/OCR image
└── docker-compose.yml   # Standalone setup
```

## Tips

- Use the integrated terminal in VS Code (it runs inside the container)
- Files automatically sync between container and host
- Data in `data/`, `checkpoints/`, `outputs/` persists on your machine
- Both dev container and standalone Docker use the same image

## Resources

- [What Changed](CHANGES.md)
- [Project Docs](../README.md)
- [Standalone Docker](../DOCKER_SETUP.md)
