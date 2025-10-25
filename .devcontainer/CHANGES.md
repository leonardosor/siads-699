# Dev Container Changes - Now Uses Root Dockerfile

## Summary

The `.devcontainer` setup now uses the **same Dockerfile** as the standalone setup (`../Dockerfile`), making it consistent with `docker-compose build ocr-cnn`.

## What Changed

### ✅ Removed
- `.devcontainer/Dockerfile` - No longer needed, now uses `../Dockerfile`

### ✅ Updated
- `.devcontainer/docker-compose.yml` - Now references `../Dockerfile`
- `.devcontainer/devcontainer.json` - Updated terminal to bash (not zsh)
- `.devcontainer/README.md` - Updated documentation

## Architecture Now

```
Before:
.devcontainer/
├── Dockerfile (separate image)
└── docker-compose.yml

Root:
├── Dockerfile (standalone image)
└── docker-compose.yml

= Two different images!

After:
.devcontainer/
└── docker-compose.yml (uses ../Dockerfile)

Root:
├── Dockerfile (shared image!)
└── docker-compose.yml

= Same image, just adds PostgreSQL for dev container!
```

## What This Means

### ✅ Advantages

1. **Single Docker Image**: Build once with `docker-compose build ocr-cnn`, use everywhere
2. **Consistency**: Dev and production environments identical
3. **Faster Builds**: No need to build separate images
4. **Easier Maintenance**: Update one Dockerfile, not two
5. **Disk Space**: ~50% savings (one image instead of two)

### The Only Difference

**Dev Container** = Root Dockerfile + PostgreSQL + VS Code integration

**Standalone** = Root Dockerfile only

## How to Use

### Option 1: Build with Root docker-compose (Your Preference)

```bash
# Build the image
docker-compose build ocr-cnn

# Then open in VS Code
# Ctrl+Shift+P → "Reopen in Container"
# VS Code will use the pre-built image
```

### Option 2: Let VS Code Build It

```bash
# Just open in VS Code
# Ctrl+Shift+P → "Reopen in Container"
# VS Code will build the image for you
```

Both options build the **same image** from `../Dockerfile`!

## What Gets Built

When you run `docker-compose build ocr-cnn`, it builds:
- PyTorch 2.0+ with CUDA
- Tesseract OCR
- All ML dependencies
- Everything from `requirements.txt`

When you open in VS Code dev container, it:
- Uses that same image
- Adds a PostgreSQL container
- Installs VS Code extensions
- Sets up port forwarding

## Quick Start

```bash
# 1. Build the image (if not already built)
docker-compose build ocr-cnn

# 2. Open in VS Code
code .

# 3. Reopen in container
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# 4. Start working!
python src/data/pdf_processor.py
python src/train.py --config config/train_config.yaml
```

## Files Structure

```
.devcontainer/
├── docker-compose.yml        ← References ../Dockerfile
├── devcontainer.json         ← VS Code config
├── README.md                 ← Usage guide
├── QUICK_REFERENCE.md        ← Quick commands
├── CHANGES.md               ← This file
└── *.backup                  ← Original files (safe to delete)

Root:
├── Dockerfile               ← SHARED IMAGE (GPU)
├── Dockerfile.cpu           ← CPU version
├── docker-compose.yml       ← Standalone setup
└── requirements.txt         ← Python packages
```

## GPU Support

Both setups support GPU! If you don't have a GPU, comment out the `deploy` section in `.devcontainer/docker-compose.yml` (lines 20-27).

## Database Access

Only available in dev container:
```bash
psql -h localhost -U postgres
# Password: postgres
```

## Troubleshooting

### "Image not found" when opening in VS Code

```bash
# Build the image first
docker-compose build ocr-cnn

# Or let VS Code build it
# Ctrl+Shift+P → "Rebuild Container"
```

### Want to use CPU version?

Edit `.devcontainer/docker-compose.yml` line 7:
```yaml
dockerfile: Dockerfile.cpu  # Change from Dockerfile
```

### Can I still use standalone Docker?

Yes! Both setups coexist:
- Dev container: `Ctrl+Shift+P` in VS Code
- Standalone: `docker-compose up -d ocr-cnn`

(But don't run both simultaneously - port conflicts!)

## Summary

You now have **one Docker image** that works for:
- ✅ VS Code development (with database)
- ✅ Standalone training/inference
- ✅ Production deployments
- ✅ CI/CD pipelines

Build once, use everywhere! 🚀
