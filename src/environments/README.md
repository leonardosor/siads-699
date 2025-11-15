# Environment Configuration

This directory contains environment configurations for different deployment targets.

## Requirements Files

The project uses a **hierarchical requirements structure** to avoid duplication:

```
requirements.txt           ← Base dependencies (shared)
├── requirements-prod.txt  ← Production (extends base + CPU PyTorch)
└── requirements-dev.txt   ← Development (extends base + GPU PyTorch + dev tools)
```

### `requirements.txt` (Base)
Core dependencies shared across all environments:
- OCR & PDF processing (pytesseract, pdf2image, pdfplumber)
- Data science libraries (numpy, pandas, scipy)
- Web framework (Streamlit)
- Database (PostgreSQL)
- Utilities (requests, pyyaml, optuna)

### `requirements-prod.txt` (Production)
Extends base with CPU-only PyTorch for production deployment:
- PyTorch CPU version (smaller, faster for inference)
- Minimal footprint for Docker containers
- **Use this for Docker deployments**

### `requirements-dev.txt` (Development)
Extends base with GPU support and development tools:
- PyTorch with CUDA support (for training)
- Development tools (jupyter, pytest, black, ruff)
- Visualization (matplotlib, seaborn, tensorboard)
- Data processing (polars, pyarrow)
- **Use this for local development and training**

## Directory Structure

### `conda/`
Contains Conda environment specifications for HPC environments (e.g., Great Lakes).

### `docker/`
Docker-specific configuration files:
- `Dockerfile` - Container image definition
- `compose.yml` - Docker Compose orchestration
- `.dockerignore` - Files to exclude from Docker builds
- `.env` - Docker environment variables

## Usage

### Local Development
```bash
pip install -r requirements-dev.txt
```

### Production Deployment
```bash
pip install -r requirements-prod.txt
```

### HPC (Great Lakes)
```bash
conda env create -f conda/great-lakes-env.yml
```

### Docker
```bash
cd docker/
docker-compose up
```

## Benefits of Hierarchical Structure

1. **No Duplication** - Core dependencies defined once in `requirements.txt`
2. **Easy Maintenance** - Update shared dependencies in one place
3. **Clear Separation** - Production vs development requirements clearly separated
4. **Flexible** - Easy to add new environment variants (e.g., requirements-test.txt)

## Troubleshooting

### Dependency Conflicts
If you encounter version conflicts:
```bash
# Clean install
pip uninstall -y -r requirements-dev.txt
pip install -r requirements-dev.txt
```

### CUDA Issues (Development)
If PyTorch doesn't detect your GPU:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If False, reinstall with explicit CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
