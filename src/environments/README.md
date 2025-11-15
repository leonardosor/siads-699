# Environment Configuration

This directory contains environment configurations for different deployment targets.

## Requirements Files

### `requirements-prod.txt`
Production deployment requirements for CPU-based inference (minimal dependencies):
- PyTorch CPU version
- Streamlit for web interface
- Core OCR and ML libraries
- Use this for Docker deployments

### `requirements-dev.txt`
Full development environment with GPU support:
- PyTorch with CUDA support
- Development tools (jupyter, pytest, black, ruff)
- Complete ML stack (scikit-learn, seaborn, tensorboard)
- Data processing libraries
- Use this for local development and training

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
