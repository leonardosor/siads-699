# Project Structure

## Directory Organization

```
.
├── data/                      # Data files (gitignored except structure)
│   ├── input/
│   │   ├── ground-truth/      # 100 manually annotated images (source of truth)
│   │   ├── training/          # Generated training split (images + labels)
│   │   ├── validation/        # Generated validation split
│   │   └── testing/           # Generated test split
│   ├── output/                # OCR results, visualizations
│   └── raw/                   # Raw datasets (RVL-CDIP, parquet files)
│
├── models/                    # Model weights and training artifacts
│   ├── pretrained/            # Base YOLOv8 weights (yolov8n.pt)
│   ├── trained/               # Production models (best.pt, active_run.txt)
│   ├── runs/                  # Successful training experiments
│   ├── archive/               # Historical/incomplete runs
│   └── artifacts/             # Compressed run archives
│
├── src/                       # All source code
│   ├── config/                # Configuration templates
│   ├── database/              # PostgreSQL schemas and notebooks
│   │   ├── init-db.sql        # Database initialization
│   │   └── postgresql.ipynb   # Database exploration
│   ├── environments/          # Environment configurations
│   │   ├── docker/            # Dockerfile, compose.yml, requirements.txt
│   │   └── conda/             # Great Lakes HPC environment (great-lakes-env.yml)
│   ├── processing/            # OCR processing pipeline
│   │   └── ocr_processor.py   # Batch OCR with YOLO + Tesseract
│   ├── training/              # Model training scripts
│   │   ├── train.py           # Main training script with Optuna support
│   │   ├── batch_job.sh       # SLURM job for Great Lakes
│   │   └── finance-image-parser.yaml  # Dataset configuration
│   ├── utils/                 # Utility scripts organized by purpose
│   │   ├── dataset/           # Dataset preparation and validation
│   │   ├── deployment/        # Model deployment automation
│   │   ├── models/            # Model management utilities
│   │   └── monitoring/        # Training monitoring tools
│   └── web/                   # Streamlit web application
│       └── streamlit_application.py
│
├── tests/                     # Test suite (structure exists, minimal tests)
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── fixtures/              # Test fixtures
│   └── conftest.py            # Pytest configuration
│
├── docs/                      # Documentation and references
│   ├── imgs/                  # Architecture diagrams
│   ├── references/            # Research papers, citations
│   ├── COMMANDS.md            # Common command reference
│   ├── GPU_SETUP.md           # GPU configuration guide
│   └── README.md              # Main documentation
│
├── notebooks/                 # Jupyter notebooks for exploration
├── scripts/                   # Legacy/utility scripts
└── config/                    # Project-level configs (docker-compose, pre-commit)
```

## Key File Conventions

### Model Files
- `models/trained/best.pt` - Currently deployed model (symlink or copy)
- `models/trained/active_run.txt` - Name of active training run
- Training runs named: `finance-parser-YYYYMMDD_HHMMSS`

### Dataset Files
- Images: `.jpg`, `.png` formats
- Labels: YOLO format `.txt` files (one per image, same basename)
- Label format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
- Classes: 0=header, 1=body, 2=footer

### Configuration Files
- `src/training/finance-image-parser.yaml` - YOLO dataset config
- `src/database/init-db.sql` - Database schema
- `src/environments/docker/compose.yml` - Docker orchestration
- `.env` - Environment variable overrides (gitignored)

## Code Organization Patterns

### Utilities Structure
Utilities are organized by purpose, not by technology:
- `utils/dataset/` - Dataset manipulation (count, preview, remap, prepare)
- `utils/models/` - Model management (list runs, set active)
- `utils/deployment/` - Deployment automation (sync, reset)
- `utils/monitoring/` - Training monitoring (tail logs)

### Environment Separation
- `src/environments/docker/` - Local development (CPU, Streamlit + PostgreSQL)
- `src/environments/conda/` - Great Lakes HPC (GPU, training only)

### Database Schema
Two parallel workflows:
1. **PDF Workflow**: documents_metadata → document_pages → ocr_results → ocr_bounding_boxes
2. **Parquet Workflow**: parquet_ocr_results, parquet_ocr_words, parquet_yolo_regions
3. **Training Metadata**: models, training_runs, training_epochs, predictions

## Naming Conventions

### Python Files
- Scripts: `snake_case.py` (e.g., `prepare_augmented_groundtruth.py`)
- Classes: `PascalCase` (e.g., `OCRProcessor`)
- Functions: `snake_case` (e.g., `load_model()`)
- Private functions: `_leading_underscore` (e.g., `_load_font()`)

### Shell Scripts
- Bash scripts: `kebab-case.sh` (e.g., `sync_yolov8_run.sh`)
- Executable: `chmod +x` for scripts in `utils/`

### Constants
- `UPPER_SNAKE_CASE` for module-level constants
- University of Michigan colors: `UM_BLUE = "#00274C"`, `UM_MAIZE = "#FFCB05"`

## Import Patterns

Standard import order:
1. `from __future__ import annotations` (for type hints)
2. Standard library imports
3. Third-party imports (pandas, torch, ultralytics, etc.)
4. Local imports

Example:
```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from ultralytics import YOLO

from src.utils.helpers import load_config
```

## Path Handling

- Always use `pathlib.Path` for file operations
- Paths relative to repository root
- Find repo root by looking for `src/`, `data/`, `models/` directories
- Use `Path(__file__).resolve().parents[N]` to traverse up from script location

## Git Conventions

### Gitignored
- `data/` - All data files (too large)
- `models/**/*.pt`, `models/**/*.pth` - Model weights
- `.env` - Environment secrets
- `__pycache__/`, `*.pyc` - Python cache
- `.DS_Store` - macOS metadata

### Tracked
- All source code in `src/`
- Documentation in `docs/`
- Configuration files (YAML, SQL, Dockerfile)
- `models/trained/active_run.txt` - Active model reference
- Training run metadata (results.csv, args.yaml)

## Docker Volume Mounts

- `models/` → `/app/models` (read-only for app)
- `src/` → `/app/src` (read-write for development)
- `src/database/init-db.sql` → `/docker-entrypoint-initdb.d/init.sql` (PostgreSQL init)

## Development Workflow

1. **Data Preparation**: Use `utils/dataset/` scripts to prepare training data
2. **Training**: Run on Great Lakes HPC via `batch_job.sh`
3. **Deployment**: Sync model with `utils/deployment/sync_yolov8_run.sh`
4. **Testing**: Validate via Streamlit UI at http://localhost:8501
5. **Iteration**: Adjust hyperparameters, retrain, compare metrics
