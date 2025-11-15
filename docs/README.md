# SIADS 699 - MADS Capstone
## Financial Form Text Extractor

### Overview
We are attempting to build a state of the art, full stack, text extraction application that performs on scanned images of forms. Our application's architecture will include manually labelled (Label Studio) training images (PNG, JPEG, PDFs) with bounded boxes of "header", "body" and "footer" text. This data will be used to fine tune a convolutional neural network (Yolov8). Our application will then extract text only from the body of forms (Tesseract 5). This extracted data will be stored in a relational database (postgreSQL). Our application will ultimately follow a micro-services pattern, implemented with Docker, and tied together in a web application front-end (Streamlit).

### AI Assistance Disclosure
Consistent with MADS academic integrity guidelines, assume that OpenAI's ChatGPT 5 (large language model, 2024 release) materially assisted in producing the source code within this repository. When citing or reusing this work, please include the acknowledgement: *OpenAI. (2024). ChatGPT 5 (large language model) [Computer software]. Assistance provided to the SIADS 699 Financial Form Text Extractor project.*

### Repository Architecture
```bash
.
├── README.md
├── yolov8n.pt             # Base model weight file
├── archive/               # Archived files and duplicate scripts
├── data/                  # Data directory
│   ├── input/
│   │   └── ground-truth/  # Ground truth data for evaluation
│   ├── output/
│   │   ├── ocr_results/   # OCR processing results (CSV/JSON)
│   │   └── visualizations/
│   └── raw/
│       └── rvl-cdip-invoice/  # RVL-CDIP invoice dataset
├── docs/
│   ├── imgs/              # Architecture diagrams
│   ├── other/             # Citations, requirements, common commands
│   └── google-drive-sharing.md
├── models/                # Model weights and training runs (see models/README.md)
│   ├── pretrained/        # Base YOLOv8 weights
│   ├── trained/           # Production models (best.pt, active_run.txt)
│   ├── runs/              # Successful training experiments
│   ├── archive/           # Historical/incomplete runs
│   └── artifacts/
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Utility scripts
│   └── ocr/               # OCR-related scripts
└── src/                   # Source code (see src/README.md)
    ├── config/            # Configuration templates
    ├── database/          # Database schemas and notebooks
    │   ├── init-db.sql    # PostgreSQL initialization
    │   └── postgresql.ipynb
    ├── environments/      # Docker and Conda configurations
    │   ├── docker/        # Dockerfile, compose.yml, requirements.txt
    │   └── conda/         # Great Lakes HPC environment
    ├── models/            # Model-related code
    ├── processing/        # OCR processing pipeline
    │   └── ocr_processor.py
    ├── training/          # Model training scripts
    │   ├── train.py
    │   ├── batch_job.sh
    │   └── finance-image-parser.yaml
    ├── utils/             # Utilities (dataset, models, deployment, monitoring)
    │   ├── dataset/       # Dataset utilities (count, preview, remap labels)
    │   ├── deployment/    # Deployment scripts (sync, reset)
    │   ├── models/        # Model management utilities
    │   └── monitoring/    # Training monitoring tools
    └── web/               # Streamlit application
        └── streamlit_application.py
```

### Product Architecture - Logical
![Logical](/docs/imgs/architecture_1.png)

### Product Architecture - Physical
![Physical](/docs/imgs/architecture_0.png)

### Dockerized Development
- All container build assets live in `src/environments/docker/` (`Dockerfile`, `compose.yml`, `requirements.txt`).
- From the project root, `docker compose -f src/environments/docker/compose.yml ...` builds the Streamlit image, mounts the host's `./models` and `./src` directories, and provisions PostgreSQL with `src/database/init-db.sql`.
- Default credentials/ports are injected via environment variables (`APP_PORT`, `DB_PORT`, `POSTGRES_*`, `MODEL_PATH`). Override them by creating a `.env` file in the project root.
- Ensure your YOLO weights live at `models/trained/best.pt` (or point `MODEL_PATH` elsewhere).
- For local installs outside Docker run `pip install -r src/environments/docker/requirements.txt`; Conda users can use `src/environments/conda/great-lakes-env.yml`.

### Quickstart
1. Clone the repo and enter the directory.
2. Copy your best YOLO weights to `models/trained/best.pt` (alternatively set `MODEL_PATH`).
3. Launch everything: `docker compose -f src/environments/docker/compose.yml up --build --remove-orphans`.
4. Open [http://localhost:8501](http://localhost:8501) to use the Streamlit UI.
5. Tear down when finished: `docker compose -f src/environments/docker/compose.yml down --volumes`.

Spin up the full stack with a single command:

```bash
docker compose -f src/environments/docker/compose.yml up --build --remove-orphans
```

Shut everything down with `docker compose -f src/environments/docker/compose.yml down --volumes` once you're finished testing.

### Streamlit Inference Workflow
1. Place your best YOLO checkpoint at `models/trained/best.pt` (compose mounts it inside `/app/models`).
2. Launch the stack with the compose command above and browse to `http://localhost:8501`.
3. Upload one or more JPG/PNG scans. The app runs YOLOv8 inference, draws maize labels with blue outlines over every detection, and lists coordinates/confidence in a table.
4. Adjust confidence/IoU sliders in the sidebar to tighten or loosen detections.
5. Download the UM-branded annotated image directly from the UI for documentation or model comparisons.

For detailed usage instructions, see [src/web/README.md](src/web/README.md).

### Managing YOLO Runs & Archives
- **Successful training runs** live under `models/runs/<run-name>/` (metrics, configs, preview images, weights). Git tracks the metadata, while `.pt/.pth` binaries remain ignored.
- **Production models** live in `models/trained/`:
  - `best.pt` - Currently deployed model
  - `active_run.txt` - Name of the training run currently in use
- **Historical/incomplete runs** are archived in `models/archive/` to keep `runs/` clean.
- **Base models** (pre-trained YOLOv8) are in `models/pretrained/`.

See [models/README.md](models/README.md) for detailed model management documentation.

#### Download and Deploy from Great Lakes
Pull a run down from Great Lakes and deploy in one step:
```bash
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826
```
This automatically:
- Downloads the run to `models/runs/<run-name>/`
- Copies `best.pt` to `models/trained/best.pt`
- Updates `models/trained/active_run.txt`
- Rebuilds and restarts Streamlit containers

Environment overrides if needed:
- `REMOTE_USER` (default `joehiggi`)
- `REMOTE_HOST` (default `greatlakes.arc-ts.umich.edu`)
- `REMOTE_PROJECT` (default `/home/$REMOTE_USER/siads-699`)

Add `--no-best` to skip copying weights or `--no-restart` to skip Streamlit restart.

#### Switch Models Locally
Switch to any run already downloaded:
```bash
src/utils/models/set_active_run.sh finance-parser-20251112_143826
docker compose -f src/environments/docker/compose.yml restart app
```

#### Reset to Baseline
Revert to a known-good baseline model:
```bash
src/utils/deployment/reset_streamlit_model.sh
```
Override `BASELINE_RUN` or pass `--weights <path>` for a different checkpoint. Add `--no-restart` to skip Streamlit restart.

#### Training on Great Lakes
See [src/training/README.md](src/training/README.md) for comprehensive training documentation.

### Data
- [Google Drive][1]

### Docker Container
The image build includes a PostgreSQL Alpine 15 database, Debian Frontend and Ultralytics (PyTorch, YOLO, OpenCV with GPU) support. As part of the PostgreSQL set-up we're including a database init file that creates a few tables with pre-set columns and data types as a general outline of MLOps data.

### Documentation
- [rvl-cdip-invoice][2]
- [HuggingFace Co-Lab][3]
- [Yolov8][4]
- [Tesseract 5][5]

[1]: https://drive.google.com/drive/folders/1ibqk_GzowWrwybOqg8wA88Q95gKQnrM1?usp=share_link
[2]: https://huggingface.co/datasets/chainyo/rvl-cdip-invoice
[3]: https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/save_load_dataset.ipynb#scrollTo=091FrwQDXQiM
[4]: https://arxiv.org/html/2408.15857
[5]: https://tesseract-ocr.github.io/tessdoc/
