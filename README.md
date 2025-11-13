# SIADS 699 - MADS Capstone
## Financial Form Text Extractor

### Overview
We are attempting to build a state of the art, full stack, text extraction application that performs on scanned images of forms. Our application's architecture will include manually labelled (Label Studio) training images (PNG, JPEG, PDFs) with bounded boxes of "header", "body" and "footer" text. This data will be used to fine tune a convolutional neural network (Yolov8). Our application will then extract text only from the body of forms (Tesseract 5). This extracted data will be stored in a relational database (postgreSQL). Our application will ultimately follow a micro-services pattern, implemented with Docker, and tied together in a web application front-end (Streamlit).

### AI Assistance Disclosure
Consistent with MADS academic integrity guidelines, assume that OpenAI's ChatGPT 5 (large language model, 2024 release) materially assisted in producing the source code within this repository. When citing or reusing this work, please include the acknowledgement: *OpenAI. (2024). ChatGPT 5 (large language model) [Computer software]. Assistance provided to the SIADS 699 Financial Form Text Extractor project.*

### Repository Architecture
```{bash}
.
├── .devcontainer
├── docs
├── models
│   ├── best.pt / active_run.txt
│   ├── runs/ (see models/runs/README.md)
│   └── yolov8-run/ (GreatLakes training kit)
└── src
    ├── docker          # Dockerfile, Compose, dependency lock
    ├── env             # Conda specs (e.g., environment.yml)
    ├── great-lakes-env # HPC-specific notebooks/configs
    ├── scripts         # Database bootstrap + helper notebooks
    ├── streamlit       # Front-end application
    └── yolo_v8         # Training scripts and configs
```

### Product Architecture - Logical
![Logical](/docs/imgs/architecture_1.png)

### Product Architecture - Physical
![Physical](/docs/imgs/architecture_0.png)

### Dockerized Development
- All container build assets live in `src/docker/` (`Dockerfile`, `compose.yml`, dependency lock, and helper scripts).
- `src/docker/compose.yml` builds the Streamlit image, mounts `./models` and `./src`, and provisions PostgreSQL with `src/scripts/init-db.sql`.
- Default credentials/ports are injected via environment variables (`APP_PORT`, `DB_PORT`, `POSTGRES_*`, `MODEL_PATH`). Override them inline or inject a `.env`.
- Ensure your YOLO weights live at `models/best.pt` (or point `MODEL_PATH` elsewhere).
- For local installs outside Docker run `pip install -r src/docker/requirements.txt`; Conda users can apply `src/env/environment.yml`.

### Quickstart
1. Clone the repo and enter the directory.
2. Copy your best YOLO weights to `models/best.pt` (alternatively set `MODEL_PATH`).
3. Launch everything: `docker compose -f src/docker/compose.yml up --build --remove-orphans`.
4. Open [http://localhost:8501](http://localhost:8501) to use the Streamlit UI.
5. Tear down when finished: `docker compose -f src/docker/compose.yml down --volumes`.

Spin up the full stack with a single command:

```bash
docker compose -f src/docker/compose.yml up --build --remove-orphans
```

Shut everything down with `docker compose -f src/docker/compose.yml down --volumes` once you're finished testing.

### Streamlit Inference Workflow
1. Place your best YOLO checkpoint at `models/best.pt` (compose mounts it inside `/app/models`).
2. Launch the stack with the compose command above and browse to `http://localhost:8501`.
3. Upload one or more JPG/PNG scans. The app runs YOLOv8 inference, draws maize labels with blue outlines over every detection, and lists coordinates/confidence in a table.
4. Adjust confidence/IoU sliders in the sidebar to tighten or loosen detections.
5. Download the UM-branded annotated image directly from the UI for documentation or model comparisons.

### Managing YOLO Runs & Archives
- Every training run lives under `models/runs/<run-name>/` (metrics, configs, preview images, weights). Git tracks the metadata, while `.pt/.pth` binaries remain ignored.
- `models/active_run.txt` records which run last copied its checkpoint into `models/best.pt`.
- Keep the currently deployed checkpoint at `models/best.pt`; replace it whenever you want Streamlit to pick up a new model (the helper below can copy it automatically).
- Need to revert? `./src/scripts/reset_streamlit_model.sh` restores the baseline run's `best.pt` (override `BASELINE_RUN` or pass `--weights <path>` if you prefer another checkpoint) and restarts Streamlit unless you add `--no-restart`.
- Pull a run down from Great Lakes and refresh the active weights in one step:
  ```bash
  ./src/scripts/sync_yolov8_run.sh finance-parser-20251112_143826
  ```
  Environment overrides if needed:
  - `REMOTE_USER` (default `joehiggi`)
  - `REMOTE_HOST` (default `greatlakes.arc-ts.umich.edu`)
  - `REMOTE_PROJECT` (default `/home/$REMOTE_USER/siads-699/models/yolov8-run`)
  - `LOCAL_RUNS_DIR` (default `models/runs`)
  Add `--no-best` to skip copying `weights/best.pt` into `models/best.pt`.
- Switch to any archived run locally without re-downloading:
  ```bash
  ./src/scripts/set_active_run.sh finance-image-parser4
  docker compose -f src/docker/compose.yml up --build --remove-orphans -d
  ```

### Improving YOLO Results on GreatLakes
1. **Sync code**  
   ```bash
   rsync -av --delete \
     --exclude ".git" \
     --exclude ".venv" \
     --exclude "__pycache__" \
     ./ joehiggi@greatlakes.arc-ts.umich.edu:/home/joehiggi/siads-699/
   ```
2. **Create/activate the training environment**:  
   ```bash
   module load python/3.10.8
   conda env create -f src/great-lakes-env/environment.yml  # first time only
   conda activate yolov8-env
   ```
3. **Audit class balance with the new counter** (point the config at your dataset paths):  
   ```bash
   python src/scripts/count_yolo_labels.py --data-config /home/joehiggi/siads-699/src/yolo_v8/finance-image-parser.yaml
   ```
   Fix mislabeled/imbalanced splits before retraining.
4. **Launch the tuned training job** (defaults: 150 epochs, imgsz 1024, batch 4, patience 60, mosaic off, cache on):  
   ```bash
   cd /home/joehiggi/siads-699/models/yolov8-run
   CUDA_MODULE=cuda/12.0.0 RUN_NAME=body-focus sbatch src/batch_job.sh
   ```
   Replace `cuda/12.0.0` with whichever CUDA module `module spider cuda` lists on GreatLakes (set `CUDA_MODULE` only if needed). Override hyperparameters via env vars (`EPOCHS`, `IMGSZ`, `BATCH`, `HYPERPARAMS`, etc.) or run interactively with `srun`.
5. **Monitor performance** with `squeue -u joehiggi` and `tail -f /home/joehiggi/<job-name>-<jobid>.log`.
6. **Pull down the resulting checkpoint**:  
   ```bash
   ./src/scripts/sync_yolov8_run.sh body-focus
   ```
   Streamlit rebuilds automatically with the new weights; `./src/scripts/reset_streamlit_model.sh` restores the baseline model if needed.

### Quick HPC Retrain (one-liners)
```bash
# Push repo up (skip git + venv junk)
rsync -av --delete --exclude ".git" --exclude ".venv" --exclude "__pycache__" ./ \
  joehiggi@greatlakes.arc-ts.umich.edu:/home/joehiggi/siads-699/

# Submit job from GL login node
ssh joehiggi@greatlakes.arc-ts.umich.edu <<'EOF'
cd /home/joehiggi/siads-699/models/yolov8-run
module load python/3.10.8
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
conda activate yolov8-env
CUDA_MODULE=cuda/12.0.0 RUN_NAME=body-focus2 sbatch src/batch_job.sh
EOF

# After it finishes, sync + redeploy
./src/scripts/sync_yolov8_run.sh body-focus2
```

### Label QA (catch header/body swaps)
- Visualize any labeled page to ensure class IDs align with expectations:
  ```bash
  python src/scripts/preview_yolo_labels.py \
    --image models/yolov8-run/data/input/training/images/example.jpg \
    --labels models/yolov8-run/data/input/training/labels/example.txt \
    --output preview.jpg
  ```
- Review the generated `preview.jpg` and confirm each colored box matches its intended class. Fix any mislabeled `.txt` files before retraining.

### Data
- [Google Drive][1]

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
- Fix a class-ID mix-up after label QA:
  ```bash
  python src/scripts/remap_yolo_labels.py \
    --root models/yolov8-run/data/input \
    --map 0:1 1:2 2:0
  python src/scripts/count_yolo_labels.py --data-config src/yolo_v8/finance-image-parser.yaml
  ```
  (The example mapping swaps header→body, body→footer, footer→header.)
