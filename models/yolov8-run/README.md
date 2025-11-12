# yolov8-run

Minimal toolkit for fine-tuning YOLOv8 on the University of Michigan Great Lakes (ARC) cluster.
Only the dataset, pretrained weights, training entry point, and Slurm job wrappers are kept so the repo stays light and portable.

## Layout

```
├── data/                # YOLO-format dataset (train/val/test splits under input/)
├── models/              # Base checkpoints to fine-tune from (e.g., yolov8n.pt)
├── src/
│   ├── yolo_v8/         # train.py entry point + finance-image-parser.yaml
│   └── utilities/       # Slurm batch scripts for Great Lakes
├── environment.yml      # Conda environment definition for GPU training
├── requirements.txt     # Optional pip install list (mirrors environment.yml)
└── runs/, artifacts/    # Created at runtime; ignored by git
```

## Great Lakes setup

1. Clone / sync (this training kit now lives inside `siads-699/models`)
   ```bash
   ssh uniqname@login.greatlakes.arc-ts.umich.edu
   cd /home/$USER
   git clone <your-fork-url> siads-699
   cd siads-699/models/yolov8-run
   ```

2. Create the Conda env
   ```bash
   module load mamba/py3.12
   source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
   mamba env create -n yolov8-env -f environment.yml
   conda activate yolov8-env
   ```

3. Optional CUDA sanity check
   ```bash
   python - <<'PY'
   import torch, ultralytics
   print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
   print("Ultralytics:", ultralytics.__version__)
   PY
   ```

## Running training

### Direct invocation

```
python src/yolo_v8/train.py \
  --weights models/yolov8n.pt \
  --data-config src/yolo_v8/finance-image-parser.yaml \
  --epochs 150 \
  --batch 8 \
  --imgsz 640 \
  --cos-lr \
  --clean-broken
```

Arguments map to Ultralytics CLI flags, so adjust learning rate, batch, or image size as needed. Artifacts land in `runs/detect/<run-name>/`.

### Slurm batch jobs

`src/utilities/batch_job.sh` runs the full training schedule (defaults: 200 epochs, batch 8, 640 px).  
`src/utilities/batch_job_10_epochs.sh` is a quick smoke test (10 epochs).

Override defaults via environment variables when submitting:

```bash
cd /home/$USER/siads-699/models/yolov8-run
RUN_NAME=finance-v1 EPOCHS=180 BATCH=6 IMGSZ=768 sbatch src/utilities/batch_job.sh
```

Each job activates `yolov8-env`, calls `src/yolo_v8/train.py`, saves outputs to `/home/$USER/siads-699/models/yolov8-run/runs/detect/<run-name>/`, and tars the folder into `/home/$USER/siads-699/models/yolov8-run/artifacts/<run-name>.tar.gz` for easy downloading.

### Live output with `srun`

If you want to watch the logs stream in real time, run the helper from the root of `siads-699` (or set `PROJECT_ROOT`):

```bash
cd /home/$USER/siads-699
./models/yolov8-run/src/utilities/run_with_srun.sh
```

It wraps `srun --partition=gpu --gres=gpu:1 ... src/utilities/batch_job.sh`, so you still get the same packaging behavior but with live console output. Customize resources via env vars, e.g. `CPUS=8 MEM=24G JOB_NAME=finance-live ./models/yolov8-run/src/utilities/run_with_srun.sh`.

Need a quick queue snapshot plus automatic fallback between partitions? Use:

```bash
cd /home/$USER/siads-699
./models/yolov8-run/src/utilities/queue_and_run.sh
```

By default it shows the top 10 jobs in `gpu` and `spgpu`, then tries each partition (in that order) via `srun`. Override priorities or resources by exporting `PARTITION_LIST="spgpu gpu"`, `CPUS=2`, `MEM=8G`, etc., before running the script.
Partitions reported as `down`, `drain`, or `maint` in `sinfo` are skipped automatically.

## Retrieving results

```bash
scp -r uniqname@login.greatlakes.arc-ts.umich.edu:/home/uniqname/siads-699/models/yolov8-run/runs/detect/<run-name> ./runs/
# or pull the tarball
scp uniqname@login.greatlakes.arc-ts.umich.edu:/home/uniqname/siads-699/models/yolov8-run/artifacts/<run-name>.tar.gz .
tar -xzf <run-name>.tar.gz -C ./runs
```

Each run folder contains `results.png`, `confusion_matrix.png`, `F1_curve.png`, `results.csv`, and `weights/{best,last}.pt`.

Update the dataset under `data/`, push, and submit a new batch job whenever new labels are ready.
