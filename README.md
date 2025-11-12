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
├── data
│   ├── input
│   │   ├── ground-truth
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   └── output
├── docs
├── models
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
- `src/docker/compose.yml` builds the Streamlit image, mounts `./models`, `./data`, and `./src`, and provisions PostgreSQL with `src/scripts/init-db.sql`.
- Default credentials/ports are injected via environment variables (`APP_PORT`, `DB_PORT`, `POSTGRES_*`, `MODEL_PATH`). Override them inline or inject a `.env`.
- Ensure your YOLO weights live at `models/best.pt` (or point `MODEL_PATH` elsewhere).
- For local installs outside Docker run `pip install -r src/docker/requirements.txt`; Conda users can apply `src/env/environment.yml`.

### Quickstart (students)
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

### Improving YOLO Results on GreatLakes
1. **Sync code + data**  
   ```bash
   scp -r . youruniqname@greatlakes.arc-ts.umich.edu:/scratch/youruniqname/siads-699
   ```
2. **Create the training environment** (one-time):  
   ```bash
   module load python/3.10.8
   conda env create -f src/great-lakes-env/environment.yml
   conda activate yolov8-env
   ```
3. **Launch training** with Ultralytics (edit `src/yolo_v8/finance-image-parser.yaml` to point at your dataset):  
   ```bash
   yolo detect train \
     model=yolov8n.pt \
     data=src/yolo_v8/finance-image-parser.yaml \
     epochs=50 imgsz=640 batch=16 \
     project=src/yolo_v8/runs/detect name=finance-image-parser
   ```
4. **Monitor performance** by tailing `results.csv` in the run directory or running `tensorboard --logdir src/yolo_v8/runs`.
5. **Export the best checkpoint** (`runs/detect/<run>/weights/best.pt`) back to your laptop and copy it to `models/best.pt` for inference.

### Tips for Better Inference Accuracy
- **More epochs & patience**: run at least 50 epochs on GreatLakes; enable `patience=20` to keep training as long as validation improves.
- **Balanced dataset**: ensure each class (header/body/footer) has similar representation. Use `weighted_boxes_fusion=False` if you only have one model.
- **High-resolution inputs**: training with `imgsz=1024` often yields crisper boxes on forms; adjust batch size accordingly.
- **Augmentation tuning**: disable aggressive mosaics for documents (`augment=False` or `degrees=0, scale=0.2`). Text layouts don’t benefit from heavy geometric transforms.
- **Validation split**: always include a held-out validation folder so metrics (mAP, precision, recall) reflect true generalization.
- **Post-processing**: within Streamlit, experiment with `confidence`/`iou` sliders—lowering confidence can surface faint detections, raising IoU can reduce overlapping boxes.

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
