# Streamlit Web Application

Interactive web interface for running YOLOv8 inference on financial document images.

## Features

- Upload multiple images (JPG/PNG)
- Real-time YOLOv8 object detection
- Adjustable confidence and IoU thresholds
- Visual detection results with bounding boxes
- Detection table with class labels and confidence scores
- Download annotated images
- University of Michigan branded (Maize and Blue colors)

## Quick Start

### Using Docker (Recommended)

```bash
# Start the application
docker compose -f src/environments/docker/compose.yml up -d

# Access the application
open http://localhost:8501

# View logs
docker compose -f src/environments/docker/compose.yml logs -f app

# Stop the application
docker compose -f src/environments/docker/compose.yml down
```

### Local Development

```bash
# Install dependencies
pip install -r src/environments/docker/requirements.txt

# Set model path (optional, defaults to models/trained/best.pt)
export MODEL_PATH=models/trained/best.pt

# Run Streamlit
streamlit run src/web/streamlit_application.py --server.port 8501
```

## Usage

1. **Upload Images**
   - Click "Browse files" or drag-and-drop
   - Supports JPG, JPEG, PNG formats
   - Multiple images can be processed at once

2. **Configure Detection**
   - **Confidence threshold** (0-1): Minimum confidence for detections
     - Lower = more detections, more false positives
     - Higher = fewer detections, higher precision
   - **IoU threshold** (0-1): Non-max suppression threshold
     - Lower = more aggressive filtering of overlapping boxes
     - Higher = keeps more overlapping detections

3. **View Results**
   - Annotated images show bounding boxes with labels
   - Detection table lists all detected objects
   - Colors:
     - Boxes: UM Blue (#00274C)
     - Labels: UM Maize (#FFCB05) background with blue text

4. **Download**
   - Click download button under each annotated image
   - PNG format with high quality

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/trained/best.pt` | Path to YOLOv8 weights |
| `STREAMLIT_SERVER_PORT` | `8501` | Port to run on |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Run in headless mode |

### Model Selection

The application loads the model from `MODEL_PATH` environment variable.

**To change models:**

```bash
# Option 1: Set environment variable
export MODEL_PATH=/path/to/model.pt
streamlit run src/web/streamlit_application.py

# Option 2: Use set_active_run script
src/utils/models/set_active_run.sh finance-parser-20251112_143826
docker compose -f src/environments/docker/compose.yml restart app
```

### Docker Configuration

Edit `src/environments/docker/compose.yml`:

```yaml
services:
  app:
    environment:
      MODEL_PATH: /app/models/trained/best.pt  # Change model
    ports:
      - "8501:8501"  # Change port if needed
```

## Architecture

### Key Components

**streamlit_application.py** structure:

```python
# Configuration
UM_BLUE = "#00274C"
UM_MAIZE = "#FFCB05"
DEFAULT_MODEL_PATH = ...

# Caching functions
@st.cache_resource
def load_model(weights_path) -> YOLO
    # Loads model once per session

# Main functions
def _format_detections(result) -> pd.DataFrame
    # Converts YOLO results to table

def _draw_detections(image, result) -> Image
    # Draws boxes on image with UM colors

def main()
    # Streamlit UI and processing logic
```

### Caching

The application uses Streamlit caching for performance:
- **Model loading** (`@st.cache_resource`): Loads model once, reuses across sessions
- **Font loading** (`@st.cache_data`): Loads font once per session

Models are only reloaded when:
1. Server restarts
2. Model file changes
3. Cache is manually cleared

## Customization

### Change Colors

Edit color constants in `streamlit_application.py`:

```python
UM_BLUE = "#00274C"   # Bounding box color
UM_MAIZE = "#FFCB05"  # Label background color
```

### Add New Features

Common additions:

**Export detections to CSV:**
```python
@st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="detections.csv",
    mime="text/csv"
)
```

**Batch processing:**
```python
# Process entire folder
folder_path = st.text_input("Folder path:")
if folder_path:
    images = Path(folder_path).glob("*.jpg")
    for img_path in images:
        # Process each image
```

**Confidence filtering per class:**
```python
conf_header = st.slider("Header confidence", 0.0, 1.0, 0.25)
conf_body = st.slider("Body confidence", 0.0, 1.0, 0.25)
conf_footer = st.slider("Footer confidence", 0.0, 1.0, 0.25)
```

## Troubleshooting

### Model Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'models/trained/best.pt'`

**Solution:**
```bash
# Ensure model exists
ls -la models/trained/best.pt

# Or set custom path
export MODEL_PATH=/path/to/valid/model.pt
```

### Out of Memory

**Error:** CUDA out of memory or system RAM exhausted

**Solutions:**
1. Reduce image size before upload
2. Process fewer images at once
3. Use CPU instead of GPU:
   ```python
   # In streamlit_application.py
   model = YOLO(str(weights_path))
   model.to("cpu")  # Force CPU
   ```

### Port Already in Use

**Error:** `Address already in use: 8501`

**Solution:**
```bash
# Kill existing process
docker compose -f src/environments/docker/compose.yml down

# Or use different port
docker compose -f src/environments/docker/compose.yml up -d --build
# Edit compose.yml ports: - "8502:8501"
```

### Slow Inference

**Causes:**
- Running on CPU instead of GPU
- Large images (>2000px)
- Large batch of images

**Solutions:**
1. Verify GPU usage:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```
2. Resize images before upload
3. Process images one at a time

### No Detections

**Possible reasons:**
1. Confidence threshold too high - lower it
2. Wrong model loaded - check `MODEL_PATH`
3. Image not similar to training data
4. Model not trained well

**Debug steps:**
```bash
# Check which model is loaded
docker compose -f src/environments/docker/compose.yml logs app | grep "MODEL_PATH"

# Try lowest confidence
# Set confidence slider to 0.01

# Check model performance
cat models/trained/active_run.txt
python src/utils/models/list_model_runs.py
```

## Performance Optimization

### GPU Acceleration

Docker automatically uses GPU if available. To verify:

```bash
# Check NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Update compose.yml to enable GPU
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Image Preprocessing

For better performance on large images:

```python
# Add to upload processing
max_size = 1280
if max(image.size) > max_size:
    image.thumbnail((max_size, max_size), Image.LANCZOS)
```

### Batch Processing

For multiple images, process in batches:

```python
batch_size = 4
for i in range(0, len(uploaded_files), batch_size):
    batch = uploaded_files[i:i+batch_size]
    # Process batch together
```

## API Integration

To use the model via API instead of Streamlit:

```python
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("models/trained/best.pt")

# Run inference
image = Image.open("document.jpg")
results = model(image, conf=0.25, iou=0.45)

# Get detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {cls}, Confidence: {conf:.2f}, Box: {xyxy}")
```

## Development

### Hot Reload

Streamlit automatically reloads when files change:

```bash
# Make changes to streamlit_application.py
# Browser will show "Source file changed. Rerunning..."
```

### Debugging

Add debug information:

```python
import streamlit as st

st.write("Debug info:")
st.write(f"Model path: {os.getenv('MODEL_PATH')}")
st.write(f"Image shape: {image.size}")
st.write(f"Detections: {len(result.boxes)}")
```

### Testing

```bash
# Unit tests
pytest tests/test_streamlit.py

# Manual testing
streamlit run src/web/streamlit_application.py --server.port 8502
```

## Deployment

### Production Checklist

- [ ] Set strong database password
- [ ] Use environment variables for sensitive config
- [ ] Enable HTTPS (use reverse proxy like nginx)
- [ ] Set up monitoring/logging
- [ ] Configure resource limits in Docker
- [ ] Test with production-like data volume
- [ ] Document deployment steps

### Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Support

For issues:
1. Check Docker logs: `docker compose -f src/environments/docker/compose.yml logs app`
2. Verify model path and existence
3. Test model with: `yolo predict model=models/trained/best.pt source=test.jpg`
4. Review Streamlit documentation: https://docs.streamlit.io
