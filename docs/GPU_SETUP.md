# GPU Setup Guide

## Overview
Your Docker container is now configured to use NVIDIA GPUs for accelerated OCR and ML training with Ultralytics YOLO and PyTorch.

## Prerequisites

### 1. NVIDIA GPU Drivers
Ensure NVIDIA GPU drivers are installed on your **host machine** (not in Docker):

**Windows:**
- Download from: https://www.nvidia.com/Download/index.aspx
- Install latest Game Ready or Studio drivers

**Linux:**
```bash
# Check if drivers installed
nvidia-smi

# Install if needed (Ubuntu/Debian)
sudo apt-get install nvidia-driver-535
```

### 2. NVIDIA Container Toolkit

**Windows with Docker Desktop:**
- GPU support is built-in with Docker Desktop 4.19+
- Ensure WSL2 backend is enabled
- GPU drivers must be installed on Windows host

**Linux:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

## Docker Configuration

Your `docker-compose.yml` has been updated with GPU support:

```yaml
devcontainer:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

## Verify GPU Access

### 1. Rebuild and Restart Container
```bash
# Stop existing containers
docker-compose down

# Rebuild with GPU support
docker-compose build --no-cache

# Start containers
docker-compose up -d

# Enter container
docker-compose exec devcontainer bash
```

### 2. Check GPU Availability
Inside the container:
```bash
# Check NVIDIA drivers visible
nvidia-smi

# Test PyTorch GPU
python scripts/test_gpu.py
```

Expected output:
```
GPU Configuration Test
======================================================================

1. PyTorch GPU Status:
   CUDA Available: True
   CUDA Version: 12.1
   GPU Count: 1

   GPU 0:
     Name: NVIDIA GeForce RTX 3080
     Memory: 10.00 GB
     Compute Capability: 8.6

2. CUDA Performance Test:
   100x (1000x1000) matrix multiplications on GPU: 0.1234s
   100x (1000x1000) matrix multiplications on CPU: 5.6789s
   Speedup: 46.00x

3. YOLO GPU Test:
   YOLO Model loaded
   Device: cuda:0
   ✓ YOLO will use GPU for inference

======================================================================
✓ GPU is properly configured and working!
======================================================================
```

## Using GPU in Scripts

### OCR Processing with GPU
```python
from scripts.ocr_processor import OCRProcessor

# YOLO will automatically use GPU if available
processor = OCRProcessor(
    use_yolo_ocr=True,
    use_tesseract=True,
    save_to_db=True
)

results = processor.process_all_parquets(sample_size=100)
```

### YOLO Training with GPU
```python
from scripts.train_yolo_ocr import YOLODocumentTrainer

trainer = YOLODocumentTrainer()
trainer.extract_images_from_parquet()

# Explicitly use GPU
trainer.train_classification_model(
    model_size='s',
    epochs=50,
    batch=32,
    device='0'  # Use GPU 0 (or 'cpu' for CPU only)
)
```

### Manual PyTorch GPU Usage
```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Move tensors/models to GPU
model = model.to(device)
input_tensor = input_tensor.to(device)
```

## Troubleshooting

### Issue: GPU not detected in container

**Check host GPU:**
```bash
# On host machine (not in container)
nvidia-smi
```

If this fails, install/update NVIDIA drivers on host.

**Check Docker GPU access:**
```bash
# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this fails:
- Windows: Update Docker Desktop to 4.19+, enable WSL2
- Linux: Install NVIDIA Container Toolkit

**Check container configuration:**
```bash
# Inspect container
docker inspect pdf-ocr-devcontainer | grep -i gpu
```

### Issue: CUDA out of memory

Reduce batch size in training:
```python
trainer.train_classification_model(
    batch=8,  # Reduce from 16/32
    imgsz=224  # Or reduce image size
)
```

### Issue: CUDA version mismatch

The Ultralytics base image includes PyTorch with CUDA 12.1. Ensure your host has compatible drivers:
- CUDA 12.1 requires NVIDIA driver >= 530

Update host drivers if needed.

## Performance Tips

1. **Batch Size**: Larger batches = better GPU utilization, but watch memory
2. **Image Size**: Start with 224x224, increase to 640x640 for better accuracy
3. **Mixed Precision**: Enable for faster training (Ultralytics does this automatically)
4. **Data Loading**: Use multiple workers for data loading to avoid GPU starvation

## Monitoring GPU Usage

### Inside Container:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

### Outside Container:
```bash
# Monitor GPU from host
nvidia-smi dmon -s u
```

### Python Monitoring:
```python
import torch

print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## Verification Checklist

- [ ] NVIDIA drivers installed on host (`nvidia-smi` works on host)
- [ ] Docker Desktop 4.19+ (Windows) or NVIDIA Container Toolkit (Linux)
- [ ] Container rebuilt after docker-compose.yml changes
- [ ] `python scripts/test_gpu.py` shows GPU available
- [ ] YOLO training uses GPU (device='0')
- [ ] Performance improvement vs CPU

## References

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker GPU Support](https://docs.docker.com/compose/gpu-support/)
- [PyTorch CUDA Guide](https://pytorch.org/docs/stable/cuda.html)
- [Ultralytics GPU Docs](https://docs.ultralytics.com/guides/nvidia-jetson/)
