"""Test GPU availability and configuration"""

import torch
from ultralytics import YOLO
import sys


def test_gpu():
    """Test GPU availability and performance"""
    print("=" * 70)
    print("GPU Configuration Test\n" + "=" * 70)

    # PyTorch GPU Check
    print("\n1. PyTorch GPU Status:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            print(
                f"     Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            )
            print(
                f"     Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            )

        # Test CUDA with a simple tensor operation
        print("\n2. CUDA Performance Test:")
        device = torch.device("cuda")

        # Create random tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # Time matrix multiplication
        import time

        start = time.time()
        for _ in range(100):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"   100x (1000x1000) matrix multiplications on GPU: {gpu_time:.4f}s")

        # Compare with CPU
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        for _ in range(100):
            z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start

        print(f"   100x (1000x1000) matrix multiplications on CPU: {cpu_time:.4f}s")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")

    else:
        print("\n   ⚠ CUDA is NOT available!")
        print("   Possible issues:")
        print("     1. NVIDIA GPU drivers not installed on host")
        print("     2. Docker not configured for GPU access")
        print("     3. NVIDIA Container Toolkit not installed")
        print("\n   To fix:")
        print("     - Install NVIDIA GPU drivers")
        print("     - Install NVIDIA Container Toolkit:")
        print(
            "       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        )
        print("     - Restart Docker and rebuild container")

    # Test YOLO with GPU
    print("\n3. YOLO GPU Test:")
    try:
        model = YOLO("yolov8n.pt")
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"   YOLO Model loaded")
        print(f"   Device: {device_name}")

        if torch.cuda.is_available():
            print(f"   ✓ YOLO will use GPU for inference")
        else:
            print(f"   ⚠ YOLO will use CPU (slower)")

    except Exception as e:
        print(f"   Error loading YOLO: {e}")

    print("\n" + "=" * 70)
    if torch.cuda.is_available():
        print("✓ GPU is properly configured and working!")
    else:
        print("⚠ GPU is NOT available - running on CPU only")
    print("=" * 70 + "\n")

    return torch.cuda.is_available()


if __name__ == "__main__":
    gpu_available = test_gpu()
    sys.exit(0 if gpu_available else 1)
