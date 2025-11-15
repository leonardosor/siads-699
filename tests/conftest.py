import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new("RGB", (640, 480), color="white")
    return img


@pytest.fixture
def sample_image_path(temp_dir, sample_image):
    """Save a sample image to a temporary path."""
    img_path = temp_dir / "test_image.jpg"
    sample_image.save(img_path)
    return img_path


@pytest.fixture
def sample_yolo_label():
    """Sample YOLO format label (class x_center y_center width height)."""
    return "0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.15 0.2\n"


@pytest.fixture
def mock_model_weights(temp_dir):
    """Create a mock model weights file."""
    weights_path = temp_dir / "mock_model.pt"
    weights_path.touch()
    return weights_path
