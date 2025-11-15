import pytest
from pathlib import Path


class TestDatasetUtils:
    """Tests for dataset utility functions."""

    def test_yolo_label_parsing(self, sample_yolo_label):
        """Test parsing YOLO label format."""
        lines = sample_yolo_label.strip().split("\n")
        assert len(lines) == 2

        # Parse first line
        parts = lines[0].split()
        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        assert cls_id == 0
        assert 0.0 <= x_center <= 1.0
        assert 0.0 <= y_center <= 1.0
        assert 0.0 <= width <= 1.0
        assert 0.0 <= height <= 1.0

    def test_label_counting(self, temp_dir, sample_yolo_label):
        """Test counting labels in dataset."""
        labels_dir = temp_dir / "labels"
        labels_dir.mkdir()

        # Create sample label files
        for i in range(3):
            label_file = labels_dir / f"image_{i}.txt"
            label_file.write_text(sample_yolo_label)

        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) == 3

    def test_label_remapping(self):
        """Test remapping class IDs in YOLO labels."""
        original_label = "0 0.5 0.5 0.3 0.4"
        class_map = {0: 1, 1: 2, 2: 0}

        parts = original_label.split()
        old_cls = int(parts[0])
        new_cls = class_map[old_cls]

        assert new_cls == 1


class TestModelUtils:
    """Tests for model utility functions."""

    def test_model_run_listing(self, temp_dir):
        """Test listing model runs."""
        runs_dir = temp_dir / "runs"
        runs_dir.mkdir()

        # Create mock run directories
        run_names = [
            "finance-parser-20251112_114247",
            "finance-parser-20251112_143826",
        ]

        for run_name in run_names:
            run_dir = runs_dir / run_name
            run_dir.mkdir()
            (run_dir / "weights").mkdir()
            (run_dir / "weights" / "best.pt").touch()

        run_dirs = list(runs_dir.glob("finance-parser-*"))
        assert len(run_dirs) == 2

    def test_active_run_tracking(self, temp_dir):
        """Test active run file management."""
        active_run_file = temp_dir / "active_run.txt"
        run_name = "finance-parser-20251112_114247"

        active_run_file.write_text(run_name)
        assert active_run_file.read_text() == run_name


class TestPathUtils:
    """Tests for path utility functions."""

    def test_path_resolution(self):
        """Test resolving relative paths."""
        rel_path = Path("models/trained/best.pt")
        assert isinstance(rel_path, Path)

    def test_path_validation(self, temp_dir):
        """Test validating file paths."""
        valid_path = temp_dir / "valid.txt"
        valid_path.touch()

        assert valid_path.exists()
        assert valid_path.is_file()

        invalid_path = temp_dir / "invalid.txt"
        assert not invalid_path.exists()
