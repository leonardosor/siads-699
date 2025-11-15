# Tests

Test suite for the SIADS 699 Capstone project.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_ocr_processor.py    # OCR processing tests
│   ├── test_training.py          # Training configuration tests
│   └── test_utils.py             # Utility function tests
├── integration/             # Integration tests for full workflows
│   └── test_streamlit_app.py     # Streamlit application tests
├── fixtures/                # Test fixtures and sample data
│   └── sample_images/            # Sample images for testing
└── conftest.py              # Pytest configuration and fixtures
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Suite

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_ocr_processor.py

# Specific test class
pytest tests/unit/test_ocr_processor.py::TestOCRProcessor

# Specific test method
pytest tests/unit/test_ocr_processor.py::TestOCRProcessor::test_confidence_thresholds
```

### Run with Coverage

```bash
pytest --cov=src tests/
```

### Run with Verbose Output

```bash
pytest -v tests/
```

### Run and Stop on First Failure

```bash
pytest -x tests/
```

## Test Configuration

### Pytest Configuration

Create `pytest.ini` in the project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### Required Test Dependencies

Install test dependencies from [src/environments/requirements-dev.txt](../src/environments/requirements-dev.txt):

```bash
pip install -r src/environments/requirements-dev.txt
```

Key testing packages:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking support
- `pytest-xdist` - Parallel test execution

## Writing Tests

### Test Structure

Follow this pattern for new tests:

```python
import pytest
from pathlib import Path

class TestMyComponent:
    """Tests for MyComponent functionality."""

    def test_basic_functionality(self):
        """Test basic functionality works."""
        result = my_function()
        assert result is not None

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_with_parameters(self, input, expected):
        """Test with multiple parameter combinations."""
        assert my_function(input) == expected

    def test_with_fixture(self, sample_image):
        """Test using a fixture."""
        assert sample_image.size == (640, 480)
```

### Using Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `sample_image` - PIL Image object
- `sample_image_path` - Path to saved sample image
- `sample_yolo_label` - YOLO format label string
- `mock_model_weights` - Mock model weights file

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

@patch('ultralytics.YOLO')
def test_with_mock(mock_yolo):
    """Test using mocked YOLO model."""
    mock_model = Mock()
    mock_yolo.return_value = mock_model

    # Your test code here
    model = mock_yolo("path/to/weights.pt")
    assert model is not None
```

### Parameterized Tests

```python
@pytest.mark.parametrize("confidence,iou", [
    (0.25, 0.45),
    (0.5, 0.5),
    (0.75, 0.6),
])
def test_detection_thresholds(confidence, iou):
    """Test with multiple threshold combinations."""
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= iou <= 1.0
```

## Test Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing tests/

# HTML report
pytest --cov=src --cov-report=html tests/
# Opens htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=src --cov-report=xml tests/
```

### Coverage Goals

- **Minimum**: 70% overall coverage
- **Target**: 80%+ coverage for core modules
- **Critical paths**: 90%+ coverage (OCR pipeline, model inference)

## Continuous Integration

### GitHub Actions

Example workflow (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r src/environments/requirements-dev.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml tests/
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Common Testing Patterns

### Testing Image Processing

```python
def test_image_processing(sample_image):
    """Test image processing pipeline."""
    # Process image
    processed = preprocess_image(sample_image)

    # Verify output
    assert processed.size == (640, 640)
    assert processed.mode == 'RGB'
```

### Testing File Operations

```python
def test_file_operations(temp_dir):
    """Test file read/write operations."""
    test_file = temp_dir / "test.txt"

    # Write
    test_file.write_text("test content")

    # Read
    content = test_file.read_text()
    assert content == "test content"
```

### Testing with Environment Variables

```python
@patch.dict('os.environ', {'MODEL_PATH': '/mock/path/model.pt'})
def test_with_env_var():
    """Test using environment variables."""
    import os
    model_path = os.getenv('MODEL_PATH')
    assert model_path == '/mock/path/model.pt'
```

## Troubleshooting

### Import Errors

If tests can't import source modules:

```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
pytest tests/
```

Or in the test file:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### Fixture Not Found

Ensure `conftest.py` is in the right location:
- Root `tests/conftest.py` - Available to all tests
- `tests/unit/conftest.py` - Available only to unit tests

### Tests Running Slow

Use parallel execution:

```bash
pytest -n auto tests/
```

Skip slow tests:

```bash
pytest -m "not slow" tests/
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **One Assertion**: Prefer one logical assertion per test (exceptions allowed)
3. **Arrange-Act-Assert**: Structure tests clearly
   - Arrange: Set up test data
   - Act: Execute the code under test
   - Assert: Verify the results
4. **Isolated Tests**: Each test should be independent
5. **Mock External Services**: Don't rely on external APIs, databases, etc.
6. **Fast Tests**: Keep unit tests fast (<100ms each)
7. **Document TODO**: Use `# TODO:` comments for incomplete tests
8. **Use Fixtures**: Reuse common setup code via fixtures
9. **Parameterize**: Test multiple inputs with `@pytest.mark.parametrize`
10. **Clean Up**: Use `temp_dir` fixture for file operations

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Real Python Pytest Guide](https://realpython.com/pytest-python-testing/)
