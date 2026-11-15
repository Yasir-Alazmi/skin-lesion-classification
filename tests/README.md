# Tests

This directory contains the test suite for the skin lesion classification project.

---

## Running Tests

```bash
# Install dev dependencies first
pip install pytest

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_smoke.py -v
```

---

## Test Structure

| File | Description |
|------|-------------|
| `test_smoke.py` | Imports, FocalLoss forward pass, model output shapes |
| `test_split.py`, `test_gpu_data.py`, `test_train_loop.py` | Lesion-grouped split, GPU data path, training loop |
| `test_demo.py` | Inference path (preprocess, top-3, weight loading) with a tiny stand-in model |
| `test_core.py`, `test_report.py` | Preprocessing, early stopping, metric/bootstrap helpers |

---

## CI

Tests run automatically on every push and pull request via [GitHub Actions](../.github/workflows/ci.yml).
No GPU or dataset is required; all tests use CPU and randomly generated tensors.
