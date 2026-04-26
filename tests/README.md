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
| `test_smoke.py` | Smoke tests — imports, FocalLoss forward pass, model output shapes |

---

## CI

Tests run automatically on every push and pull request via [GitHub Actions](../.github/workflows/ci.yml).
No GPU or dataset is required; all tests use CPU and randomly generated tensors.
