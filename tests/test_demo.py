"""Tests for zero-friction inference CLI and demo functions."""
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from demo import run_inference


def test_demo_inference_samples():
    sample_path = Path(__file__).parent.parent / "data" / "samples" / "sample_melanocytic_nevus.jpg"
    assert sample_path.exists(), "Sample image must exist for zero-friction testing"
    
    # Test both architectures on CPU
    run_inference(str(sample_path), model_name="vit")
    run_inference(str(sample_path), model_name="efficientnet")
