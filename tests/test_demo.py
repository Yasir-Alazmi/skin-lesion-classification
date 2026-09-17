"""Tests for zero-friction inference CLI and demo functions."""
from pathlib import Path
from demo import run_inference


def test_demo_inference_samples():
    sample_path = Path(__file__).parent.parent / "data" / "samples" / "sample_melanocytic_nevus.jpg"
    assert sample_path.exists(), "Sample image must exist for zero-friction testing"
    
    # Test both architectures on CPU
    run_inference(str(sample_path), model_name="vit")
    run_inference(str(sample_path), model_name="efficientnet")
