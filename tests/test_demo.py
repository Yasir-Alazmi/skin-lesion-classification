"""Inference-path tests on CPU with a tiny stand-in model: no checkpoint or dataset needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import CLASS_NAMES  # noqa: E402
from src.inference import load_model, predict_probs, preprocess, top_k  # noqa: E402

SAMPLE = repo_root / "data" / "samples" / "sample_mel.jpg"


class TinyNet(nn.Module):
    """Stand-in with the same input/output contract as the ViT classifier."""

    def __init__(self, num_classes: int = 7, pretrained: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(nn.functional.adaptive_avg_pool2d(x, 4).flatten(1))


def _image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (300, 400, 3), dtype=np.uint8)


def test_preprocess_shape_and_normalisation():
    t = preprocess(_image())
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == torch.float32
    assert -3.0 < float(t.min()) and float(t.max()) < 3.0  # ImageNet-normalised range


def test_preprocess_rejects_bad_input():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), dtype=np.uint8))


def test_predict_probs_is_a_distribution():
    torch.manual_seed(0)
    probs = predict_probs(TinyNet().eval(), _image())
    assert probs.shape == (len(CLASS_NAMES),)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_top_k_sorted_and_labelled():
    probs = np.array([0.05, 0.5, 0.1, 0.2, 0.05, 0.05, 0.05])
    res = top_k(probs, k=3)
    assert [r["code"] for r in res] == ["mel", "bcc", "bkl"]
    assert res[0]["name"] == "Melanoma"
    assert res[0]["prob"] >= res[1]["prob"] >= res[2]["prob"]


def test_load_model_roundtrip_both_formats(tmp_path):
    torch.manual_seed(1)
    ref = TinyNet().eval()
    expected = predict_probs(ref, _image(2))

    for name, payload in [
        ("ckpt.pth", {"model_state_dict": ref.state_dict(), "epoch": 3}),
        ("bare.pth", ref.state_dict()),
    ]:
        path = tmp_path / name
        torch.save(payload, path)
        model = load_model(path, builder=TinyNet)
        assert not model.training
        assert np.allclose(predict_probs(model, _image(2)), expected, atol=1e-6)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pth", builder=TinyNet)


def test_demo_cli_function_on_sample(tmp_path):
    from demo import run_inference

    torch.manual_seed(2)
    weights = tmp_path / "w.pth"
    torch.save(TinyNet().state_dict(), weights)
    res = run_inference(str(SAMPLE), str(weights), builder=TinyNet)
    assert len(res) == 3
    assert res[0]["prob"] >= res[1]["prob"]
