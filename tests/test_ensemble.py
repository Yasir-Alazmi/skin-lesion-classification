"""CPU-only tests for src/ensemble.py (no checkpoints or dataset needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES  # noqa: E402
from src.ensemble import EnsembleClassifier, combine_log_probs, load_ensemble, preprocess  # noqa: E402

N = len(CLASS_NAMES)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, N)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


def _image(h=40, w=60):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_combine_is_a_distribution_and_bias_shifts_argmax():
    p = np.full(N, 1.0 / N)
    out = combine_log_probs([p, p], np.zeros(N))
    assert out.shape == (N,) and abs(out.sum() - 1) < 1e-9
    bias = np.zeros(N)
    bias[3] = 2.0
    assert combine_log_probs([p], bias).argmax() == 3


def test_combine_handles_zero_probabilities():
    p = np.zeros(N)
    p[0] = 1.0
    out = combine_log_probs([p], np.zeros(N))
    assert np.isfinite(out).all() and out.argmax() == 0


@pytest.mark.parametrize("size", [384, 448])
def test_preprocess_shape(size):
    x = preprocess(_image(), size)
    assert x.shape == (1, 3, size, size) and torch.isfinite(x).all()


def test_preprocess_rejects_bad_input():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), dtype=np.uint8), 384)


def test_ensemble_predicts_distribution_with_mixed_sizes():
    ens = EnsembleClassifier([(Tiny(), 64), (Tiny(), 96)], np.zeros(N))
    p = ens.predict_probs(_image())
    assert p.shape == (N,) and abs(p.sum() - 1) < 1e-6


def test_ensemble_validates_arguments():
    with pytest.raises(ValueError):
        EnsembleClassifier([], np.zeros(N))
    with pytest.raises(ValueError):
        EnsembleClassifier([(Tiny(), 64)], np.zeros(N - 1))


def test_load_ensemble_roundtrip_and_errors(tmp_path):
    cfg = {"timm_model": "tiny", "members": [{"file": "a.pt", "size": 64}],
           "class_order": ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"], "bias": [0.0] * N}
    (tmp_path / "ensemble_config.json").write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError):  # weights missing
        load_ensemble(tmp_path, builder=lambda name: Tiny())
    torch.save(Tiny().state_dict(), tmp_path / "a.pt")
    ens = load_ensemble(tmp_path, builder=lambda name: Tiny())
    assert abs(ens.predict_probs(_image()).sum() - 1) < 1e-6
    cfg["class_order"] = ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"]
    (tmp_path / "ensemble_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_ensemble(tmp_path, builder=lambda name: Tiny())
    with pytest.raises(FileNotFoundError):
        load_ensemble(tmp_path / "missing", builder=lambda name: Tiny())


def test_shipped_config_is_consistent():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "models/ensemble/ensemble_config.json").read_text())
    assert len(cfg["bias"]) == N and len(cfg["members"]) == 4
    assert cfg["class_order"] == [c for c in ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")]
