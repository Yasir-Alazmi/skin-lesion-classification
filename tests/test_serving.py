"""Tests for the torch-free ONNX service. They use tiny stand-in models, so no trained weights are needed."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from serving.app import MAX_BYTES, create_app  # noqa: E402
from serving.model import combine_log_probs, load_onnx_ensemble, onnx_name  # noqa: E402
from serving.preprocess import CLASS_NAMES, preprocess, tta_views  # noqa: E402
from src import ensemble as torch_ens  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "sample_mel.jpg"


class Tiny(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 8, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, len(CLASS_NAMES))

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ens")
    members = [{"file": "a.pt", "size": 32}, {"file": "b.pt", "size": 48}]
    cfg = {"timm_model": "tiny", "members": members, "class_order": list(CLASS_NAMES),
           "bias": [0.1, -0.2, 0.0, 0.3, -0.1, 0.2, -0.3], "description": "tiny test ensemble"}
    (d / "ensemble_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for i, m in enumerate(members):
        net = Tiny(seed=i).eval()
        torch.save(net.state_dict(), d / m["file"])
        torch.onnx.export(net, torch.randn(1, 3, m["size"], m["size"]), str(d / onnx_name(m["file"])),
                          opset_version=17, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    return d


def _sample_rgb() -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(SAMPLE)), cv2.COLOR_BGR2RGB)


def test_numpy_preprocess_matches_torch_preprocess():
    img = _sample_rgb()
    for size in (64, 384):
        np.testing.assert_allclose(preprocess(img, size), torch_ens.preprocess(img, size).numpy(),
                                   atol=1e-5)


def test_tta_views_match_torch_flips():
    x = np.random.default_rng(0).standard_normal((1, 3, 5, 7)).astype(np.float32)
    t = torch.from_numpy(x)
    expected = torch.cat([t, t.flip(3), t.flip(2), t.flip(2).flip(3)]).numpy()
    np.testing.assert_array_equal(tta_views(x), expected)


def test_combine_rule_matches_torch_module():
    rng = np.random.default_rng(1)
    probs = [rng.dirichlet(np.ones(7)) for _ in range(3)]
    bias = rng.standard_normal(7)
    np.testing.assert_allclose(combine_log_probs(probs, bias), torch_ens.combine_log_probs(probs, bias),
                               atol=1e-12)


def test_onnx_ensemble_matches_pytorch_ensemble(model_dir):
    img = _sample_rgb()
    onnx_probs = load_onnx_ensemble(model_dir).predict_probs(img)
    pt = torch_ens.load_ensemble(model_dir, builder=lambda name: Tiny(seed=99))
    np.testing.assert_allclose(onnx_probs, pt.predict_probs(img), atol=1e-5)
    assert onnx_probs.shape == (7,) and abs(onnx_probs.sum() - 1.0) < 1e-9


def test_missing_onnx_file_is_reported(tmp_path, model_dir):
    (tmp_path / "ensemble_config.json").write_text((model_dir / "ensemble_config.json").read_text())
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        load_onnx_ensemble(tmp_path)


def test_api_predict_info_and_health(model_dir):
    client = TestClient(create_app(model=load_onnx_ensemble(model_dir)))
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/info").json()
    assert info["members"] == 2 and info["input_sizes"] == [32, 48] and info["tta_views"] == 4
    r = client.post("/predict", files={"file": ("x.jpg", SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in CLASS_NAMES
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert "Not a medical device" in body["disclaimer"]


def test_api_rejects_non_image(model_dir):
    client = TestClient(create_app(model=load_onnx_ensemble(model_dir)))
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_api_rejects_oversize_upload(model_dir):
    client = TestClient(create_app(model=load_onnx_ensemble(model_dir)))
    r = client.post("/predict", files={"file": ("big.jpg", b"x" * (MAX_BYTES + 1), "image/jpeg")})
    assert r.status_code == 413


def test_api_accepts_upload_at_the_limit_boundary(model_dir):
    # Exactly MAX_BYTES is not rejected for size (it fails decoding instead, as it is not an image).
    client = TestClient(create_app(model=load_onnx_ensemble(model_dir)))
    r = client.post("/predict", files={"file": ("edge.jpg", b"x" * MAX_BYTES, "image/jpeg")})
    assert r.status_code == 400


def test_export_script_runs_on_installed_torch(tmp_path):
    """scripts/export_onnx.py must work on the pinned torch (2.1 has no ``dynamo`` argument)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_onnx", ROOT / "scripts" / "export_onnx.py")
    export_onnx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_onnx)

    import inspect

    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    assert export_onnx.export_kwargs() == ({"dynamo": False} if has_dynamo else {})

    import onnxruntime as ort

    net = Tiny(seed=3).eval()
    path = tmp_path / "tiny.onnx"
    export_onnx.export_model(net, 32, path)
    x = np.random.default_rng(0).standard_normal((2, 3, 32, 32)).astype(np.float32)
    out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    with torch.no_grad():
        np.testing.assert_allclose(out, net(torch.from_numpy(x)).numpy(), atol=1e-5)


def test_health_reports_missing_model(tmp_path):
    client = TestClient(create_app(model_dir=tmp_path))
    assert client.get("/health").status_code == 503
