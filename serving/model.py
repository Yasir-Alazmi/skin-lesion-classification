"""
ONNX Runtime version of the ConvNeXt-S ensemble (``src.ensemble.EnsembleClassifier``).

Per member: preprocess at the member's size, run the 4 flip views as one batch, average the softmax
probabilities. Across members: mean log-probability plus the per-class bias, then softmax.

The served model is the 4-member ensemble in ``ensemble_config.json``. Its test-set score was never
measured (the test set is not re-used); the reported 0.819 balanced accuracy belongs to a 5-member
ensemble. Research code only: not a medical device.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort

from serving.preprocess import CLASS_NAMES, preprocess, tta_views

_EPS = 1e-6


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def combine_log_probs(member_probs: Sequence[np.ndarray], bias: np.ndarray) -> np.ndarray:
    """Same rule as ``src.ensemble.combine_log_probs``: mean log-prob + bias, then softmax."""
    logs = np.mean([np.log(np.clip(p, _EPS, 1.0)) for p in member_probs], axis=0) + np.asarray(bias)
    return softmax(logs)


def onnx_name(pt_file: str) -> str:
    return Path(pt_file).with_suffix(".onnx").name


class OnnxEnsemble:
    def __init__(self, sessions: Sequence[tuple[ort.InferenceSession, int]], bias: Sequence[float],
                 description: str = ""):
        if not sessions:
            raise ValueError("ensemble needs at least one member")
        if len(bias) != len(CLASS_NAMES):
            raise ValueError(f"bias must have {len(CLASS_NAMES)} entries, got {len(bias)}")
        self.sessions = [(s, int(size)) for s, size in sessions]
        self.bias = np.asarray(bias, dtype=np.float64)
        self.description = description

    @property
    def sizes(self) -> list[int]:
        return [size for _, size in self.sessions]

    def member_probs(self, image_rgb: np.ndarray) -> list[np.ndarray]:
        cache: dict[int, np.ndarray] = {}
        out = []
        for session, size in self.sessions:
            if size not in cache:
                cache[size] = tta_views(preprocess(image_rgb, size))
            logits = session.run(None, {session.get_inputs()[0].name: cache[size]})[0]
            out.append(softmax(logits.astype(np.float64), axis=1).mean(axis=0))
        return out

    def predict_probs(self, image_rgb: np.ndarray) -> np.ndarray:
        return combine_log_probs(self.member_probs(image_rgb), self.bias)


def load_onnx_ensemble(directory: str | Path, threads: int | None = None) -> OnnxEnsemble:
    """Read ``ensemble_config.json`` and the ``.onnx`` file of each member next to it."""
    directory = Path(directory)
    cfg_path = directory / "ensemble_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"ensemble config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if list(cfg["class_order"]) != list(CLASS_NAMES):
        raise ValueError("class order in the config does not match the service label map")
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = int(threads)
    sessions = []
    for m in cfg["members"]:
        path = directory / onnx_name(m["file"])
        if not path.is_file():
            raise FileNotFoundError(f"ONNX member not found: {path} (run scripts/export_onnx.py)")
        sessions.append((ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"]),
                         m["size"]))
    return OnnxEnsemble(sessions, cfg["bias"], cfg.get("description", ""))
