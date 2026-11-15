"""
ensemble.py
===========
Inference with the ConvNeXt-S ensemble (v2.1.0). Per member: hair removal -> resize to the member's
size -> ImageNet normalisation -> 4-view flip TTA (probabilities averaged). Across members: mean of
log-probabilities, plus a per-class log bias (tuned on the validation set), then softmax.

This mirrors ``scripts/final_eval_r2.py``. Research code only: not a medical device and not
clinically validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src.preprocess import remove_hair

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models" / "ensemble"
_EPS = 1e-6


class ConvNeXtClassifier(nn.Module):
    """Same layout as the Kaggle training script: backbone, dropout, linear head."""

    def __init__(
        self, timm_model: str, num_classes: int = len(CLASS_NAMES), pretrained: bool = False
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(timm_model, pretrained=pretrained, num_classes=0)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.backbone(x)))


def combine_log_probs(member_probs: Sequence[np.ndarray], bias: np.ndarray) -> np.ndarray:
    """Mean log-probability over members plus class bias, then softmax. Inputs are (C,) arrays."""
    logs = np.mean([np.log(np.clip(p, _EPS, 1.0)) for p in member_probs], axis=0) + np.asarray(bias)
    e = np.exp(logs - logs.max())
    return e / e.sum()


def preprocess(image_rgb: np.ndarray, size: int) -> torch.Tensor:
    """uint8 RGB (H, W, 3) -> normalised tensor (1, 3, size, size)."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image_rgb.shape}")
    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(remove_hair(bgr), (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


class EnsembleClassifier:
    """``members`` is a list of ``(model, size)``; ``bias`` has one entry per class."""

    def __init__(self, members: Sequence[tuple[nn.Module, int]], bias: Sequence[float]):
        if not members:
            raise ValueError("ensemble needs at least one member")
        if len(bias) != len(CLASS_NAMES):
            raise ValueError(f"bias must have {len(CLASS_NAMES)} entries, got {len(bias)}")
        self.members = [(m.eval(), int(s)) for m, s in members]
        self.bias = np.asarray(bias, dtype=np.float64)

    @torch.no_grad()
    def _member_probs(self, model: nn.Module, x: torch.Tensor) -> np.ndarray:
        device = next(model.parameters()).device
        x = x.to(device)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
        probs = torch.stack([torch.softmax(model(v).float(), dim=1) for v in views]).mean(0)
        return probs[0].cpu().numpy()

    def predict_probs(self, image_rgb: np.ndarray) -> np.ndarray:
        """Final probabilities over the 7 classes, in ``config.CLASS_NAMES`` order."""
        cache: dict[int, torch.Tensor] = {}
        per_member = []
        for model, size in self.members:
            if size not in cache:
                cache[size] = preprocess(image_rgb, size)
            per_member.append(self._member_probs(model, cache[size]))
        return combine_log_probs(per_member, self.bias)


def load_ensemble(
    directory: str | Path | None = None,
    device: str | torch.device = "cpu",
    builder: Callable[[str], nn.Module] | None = None,
) -> EnsembleClassifier:
    """Load ``ensemble_config.json`` and the member weights found next to it."""
    directory = Path(directory) if directory else DEFAULT_DIR
    cfg_path = directory / "ensemble_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"ensemble config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["class_order"] != [c for c in ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")]:
        raise ValueError("class order in the config does not match the project label map")
    build = builder or (lambda name: ConvNeXtClassifier(name, pretrained=False))
    members = []
    for m in cfg["members"]:
        path = directory / m["file"]
        if not path.is_file():
            raise FileNotFoundError(f"ensemble weights not found: {path}")
        model = build(cfg["timm_model"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        members.append((model.to(device), m["size"]))
    return EnsembleClassifier(members, cfg["bias"])
