"""
inference.py
============
Single-image inference with the trained ViT-B/16, using exactly the evaluation
preprocessing: hair removal (Black-Hat + TELEA) -> 224x224 resize -> ImageNet
normalisation.  Shared by ``app.py`` (Gradio demo) and ``demo.py`` (CLI).

Research code only: not a medical device and not clinically validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from src.config import CLASS_NAMES, IDX_TO_LABEL, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from src.models import build_vit
from src.preprocess import remove_hair

_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_model(
    weights_path: str | Path,
    builder: Callable[..., nn.Module] = build_vit,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build the architecture (no pretrained download) and load trained weights.

    Accepts either a training checkpoint (``{"model_state_dict": ...}``) or a bare
    state dict.  Returns the model in eval mode.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    model = builder(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB array (H, W, 3) -> normalised tensor (1, 3, 224, 224)."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image_rgb.shape}")
    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    clean = remove_hair(bgr)
    pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    return _TRANSFORM(pil).unsqueeze(0)


@torch.no_grad()
def predict_probs(model: nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """Softmax probabilities over the 7 classes, in ``config.CLASS_NAMES`` order."""
    device = next(model.parameters()).device
    logits = model(preprocess(image_rgb).to(device))
    return torch.softmax(logits.float(), dim=1)[0].cpu().numpy()


def top_k(probs: np.ndarray, k: int = 3) -> list[dict]:
    """The ``k`` most likely classes as ``{"code", "name", "prob"}``, best first."""
    order = np.argsort(probs)[::-1][:k]
    return [
        {"code": IDX_TO_LABEL[int(i)], "name": CLASS_NAMES[int(i)], "prob": float(probs[i])}
        for i in order
    ]
