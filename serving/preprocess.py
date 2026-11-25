"""
Torch-free preprocessing for the ONNX service.

Mirrors ``src.ensemble.preprocess`` (hair removal -> resize -> ImageNet normalisation) using only
OpenCV and NumPy, so the serving image does not need PyTorch. ``tests/test_serving.py`` checks that
both give the same tensor.
"""

from __future__ import annotations

import cv2
import numpy as np

CLASS_NAMES: tuple[str, ...] = ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def remove_hair(image_bgr: np.ndarray, kernel_size: int = 17, threshold: int = 10,
                inpaint_radius: int = 6) -> np.ndarray:
    """Black-hat hair mask + Telea inpainting (same defaults as ``src.preprocess.remove_hair``)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image_bgr, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)


def preprocess(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 array (1, 3, size, size)."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image_rgb.shape}")
    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(remove_hair(bgr), (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD)[None].astype(np.float32)


def tta_views(x: np.ndarray) -> np.ndarray:
    """(1, 3, H, W) -> (4, 3, H, W): identity, horizontal flip, vertical flip, both."""
    return np.concatenate([x, x[..., ::-1], x[..., ::-1, :], x[..., ::-1, ::-1]], axis=0).copy()
