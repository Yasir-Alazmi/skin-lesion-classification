"""
gpu_data.py
===========
Batch-level data pipeline that replaces the multi-process ``DataLoader`` when the
224x224 uint8 image cache (``src/cache.py``) is available.

Why: the cache already holds hair-removed, resized images, so the remaining per-image
work (flips, rotation, colour jitter, normalisation) is cheap tensor maths.  Doing it
per batch on the training device removes the worker processes (slow to spawn on
Windows, and the bottleneck of the original ~225 s epochs) and starves nothing.

Augmentation semantics vs. the old torchvision/PIL pipeline (train split only)
------------------------------------------------------------------------------
* Horizontal / vertical flip, p=0.5 each, per image: identical.
* Rotation uniform in [-30, 30] degrees, nearest-neighbour, zero fill, no expand,
  about the image centre, per image: identical (grid_sample instead of PIL).
* Brightness / contrast / saturation factors in [0.8, 1.2] and hue shift in
  [-0.1, 0.1], drawn per image with torchvision's formulas.  Difference: the
  application order is randomly permuted per *batch* instead of per image, and
  maths is done in float (no intermediate uint8 rounding).
* ToTensor + Normalize(ImageNet): identical.  Val / test: only that last step.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.config import IMAGENET_MEAN, IMAGENET_STD

_GRAY = (0.299, 0.587, 0.114)


def _gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return _GRAY[0] * r + _GRAY[1] * g + _GRAY[2] * b


def _uniform(n: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Batched RGB -> HSV (same maths as torchvision's tensor implementation)."""
    r, g, b = img.unbind(dim=1)
    maxc = torch.max(img, dim=1).values
    minc = torch.min(img, dim=1).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_div = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_div
    gc = (maxc - g) / cr_div
    bc = (maxc - b) / cr_div
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), dim=1)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(torch.int32)
    p = torch.clamp(v * (1.0 - s), 0.0, 1.0)
    q = torch.clamp(v * (1.0 - s * f), 0.0, 1.0)
    t = torch.clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(1) == torch.arange(6, device=i.device).view(1, -1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=1)
    a2 = torch.stack((t, v, v, q, p, p), dim=1)
    a3 = torch.stack((p, p, t, v, v, q), dim=1)
    a4 = torch.stack((a1, a2, a3), dim=1)
    return torch.einsum("...ijk,...xijk->...xjk", mask.to(img.dtype), a4)


def to_float(batch_uint8: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float in [0, 1] (what ToTensor does)."""
    # contiguous NCHW: channels_last input makes EfficientNet ~10x slower on this stack
    return batch_uint8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor, max_degrees: float = 30.0) -> torch.Tensor:
    """Random flips, rotation and colour jitter on a (B,3,H,W) float batch in [0, 1]."""
    n, dev = x.shape[0], x.device
    hflip = (torch.rand(n, device=dev) < 0.5).view(n, 1, 1, 1)
    x = torch.where(hflip, x.flip(3), x)
    vflip = (torch.rand(n, device=dev) < 0.5).view(n, 1, 1, 1)
    x = torch.where(vflip, x.flip(2), x)

    ang = _uniform(n, -max_degrees, max_degrees, dev) * (math.pi / 180.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    zero = torch.zeros_like(cos)
    theta = torch.stack((cos, -sin, zero, sin, cos, zero), dim=1).view(n, 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    for op in torch.randperm(4).tolist():
        if op == 0:  # brightness
            x = (x * _uniform(n, 0.8, 1.2, dev).view(n, 1, 1, 1)).clamp_(0, 1)
        elif op == 1:  # contrast
            f = _uniform(n, 0.8, 1.2, dev).view(n, 1, 1, 1)
            mean = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
            x = (f * x + (1 - f) * mean).clamp_(0, 1)
        elif op == 2:  # saturation
            f = _uniform(n, 0.8, 1.2, dev).view(n, 1, 1, 1)
            x = (f * x + (1 - f) * _gray(x)).clamp_(0, 1)
        else:  # hue
            shift = _uniform(n, -0.1, 0.1, dev).view(n, 1, 1)
            hsv = _rgb_to_hsv(x)
            h = torch.remainder(hsv[:, 0] + shift, 1.0)
            x = _hsv_to_rgb(torch.stack((h, hsv[:, 1], hsv[:, 2]), dim=1))
    return x.contiguous()  # einsum/stack leave NHWC strides, which slow the convnets


class _FrameView:
    """Minimal ``dataset`` stand-in: exposes ``df`` and ``len`` like HAMDataset."""

    def __init__(self, df, labels: np.ndarray) -> None:
        self.df = df.reset_index(drop=True)
        self._labels = labels

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> list[int]:
        return self._labels.tolist()


class GPUBatchLoader:
    """
    Iterable of ``(images, labels)`` batches already on ``device`` (normalised float).

    Parameters
    ----------
    images : (N, H, W, 3) uint8 array shared between loaders (kept in host RAM).
    rows : index into ``images`` for each sample of this split.
    labels : int labels per sample.
    sampler : optional iterable of sample positions per epoch (e.g. WeightedRandomSampler).
    """

    def __init__(
        self,
        images: np.ndarray,
        rows: Sequence[int],
        labels: Sequence[int],
        df,
        batch_size: int,
        device: torch.device,
        train: bool = False,
        sampler=None,
        drop_last: bool = False,
    ) -> None:
        self.images = images
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.dataset = _FrameView(df, np.asarray(labels))
        self.batch_size = batch_size
        self.device = device
        self.train = train
        self.sampler = sampler
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.rows)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _order(self) -> np.ndarray:
        if self.sampler is not None:
            return np.fromiter(iter(self.sampler), dtype=np.int64)
        if self.train:
            return torch.randperm(len(self.rows)).numpy()
        return np.arange(len(self.rows))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        order = self._order()
        for k in range(len(self)):
            idx = order[k * self.batch_size : (k + 1) * self.batch_size]
            batch = torch.from_numpy(self.images[self.rows[idx]])
            x = to_float(batch.to(self.device, non_blocking=True))
            if self.train:
                x = augment(x)
            yield normalize(x), self.labels[idx].to(self.device, non_blocking=True)
