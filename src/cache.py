"""
cache.py
========
One-off image cache: hair removal + resize to ``IMG_SIZE`` stored as a uint8
memory-mapped array.

Hair removal (TELEA inpainting) costs ~0.2 s per full-size image, which would
dominate every epoch.  The pipeline order is  read -> hair removal -> Resize ->
augmentation, so caching the result of the first three steps is exactly
equivalent to doing them on the fly, and leaves the random augmentations intact.
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

from src.config import IMG_SIZE
from src.preprocess import read_image_bgr, remove_hair

_RESIZE = T.Resize((IMG_SIZE, IMG_SIZE))


def cache_files(cache_dir: str | Path, hair_removal: bool) -> tuple[Path, Path]:
    tag = f"ham{IMG_SIZE}_hair{int(hair_removal)}"
    d = Path(cache_dir)
    return d / f"{tag}.npy", d / f"{tag}_ids.json"


def load_and_prepare(path: str | Path, hair_removal: bool) -> np.ndarray:
    """Read one image, optionally remove hair, resize; returns (H, W, 3) uint8 RGB."""
    img_bgr = read_image_bgr(path)
    if hair_removal:
        img_bgr = remove_hair(img_bgr)
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return np.asarray(_RESIZE(pil), dtype=np.uint8)


def _worker(args: tuple[str, bool]) -> np.ndarray:
    return load_and_prepare(*args)


def _locate(image_id: str, image_dirs: Sequence[Path]) -> Path:
    for d in image_dirs:
        p = Path(d) / f"{image_id}.jpg"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image '{image_id}.jpg' not found in: {list(image_dirs)}")


def build_cache(
    image_ids: Sequence[str],
    image_dirs: Sequence[str | Path],
    cache_dir: str | Path,
    hair_removal: bool = True,
    workers: int = 8,
    overwrite: bool = False,
) -> Path:
    """Create (or reuse) the cache for ``image_ids``; returns the .npy path."""
    npy, ids_json = cache_files(cache_dir, hair_removal)
    ids = sorted(image_ids)
    if npy.exists() and ids_json.exists() and not overwrite:
        if json.loads(ids_json.read_text()) == ids:
            return npy
    npy.parent.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in image_dirs]
    paths = [str(_locate(i, dirs)) for i in ids]
    arr = np.lib.format.open_memmap(
        str(npy), mode="w+", dtype=np.uint8, shape=(len(ids), IMG_SIZE, IMG_SIZE, 3)
    )
    jobs = [(p, hair_removal) for p in paths]
    with Pool(workers) as pool:
        for k, out in enumerate(pool.imap(_worker, jobs, chunksize=16)):
            arr[k] = out
            if (k + 1) % 1000 == 0:
                print(f"  cached {k + 1}/{len(ids)}", flush=True)
    arr.flush()
    del arr
    ids_json.write_text(json.dumps(ids))
    return npy


class ImageCache:
    """Read-only view of a cache built by :func:`build_cache`."""

    def __init__(self, cache_dir: str | Path, hair_removal: bool = True) -> None:
        npy, ids_json = cache_files(cache_dir, hair_removal)
        self._npy = npy
        self._ids = {i: k for k, i in enumerate(json.loads(ids_json.read_text()))}
        self._arr: np.ndarray | None = None  # opened lazily (per DataLoader worker)

    def load_all(self) -> np.ndarray:
        """Whole cache as one in-RAM (N, H, W, 3) uint8 array (about 1.5 GB for HAM10000)."""
        return np.load(self._npy)

    def row(self, image_id: str) -> int:
        return self._ids[image_id]

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._ids

    def get(self, image_id: str) -> Image.Image:
        if self._arr is None:
            self._arr = np.load(self._npy, mmap_mode="r")
        return Image.fromarray(np.array(self._arr[self._ids[image_id]]))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arr"] = None  # never pickle the memmap into workers
        return state

    @staticmethod
    def exists(cache_dir: str | Path, hair_removal: bool = True) -> bool:
        npy, ids_json = cache_files(cache_dir, hair_removal)
        return os.path.exists(npy) and os.path.exists(ids_json)
