"""
dataset.py
==========
PyTorch Dataset and data-splitting utilities for the HAM10000 pipeline.

Classes
-------
HAMDataset
    Map-style dataset that loads dermoscopic images from a DataFrame,
    optionally applies hair removal, and returns (tensor, label) pairs.

Functions
---------
get_transforms(split)
    Returns the appropriate torchvision transform pipeline.
get_splits(df, seed)
    Produces lesion-grouped, class-stratified train / val / test DataFrames
    (70 / 15 / 15 %); see ``src/split.py``.
build_loaders(df, seed, batch_size, num_workers, apply_hair_removal)
    Returns (train_loader, val_loader, test_loader) ready for training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.cache import ImageCache
from src.config import (
    HAM_IMAGES_DIR_1,
    HAM_IMAGES_DIR_2,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    LABEL_MAP,
    SEED,
    SPLITS_CSV,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.preprocess import read_image_bgr, remove_hair
from src.split import make_lesion_split, split_frames

# ──────────────────────────────────────────────────────────────────────────────
# Transform pipelines
# ──────────────────────────────────────────────────────────────────────────────


def get_transforms(split: str = "train") -> T.Compose:
    """
    Build the torchvision transform pipeline for a given data split.

    Training augmentations include spatial flips, rotation, and colour
    jitter to improve generalisation on the heavily class-imbalanced
    HAM10000 dataset.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
        Which augmentation set to use.

    Returns
    -------
    torchvision.transforms.Compose
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=30),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                T.ToTensor(),
                normalize,
            ]
        )
    else:  # val / test
        return T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                normalize,
            ]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────


class HAMDataset(Dataset):
    """
    Map-style PyTorch Dataset for the HAM10000 dermoscopic image collection.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain columns ``image_id`` (str) and ``dx`` (str label).
    transform : torchvision.transforms.Compose, optional
        Transform applied to each PIL image before returning.  Defaults to
        the val/test pipeline (resize + tensor + normalise).
    apply_hair_removal : bool
        If ``True``, ``remove_hair()`` is applied to each image at load time.
        For large training runs consider preprocessing offline with
        ``preprocess.batch_preprocess()`` and setting this to ``False``.
    image_dirs : list of str, optional
        Directories to search for image files.  Defaults to the two standard
        HAM10000 image folders defined in ``config.py``.
    cache : ImageCache, optional
        Pre-computed (hair-removed, resized) images from ``src/cache.py``.  When
        given, images are read from it and ``apply_hair_removal`` is ignored.

    Item  shape
    -----------
    Tensor  : (3, IMG_SIZE, IMG_SIZE) float32
    Label   : int (0–6, from LABEL_MAP)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[T.Compose] = None,
        apply_hair_removal: bool = True,
        image_dirs: Optional[list[str]] = None,
        cache: Optional[ImageCache] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform if transform is not None else get_transforms("val")
        self.apply_hair_removal = apply_hair_removal

        # Build image_id → filepath lookup
        if image_dirs is None:
            image_dirs = [HAM_IMAGES_DIR_1, HAM_IMAGES_DIR_2]
        self.image_dirs = [Path(d) for d in image_dirs if os.path.isdir(d)]

        self.cache = cache
        self._path_cache: dict[str, Path] = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_image(self, image_id: str) -> Path:
        """Locate the .jpg file for `image_id` across all image directories."""
        if image_id in self._path_cache:
            return self._path_cache[image_id]

        for d in self.image_dirs:
            candidate = d / f"{image_id}.jpg"
            if candidate.exists():
                self._path_cache[image_id] = candidate
                return candidate

        raise FileNotFoundError(f"Image '{image_id}.jpg' not found in: {self.image_dirs}")

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id: str = row["image_id"]
        label: int = LABEL_MAP[row["dx"]]

        if self.cache is not None:
            return self.transform(self.cache.get(image_id)), label

        # Load image via Unicode-safe BGR reader then convert to RGB PIL
        path = self._find_image(image_id)
        img_bgr = read_image_bgr(path)

        if self.apply_hair_removal:
            img_bgr = remove_hair(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        tensor = self.transform(pil_img)  # → (3, H, W) float32
        return tensor, label

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def labels(self) -> list[int]:
        """All integer labels in dataset order (used by WeightedRandomSampler)."""
        return [LABEL_MAP[dx] for dx in self.df["dx"]]


# ──────────────────────────────────────────────────────────────────────────────
# Data-splitting utilities
# ──────────────────────────────────────────────────────────────────────────────


def get_splits(
    df: pd.DataFrame,
    seed: int = SEED,
    group_by_lesion: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    70 / 15 / 15 train / val / test split.

    With ``group_by_lesion=True`` (default) every ``lesion_id`` lands in exactly one
    split and classes are stratified at lesion level (``src/split.py``).  The
    image-level fallback (``group_by_lesion=False``) leaks near-duplicate images of
    one lesion across splits and exists only to reproduce the original, flawed
    protocol; do not use it for reported results.

    Parameters
    ----------
    df : pd.DataFrame
        Full metadata DataFrame with ``image_id``, ``lesion_id`` and ``dx``.
    seed : int
        Random state for reproducibility.
    group_by_lesion : bool, default=True

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
    """
    if group_by_lesion:
        if "lesion_id" not in df.columns:
            raise ValueError("lesion_id column is required for a lesion-grouped split")
        merged = df.merge(make_lesion_split(df, seed=seed)[["image_id", "split"]], on="image_id")
        return split_frames(merged)

    train_val_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, stratify=df["dx"], random_state=seed
    )
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_fraction, stratify=train_val_df["dx"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# Private re-export for config.py (avoids circular import)
TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────


def _head_per_class(frame: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """Small class-balanced subset (smoke tests only); keeps every class present."""
    per_class = max(1, -(-n_total // frame["dx"].nunique()))
    return frame.groupby("dx", sort=True).head(per_class).reset_index(drop=True)


def build_loaders(
    df: pd.DataFrame,
    seed: int = SEED,
    batch_size: int = 32,
    num_workers: int = 4,
    apply_hair_removal: bool = True,
    use_weighted_sampler: bool = True,
    splits_csv: Optional[str] = SPLITS_CSV,
    cache_dir: Optional[str] = None,
    smoke_n: Optional[int] = None,
    gpu_pipeline: Optional[bool] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Create train / val / test DataLoaders from the full metadata DataFrame.

    The split is lesion-grouped and class-stratified.  When ``splits_csv`` is given
    it is loaded (or created on first use) so the exact assignment is auditable
    and identical across runs; a CSV that does not match ``df`` is rejected.

    The training loader uses ``WeightedRandomSampler`` by default to mitigate
    the class imbalance present in HAM10000 (~67 % nevi).

    Parameters
    ----------
    df : pd.DataFrame
        Full HAM10000 metadata (image_id, lesion_id, dx, ...).
    seed : int
        Seed for the split (only used when the CSV does not exist yet).
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Dataloader worker processes (persistent, so Windows spawn cost is paid once).
    apply_hair_removal : bool
        Forwarded to ``HAMDataset`` (ignored for images served from the cache).
    use_weighted_sampler : bool
        Use ``WeightedRandomSampler`` on the training set.
    splits_csv : str or None
        Path of the audited split CSV; ``None`` splits in memory without saving.
    cache_dir : str or None
        If it holds a cache built by ``src/cache.py`` it is used for all splits.
    smoke_n : int or None
        Debug only: class-balanced subset of about this many training images
        (and half as many for val / test).
    gpu_pipeline : bool or None
        Serve batches from RAM with augmentation on ``device`` (``src/gpu_data.py``)
        instead of multi-process DataLoaders.  ``None`` = on whenever the cache exists.
    device : torch.device or None
        Device of the GPU pipeline (default: CUDA if available).

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    from src.loss import make_weighted_sampler  # local import avoids circularity
    from src.split import load_or_create_splits

    if splits_csv:
        table = load_or_create_splits(df, splits_csv, seed=seed)
    else:
        table = make_lesion_split(df, seed=seed)
    train_df, val_df, test_df = split_frames(table)
    if smoke_n:
        train_df = _head_per_class(train_df, smoke_n)
        val_df = _head_per_class(val_df, max(smoke_n // 2, 7))
        test_df = _head_per_class(test_df, max(smoke_n // 2, 7))

    cache = None
    if cache_dir and ImageCache.exists(cache_dir, apply_hair_removal):
        cache = ImageCache(cache_dir, apply_hair_removal)

    def _ds(frame: pd.DataFrame, split: str) -> HAMDataset:
        return HAMDataset(
            frame,
            transform=get_transforms(split),
            apply_hair_removal=apply_hair_removal,
            cache=cache,
        )

    if gpu_pipeline is None:
        gpu_pipeline = cache is not None
    if gpu_pipeline and cache is None:
        raise ValueError("gpu_pipeline needs the image cache (src/cache.py)")

    train_dataset = _ds(train_df, "train")
    val_dataset = _ds(val_df, "val")
    test_dataset = _ds(test_df, "test")

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset.labels)
        shuffle = False  # mutually exclusive with sampler

    if gpu_pipeline:
        from src.gpu_data import GPUBatchLoader

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = cache.load_all()

        def _gpu(frame: pd.DataFrame, **kw) -> GPUBatchLoader:
            labels = [LABEL_MAP[d] for d in frame["dx"]]
            rows = [cache.row(i) for i in frame["image_id"]]
            return GPUBatchLoader(images, rows, labels, frame, batch_size, dev, **kw)

        return (
            _gpu(train_df, train=True, sampler=sampler, drop_last=True),
            _gpu(val_df),
            _gpu(test_df),
        )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset, sampler=sampler, shuffle=shuffle, drop_last=True, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return train_loader, val_loader, test_loader
