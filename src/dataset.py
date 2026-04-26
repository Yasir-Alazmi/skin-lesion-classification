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
    Produces stratified train / val / test DataFrames (70 / 15 / 15 %).
build_loaders(df, seed, batch_size, num_workers, apply_hair_removal)
    Returns (train_loader, val_loader, test_loader) ready for training.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from src.config import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAIN_RATIO,
    VAL_RATIO,
    SEED,
    LABEL_MAP,
    HAM_IMAGES_DIR_1,
    HAM_IMAGES_DIR_2,
)
from src.preprocess import remove_hair


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
        return T.Compose([
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
        ])
    else:  # val / test
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            normalize,
        ])


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
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform if transform is not None else get_transforms("val")
        self.apply_hair_removal = apply_hair_removal

        # Build image_id → filepath lookup
        if image_dirs is None:
            image_dirs = [HAM_IMAGES_DIR_1, HAM_IMAGES_DIR_2]
        self.image_dirs = [Path(d) for d in image_dirs if os.path.isdir(d)]

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

        raise FileNotFoundError(
            f"Image '{image_id}.jpg' not found in: {self.image_dirs}"
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id: str = row["image_id"]
        label: int = LABEL_MAP[row["dx"]]

        # Load image via OpenCV (BGR) then convert to RGB PIL
        path = self._find_image(image_id)
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise RuntimeError(f"cv2.imread failed for: {path}")

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
        return [LABEL_MAP[row["dx"]] for _, row in self.df.iterrows()]


# ──────────────────────────────────────────────────────────────────────────────
# Data-splitting utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_splits(
    df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70 / 15 / 15 train / val / test split.

    Stratification is on the ``dx`` column to preserve the class distribution
    in every split, which is crucial for the highly imbalanced HAM10000 set.

    Parameters
    ----------
    df : pd.DataFrame
        Full metadata DataFrame.  Must contain a ``dx`` column.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
    """
    # Step 1: separate out test set (15 %)
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["dx"],
        random_state=seed,
    )

    # Step 2: split the remainder into train (70 %) / val (15 %)
    # val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO) = 0.15 / 0.85 ≈ 0.176
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        stratify=train_val_df["dx"],
        random_state=seed,
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

def build_loaders(
    df: pd.DataFrame,
    seed: int = SEED,
    batch_size: int = 32,
    num_workers: int = 4,
    apply_hair_removal: bool = True,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders from the full metadata DataFrame.

    The training loader uses ``WeightedRandomSampler`` by default to mitigate
    the class imbalance present in HAM10000 (~67 % nevi).

    Parameters
    ----------
    df : pd.DataFrame
        Full HAM10000 metadata (image_id, dx, …).
    seed : int
        Seed forwarded to ``get_splits()``.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Dataloader worker processes.
    apply_hair_removal : bool
        Forwarded to ``HAMDataset``.
    use_weighted_sampler : bool
        Use ``WeightedRandomSampler`` on the training set.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    from src.loss import make_weighted_sampler  # local import avoids circularity

    train_df, val_df, test_df = get_splits(df, seed=seed)

    train_dataset = HAMDataset(
        train_df,
        transform=get_transforms("train"),
        apply_hair_removal=apply_hair_removal,
    )
    val_dataset = HAMDataset(
        val_df,
        transform=get_transforms("val"),
        apply_hair_removal=apply_hair_removal,
    )
    test_dataset = HAMDataset(
        test_df,
        transform=get_transforms("test"),
        apply_hair_removal=apply_hair_removal,
    )

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset.labels)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
