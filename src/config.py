"""
config.py
=========
Central configuration for the HAM10000 skin-lesion classification project.
All hyper-parameters, paths, and reproducibility seeds live here so that
every other module can simply `from src.config import *` or import
individual constants.
"""

import os
import random
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slightly slower but fully reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Image settings
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE: int = 224  # Both EfficientNet-B3 and ViT-B/16 accept 224×224

# ImageNet normalisation statistics (used for pre-trained backbone weights)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float]  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Dataset / label map
# ──────────────────────────────────────────────────────────────────────────────
# HAM10000 lesion categories
LABEL_MAP: dict[str, int] = {
    "nv":    0,   # Melanocytic nevi
    "mel":   1,   # Melanoma
    "bkl":   2,   # Benign keratosis-like lesions
    "bcc":   3,   # Basal cell carcinoma
    "akiec": 4,   # Actinic keratoses / intraepithelial carcinoma
    "vasc":  5,   # Vascular lesions
    "df":    6,   # Dermatofibroma
}

# Reverse mapping: integer → class abbreviation
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Human-readable class names (same order as LABEL_MAP values)
CLASS_NAMES: list[str] = [
    "Melanocytic Nevi",
    "Melanoma",
    "Benign Keratosis",
    "Basal Cell Carcinoma",
    "Actinic Keratoses",
    "Vascular Lesions",
    "Dermatofibroma",
]

NUM_CLASSES: int = len(LABEL_MAP)  # 7

# ──────────────────────────────────────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE: int   = 32
NUM_EPOCHS: int   = 30
LR: float         = 1e-4          # Initial learning rate (AdamW)
WEIGHT_DECAY: float = 1e-4        # L2 regularisation
GRAD_CLIP: float  = 1.0           # Max gradient norm for clipping
UNFREEZE_EPOCH: int = 5           # Epoch at which backbone is unfrozen

# ReduceLROnPlateau settings
LR_PATIENCE: int  = 3             # Epochs with no improvement before LR drops
LR_FACTOR: float  = 0.5           # Multiplicative factor for LR reduction
LR_MIN: float     = 1e-7          # Lower bound on learning rate

# EarlyStopping settings
ES_PATIENCE: int  = 7             # Epochs to wait before stopping
ES_MIN_DELTA: float = 1e-4        # Minimum change to qualify as improvement

# FocalLoss settings
FOCAL_GAMMA: float = 2.0

# ──────────────────────────────────────────────────────────────────────────────
# Data split ratios
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO: float = 0.70
VAL_RATIO: float   = 0.15
TEST_RATIO: float  = 0.15         # Implicit: 1 - TRAIN_RATIO - VAL_RATIO

# ──────────────────────────────────────────────────────────────────────────────
# Directory paths  (project root is the parent of this file's directory)
# ──────────────────────────────────────────────────────────────────────────────
# Resolve project root relative to this file so the package is portable.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR      = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR        = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR   = os.path.join(MODELS_DIR, "checkpoints")

OUTPUTS_DIR       = os.path.join(PROJECT_ROOT, "outputs")
NOTEBOOKS_DIR     = os.path.join(PROJECT_ROOT, "notebooks")
DOCS_DIR          = os.path.join(PROJECT_ROOT, "docs")

# Convenience: paths to specific dataset artefacts inside RAW_DATA_DIR
HAM_METADATA_CSV = os.path.join(RAW_DATA_DIR, "HAM10000_metadata.csv")
HAM_IMAGES_DIR_1 = os.path.join(RAW_DATA_DIR, "HAM10000_images_part_1")
HAM_IMAGES_DIR_2 = os.path.join(RAW_DATA_DIR, "HAM10000_images_part_2")

# Paths for saved model checkpoints
EFFICIENTNET_CKPT = os.path.join(CHECKPOINTS_DIR, "best_efficientnet_b3.pth")
VIT_CKPT          = os.path.join(CHECKPOINTS_DIR, "best_vit_b16.pth")

# ──────────────────────────────────────────────────────────────────────────────
# Ensure output directories exist at import time
# ──────────────────────────────────────────────────────────────────────────────
for _dir in [PROCESSED_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
