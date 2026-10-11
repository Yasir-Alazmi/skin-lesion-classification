"""Comprehensive unit tests for preprocessing, ABCD biomarkers, saliency, and app pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app import classify_lesion  # noqa: E402
from src.preprocess import (  # noqa: E402
    compute_calibrated_probabilities,
    extract_abcd_features,
    read_image_bgr,
    remove_hair_with_mask,
)
from src.train import EarlyStopping  # noqa: E402
from src.visualize import generate_saliency_overlay  # noqa: E402


def test_all_seven_ham10000_samples_exist_and_readable():
    """All 7 real HAM10000 sample images must exist and decode cleanly via Unicode-safe reader."""
    samples_dir = repo_root / "data" / "samples"
    expected = [
        "sample_nv.jpg",
        "sample_mel.jpg",
        "sample_bkl.jpg",
        "sample_bcc.jpg",
        "sample_akiec.jpg",
        "sample_vasc.jpg",
        "sample_df.jpg",
    ]
    for name in expected:
        p = samples_dir / name
        assert p.exists(), f"Missing HAM10000 sample image: {name}"
        img = read_image_bgr(p)
        assert img is not None and img.ndim == 3 and img.shape[2] == 3


def test_remove_hair_with_mask_and_abcd():
    """Black-Hat hair removal and Stolz ABCD feature extractor must return valid bounds."""
    sample_path = repo_root / "data" / "samples" / "sample_mel.jpg"
    img_bgr = read_image_bgr(sample_path)
    clean_bgr, mask, occlusion_pct = remove_hair_with_mask(img_bgr, kernel_size=17, threshold=10)

    assert clean_bgr.shape == img_bgr.shape
    assert mask.shape == img_bgr.shape[:2]
    assert 0.0 <= occlusion_pct <= 100.0

    abcd = extract_abcd_features(clean_bgr)
    assert 0.0 <= abcd["asymmetry"] <= 100.0
    assert 0.0 <= abcd["border_irregularity"] <= 100.0
    assert 1 <= abcd["color_zones"] <= 6
    assert 0.0 <= abcd["structure_score"] <= 100.0
    assert abcd["tds_score"] > 0.0

    probs = compute_calibrated_probabilities(clean_bgr, abcd=abcd, model_key="vit")
    assert probs.shape == (7,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)


def test_saliency_overlay_and_gradio_classify_lesion():
    """Saliency heatmap and Gradio classify_lesion must return valid RGB arrays and 7 classes."""
    sample_path = repo_root / "data" / "samples" / "sample_nv.jpg"
    img_bgr = read_image_bgr(sample_path)
    img_rgb = img_bgr[:, :, ::-1].copy()

    overlay = generate_saliency_overlay(img_rgb)
    assert overlay.shape == (224, 224, 3)

    clean_rgb, contour_rgb, saliency_rgb, confidences, abcd_md = classify_lesion(
        img_rgb, "Vision Transformer (ViT-B/16)", 17, 10
    )
    assert clean_rgb.shape == img_rgb.shape
    assert contour_rgb.shape == (224, 224, 3)
    assert saliency_rgb.shape == (224, 224, 3)
    assert len(confidences) == 7
    assert "Stolz ABCD" in abcd_md


def test_early_stopping_behavior():
    """EarlyStopping should trigger after `patience` non-improving epochs."""
    es = EarlyStopping(patience=2, min_delta=1e-3, mode="max")
    assert not es(0.80)
    assert not es(0.85)  # new best
    assert not es(0.84)  # counter = 1
    assert es(0.84)  # counter = 2 -> stop!
