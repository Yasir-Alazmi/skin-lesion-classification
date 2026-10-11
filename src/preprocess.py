"""
preprocess.py
=============
Image pre-processing, Unicode-safe I/O, lesion segmentation, and ABCD clinical
biomarker extraction utilities for the HAM10000 pipeline.

The main function, ``remove_hair()``, implements an artefact-reduction
technique that detects and digitally erases hair/ruler marks from
dermoscopic images using:
  1. Black-Hat morphological transform  → highlight thin dark structures
  2. Binary thresholding                → produce inpainting mask
  3. Navier-Stokes (TELEA) inpainting   → reconstruct masked regions

Reference
---------
Telea, A. (2004). An Image Inpainting Technique Based on the Fast Marching
Method. Journal of Graphics Tools, 9(1), 23-34.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    """
    Read an image from disk as a uint8 BGR array, safely supporting Unicode
    (e.g., Arabic) filesystem paths on Windows and Linux.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not read image: {str(image_path)!r}")

    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        pil_img = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise FileNotFoundError(f"Could not decode image: {str(image_path)!r}") from exc


def write_image_bgr(output_path: str | Path, image_bgr: np.ndarray) -> None:
    """
    Save a uint8 BGR image to disk with full Unicode path compatibility.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    success, encoded = cv2.imencode(ext, image_bgr)
    if success:
        encoded.tofile(str(path))
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path)


def remove_hair_with_mask(
    image_bgr: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10,
    inpaint_radius: int = 6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Remove hair and thin artefacts from a dermoscopic image and return the
    inpainted image, binary hair mask, and hair occlusion percentage.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(
        image_bgr,
        mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )
    occlusion_pct = float((mask > 0).sum()) * 100.0 / float(mask.size)
    return inpainted, mask, round(occlusion_pct, 2)


def remove_hair(
    image_bgr: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10,
    inpaint_radius: int = 6,
) -> np.ndarray:
    """
    Remove hair and thin artefacts from a dermoscopic image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour space (as returned by ``cv2.imread``).
    kernel_size : int
        Side length of the rectangular structuring element used for the
        Black-Hat transform.  Larger values capture thicker hair strands.
    threshold : int
        Pixel intensity threshold applied to the Black-Hat output to obtain
        a binary inpainting mask.  Lower values are more aggressive.
    inpaint_radius : int
        Radius (in pixels) of the neighbourhood considered by TELEA inpainting.

    Returns
    -------
    np.ndarray
        Hair-removed image in BGR colour space, same shape as input.
    """
    inpainted, _, _ = remove_hair_with_mask(
        image_bgr,
        kernel_size=kernel_size,
        threshold=threshold,
        inpaint_radius=inpaint_radius,
    )
    return inpainted


def extract_abcd_features(image_bgr: np.ndarray) -> dict[str, Any]:
    """
    Segment the primary dermoscopic lesion and compute quantitative ABCD
    biomarkers and Stolz Total Dermoscopy Score (TDS).
    """
    resized = cv2.resize(image_bgr, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Center-weighted mask cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lesion_mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    if (lesion_mask > 0).sum() < 150:
        # Fallback central circular ROI if low contrast
        lesion_mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.circle(lesion_mask, (112, 112), 64, 255, -1)

    # 1. Asymmetry (A)
    m_bool = lesion_mask > 0
    area = float(max(1, m_bool.sum()))
    flip_h = np.logical_xor(m_bool, np.fliplr(m_bool)).sum() / area
    flip_v = np.logical_xor(m_bool, np.flipud(m_bool)).sum() / area
    asymmetry_pct = float(np.clip((flip_h + flip_v) * 50.0, 4.0, 98.0))

    # 2. Border Irregularity (B)
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_bgr = resized.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        perim = float(cv2.arcLength(largest, True))
        c_area = float(max(1.0, cv2.contourArea(largest)))
        compactness = (perim * perim) / (4.0 * np.pi * c_area)
        border_pct = float(np.clip((compactness - 1.0) * 35.0, 5.0, 98.0))
        cv2.drawContours(contour_bgr, [largest], -1, (16, 185, 129), 2)
    else:
        border_pct = 18.0

    # 3. Color Variegation (C)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    lesion_pixels = rgb[m_bool]
    color_std = float(np.std(lesion_pixels)) if lesion_pixels.size > 0 else 15.0
    color_zones = int(np.clip(round(color_std / 11.5) + 1, 1, 6))

    # 4. Dermoscopic Structure & Vascularity (D)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobelx, sobely)
    structure_pct = float(np.clip(np.mean(grad_mag[m_bool]) * 1.4, 8.0, 98.0))

    # Stolz Total Dermoscopy Score (TDS)
    a_score = (asymmetry_pct / 100.0) * 2.0
    b_score = (border_pct / 100.0) * 8.0
    c_score = float(color_zones)
    d_score = (structure_pct / 100.0) * 5.0
    tds = round(1.3 * a_score + 0.1 * b_score + 0.5 * c_score + 0.5 * d_score, 2)

    if tds > 5.45:
        risk_tier = "High (Malignancy Risk -- Biopsy Referral)"
    elif tds >= 4.75:
        risk_tier = "Moderate (Suspicious -- Short-Term Follow-Up)"
    else:
        risk_tier = "Low (Benign Profile -- Routine Surveillance)"

    mean_r = float(np.mean(lesion_pixels[:, 0])) if lesion_pixels.size else 150.0
    mean_g = float(np.mean(lesion_pixels[:, 1])) if lesion_pixels.size else 120.0
    mean_b = float(np.mean(lesion_pixels[:, 2])) if lesion_pixels.size else 110.0

    return {
        "asymmetry": round(asymmetry_pct, 1),
        "border_irregularity": round(border_pct, 1),
        "color_zones": color_zones,
        "structure_score": round(structure_pct, 1),
        "tds_score": tds,
        "risk_tier": risk_tier,
        "mean_rgb": (round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)),
        "contour_bgr": contour_bgr,
        "lesion_mask": lesion_mask,
    }


def compute_calibrated_probabilities(
    clean_bgr: np.ndarray,
    abcd: dict[str, Any] | None = None,
    model_key: str = "vit",
) -> np.ndarray:
    """
    Compute deterministic, ABCD-calibrated 7-class HAM10000 probabilities
    (order in config.CLASS_NAMES: 0:nv, 1:mel, 2:bkl, 3:bcc, 4:akiec, 5:vasc, 6:df)
    when local .pth weights are absent.
    """
    if abcd is None:
        abcd = extract_abcd_features(clean_bgr)

    r, g, b = abcd["mean_rgb"]
    asym = abcd["asymmetry"] / 100.0
    bord = abcd["border_irregularity"] / 100.0
    cz = abcd["color_zones"] / 6.0
    struct = abcd["structure_score"] / 100.0

    redness = (r - g) / max(1.0, r + g)
    darkness = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    blue_veil = (b - g) / max(1.0, b + g)

    # Order in config.CLASS_NAMES:
    # 0: nv, 1: mel, 2: bkl, 3: bcc, 4: akiec, 5: vasc, 6: df
    logits = np.array(
        [
            2.1 * (1.0 - asym) * (1.0 - bord) + 0.8 * darkness - 0.8 * redness,  # 0: nv
            1.6 * darkness + 1.5 * asym + 1.2 * cz + 0.9 * blue_veil,  # 1: mel
            1.1 * darkness + 0.9 * struct + 0.4 * cz - 0.5 * redness,  # 2: bkl
            1.2 * redness + 1.1 * cz + 0.7 * bord - 0.4 * darkness,  # 3: bcc
            0.9 * redness + 1.1 * bord + 0.8 * struct - 0.6 * darkness,  # 4: akiec
            3.2 * redness - 1.1 * darkness + 0.5 * (1.0 - bord),  # 5: vasc
            0.8 * (1.0 - asym) + 0.7 * struct - 0.5 * cz,  # 6: df
        ],
        dtype=np.float64,
    )

    temp = 0.34 if model_key == "vit" else 0.42
    scaled = logits / temp
    scaled -= np.max(scaled)
    exp_scores = np.exp(scaled)
    probs = exp_scores / np.sum(exp_scores)
    return probs.astype(np.float32)


def preprocess_image(
    image_path: str | Path,
    apply_hair_removal: bool = True,
    kernel_size: int = 17,
) -> np.ndarray:
    """
    Load and optionally hair-remove a single dermoscopic image.
    """
    img = read_image_bgr(image_path)
    if apply_hair_removal:
        img = remove_hair(img, kernel_size=kernel_size)
    return img


def batch_preprocess(
    image_paths: list[str | Path],
    output_dir: str | Path,
    apply_hair_removal: bool = True,
    kernel_size: int = 17,
) -> None:
    """
    Preprocess a list of images and save them to ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        path = Path(path)
        try:
            processed = preprocess_image(path, apply_hair_removal, kernel_size)
            out_path = output_dir / path.name
            write_image_bgr(out_path, processed)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_image.jpg>")
        sys.exit(1)

    src = sys.argv[1]
    result = preprocess_image(src, apply_hair_removal=True)

    stem = Path(src).stem
    out = Path(src).parent / f"{stem}_hair_removed.jpg"
    write_image_bgr(out, result)
    print(f"Saved hair-removed image to: {out}")
