"""
preprocess.py
=============
Image pre-processing utilities for the HAM10000 pipeline.

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

import os
import cv2
import numpy as np
from pathlib import Path


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

    Notes
    -----
    * Input must be a uint8 BGR image with three channels.
    * The function is side-effect free — the original array is not modified.
    """
    # ── 1. Convert to grayscale for morphological analysis ────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── 2. Black-Hat transform ────────────────────────────────────────────────
    # Black-Hat = closing(I) - I  →  highlights dark objects smaller than
    # the structuring element (i.e. hair strands) against a bright background.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # ── 3. Binary threshold → inpainting mask ────────────────────────────────
    # Pixels above `threshold` in the Black-Hat image are treated as hair.
    _, mask = cv2.threshold(
        blackhat, threshold, 255, cv2.THRESH_BINARY
    )

    # ── 4. TELEA inpainting ───────────────────────────────────────────────────
    # Reconstruct masked pixels from surrounding neighbourhood context.
    inpainted = cv2.inpaint(
        image_bgr,
        mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    return inpainted


def preprocess_image(
    image_path: str | Path,
    apply_hair_removal: bool = True,
    kernel_size: int = 17,
) -> np.ndarray:
    """
    Load and optionally hair-remove a single dermoscopic image.

    Parameters
    ----------
    image_path : str or Path
        Filesystem path to the image file (JPEG/PNG).
    apply_hair_removal : bool
        Whether to apply ``remove_hair()``.
    kernel_size : int
        Forwarded to ``remove_hair()``.

    Returns
    -------
    np.ndarray
        Preprocessed image as a uint8 BGR array, or ``None`` if the file
        cannot be read.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path!r}")

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

    Useful for an offline preprocessing step so that training does not pay
    the OpenCV cost at every epoch.

    Parameters
    ----------
    image_paths : list of str or Path
        Source image file paths.
    output_dir : str or Path
        Directory where processed images are saved (same filename).
    apply_hair_removal : bool
        Whether to apply hair removal.
    kernel_size : int
        Structuring-element size for Black-Hat transform.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        path = Path(path)
        try:
            processed = preprocess_image(path, apply_hair_removal, kernel_size)
            out_path = output_dir / path.name
            cv2.imwrite(str(out_path), processed)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test when run as a script
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_image.jpg>")
        sys.exit(1)

    src = sys.argv[1]
    result = preprocess_image(src, apply_hair_removal=True)

    stem = Path(src).stem
    out = Path(src).parent / f"{stem}_hair_removed.jpg"
    cv2.imwrite(str(out), result)
    print(f"Saved hair-removed image to: {out}")
