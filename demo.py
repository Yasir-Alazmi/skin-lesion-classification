"""
demo.py -- command-line inference (ConvNeXt-S ensemble; legacy ViT-B/16 via --weights)
===========================================================
    python demo.py --image data/samples/sample_mel.jpg [--ensemble-dir models/ensemble | --weights path/to/best_vit_b16.pth]

Weights come from ``--weights``, else the same resolution as ``app.py``
(``MODEL_PATH`` / Hugging Face Hub / local checkpoint).
Research use only: not a medical device and not clinically validated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import load_model, predict_probs, top_k  # noqa: E402
from src.models import build_vit  # noqa: E402
from src.preprocess import read_image_bgr  # noqa: E402


def run_inference(
    image_path: str, weights_path: str | None = None, k: int = 3, builder=build_vit
) -> list[dict]:
    """Classify one image file; returns the top-k classes and prints them."""
    if weights_path is None:
        from app import resolve_weights

        weights_path = resolve_weights()
    model = load_model(weights_path, builder=builder)
    image_rgb = read_image_bgr(image_path)[:, :, ::-1]
    result = top_k(predict_probs(model, image_rgb), k=k)
    print(f"Image: {Path(image_path).name}")
    for r in result:
        print(f"  {r['code']:<6} {r['name']:<24} {r['prob'] * 100:5.1f}%")
    print("Research demo only: not a medical device, not clinically validated.")
    return result


def run_ensemble_inference(image_path: str, directory: str | None = None, k: int = 3) -> list[dict]:
    """Classify one image with the ConvNeXt-S ensemble; returns and prints the top-k classes."""
    from app import resolve_ensemble_dir
    from src.ensemble import load_ensemble

    ens = load_ensemble(directory or resolve_ensemble_dir())
    image_rgb = read_image_bgr(image_path)[:, :, ::-1]
    result = top_k(ens.predict_probs(image_rgb), k=k)
    print(f"Image: {Path(image_path).name}")
    for r in result:
        print(f"  {r['code']:<6} {r['name']:<24} {r['prob'] * 100:5.1f}%")
    print("Research demo only: not a medical device, not clinically validated.")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Skin-lesion inference (research only)")
    ap.add_argument("--image", default="data/samples/sample_nv.jpg")
    ap.add_argument("--weights", default=None, help="single ViT-B/16 checkpoint (legacy path)")
    ap.add_argument("--ensemble-dir", default=None, help="directory with ensemble_config.json")
    args = ap.parse_args()
    if args.weights:
        run_inference(args.image, args.weights)
    else:
        run_ensemble_inference(args.image, args.ensemble_dir)


if __name__ == "__main__":
    main()
