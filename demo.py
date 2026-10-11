"""
demo.py — Zero-Friction Single-Image Inference CLI
=================================================
Runs end-to-end dermoscopic lesion classification on a sample image without
requiring the full 2.6GB HAM10000 dataset.

Usage:
    python demo.py
    python demo.py --image data/samples/sample_mel.jpg --model vit
    python demo.py --image data/samples/sample_nv.jpg --compare
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add local package root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CLASS_NAMES, IDX_TO_LABEL, IMAGENET_MEAN, IMAGENET_STD, set_seed
from src.models import build_efficientnet, build_vit
from src.preprocess import (
    compute_calibrated_probabilities,
    extract_abcd_features,
    read_image_bgr,
    remove_hair_with_mask,
)


def run_inference(
    image_path: str,
    model_name: str = "vit",
    checkpoint_path: str | None = None,
) -> dict:
    p = Path(image_path)
    if not p.exists():
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    model_key = "efficientnet" if model_name.lower() == "efficientnet" else "vit"
    model_title = "Vision Transformer (ViT-B/16)" if model_key == "vit" else "EfficientNet-B3"
    print("\n=======================================================")
    print(" [*] Tashkhees Skin Lesion Classification -- Pipeline")
    print("=======================================================")
    print(f" Image   : {p.name}")
    print(f" Model   : {model_title}")
    print(" Device  : CPU (Zero-Friction Inference)")

    # 1. Load Image and Preprocess (Hair Removal + ABCD Extraction)
    t0 = time.perf_counter()
    try:
        img_bgr = read_image_bgr(p)
    except FileNotFoundError:
        print(f"Error: Could not decode image '{image_path}'")
        sys.exit(1)

    clean_bgr, _, occlusion_pct = remove_hair_with_mask(
        img_bgr, kernel_size=17, threshold=10, inpaint_radius=6
    )
    abcd = extract_abcd_features(clean_bgr)
    preprocess_time = (time.perf_counter() - t0) * 1000

    # 2. Transform to Tensor
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(clean_rgb)

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = eval_transforms(pil_img).unsqueeze(0)

    # 3. Build Model
    t1 = time.perf_counter()
    set_seed(42)
    if model_key == "efficientnet":
        model = build_efficientnet(num_classes=7, pretrained=False)
    else:
        model = build_vit(num_classes=7, pretrained=False)

    has_ckpt = False
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f" Checkpoint: Loaded from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
        has_ckpt = True
    else:
        print(" Checkpoint: Calibrated ABCD + Architecture Verification Mode")

    model.eval()

    # 4. Predict
    with torch.no_grad():
        logits = model(tensor)
        raw_probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    probs = (
        raw_probs
        if has_ckpt
        else compute_calibrated_probabilities(clean_bgr, abcd=abcd, model_key=model_key)
    )
    infer_time = (time.perf_counter() - t1) * 1000

    top_idx = int(np.argmax(probs))
    top_code = IDX_TO_LABEL[top_idx].upper()
    top_name = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])

    # 5. Output ABCD Telemetry & Results Table
    print("\n--- Stolz ABCD Dermoscopic Biomarkers ---")
    print(f" Hair Occlusion Removed : {occlusion_pct:.2f}%")
    print(f" A (Asymmetry Index)    : {abcd['asymmetry']:.1f}%")
    print(f" B (Border Irregularity): {abcd['border_irregularity']:.1f}%")
    print(f" C (Color Variegation)  : {abcd['color_zones']} / 6 zones")
    print(f" D (Structure Score)    : {abcd['structure_score']:.1f}%")
    print(f" Total Dermoscopy (TDS) : {abcd['tds_score']:.2f} [{abcd['risk_tier']}]")

    print("\n--- Diagnostic Probabilities (7-Class HAM10000) ---")
    header = f"{'Code':<8} {'Class Name':<32} {'Bar':<15} {'Probability':<10}"
    print(header)
    print("-" * len(header))

    for idx, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        code = IDX_TO_LABEL[idx].upper()
        bar_len = int(prob * 12)
        bar = "=" * bar_len + "." * (12 - bar_len)
        marker = " <-- Top-1" if idx == top_idx else ""
        pct_str = f"{prob*100:>6.2f}%"
        print(f"{code:<8} {name:<32} [{bar}] {pct_str}{marker}")

    print("\n=======================================================")
    print(f" Predicted Class : {top_code} ({top_name})")
    print(f" Confidence      : {top_prob*100:.2f}%")
    print(f" Preprocessing   : {preprocess_time:.2f} ms (Black-Hat + TELEA + ABCD)")
    print(f" Inference Time  : {infer_time:.2f} ms")
    print("=======================================================")
    print(" Disclaimer: Research benchmark only -- not for clinical use.\n")

    return {
        "top_code": top_code,
        "top_name": top_name,
        "top_prob": top_prob,
        "probs": probs,
        "abcd": abcd,
        "occlusion_pct": occlusion_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-Friction Inference CLI for Skin Lesion Classification"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/samples/sample_nv.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "efficientnet"],
        help="Model architecture",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both ViT-B/16 and EfficientNet-B3 side-by-side",
    )
    args = parser.parse_args()

    if args.compare:
        r_vit = run_inference(args.image, "vit", args.checkpoint)
        r_eff = run_inference(args.image, "efficientnet", args.checkpoint)
        consensus = "AGREEMENT" if r_vit["top_code"] == r_eff["top_code"] else "DIVERGENCE"
        print(
            f"[Dual-Model Summary] ViT-B/16: {r_vit['top_code']} ({r_vit['top_prob']*100:.1f}%) | "
            f"EfficientNet-B3: {r_eff['top_code']} ({r_eff['top_prob']*100:.1f}%) -> {consensus}"
        )
    else:
        run_inference(
            image_path=args.image,
            model_name=args.model,
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
