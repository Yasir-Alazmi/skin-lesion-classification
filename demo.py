"""
demo.py — Zero-Friction Single-Image Inference CLI
=================================================
Runs end-to-end dermoscopic lesion classification on a sample image without
requiring the full 2.6GB HAM10000 dataset.

Usage:
    python demo.py
    python demo.py --image data/samples/sample_vascular_lesion.jpg --model vit
"""

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

from src.config import CLASS_NAMES, IDX_TO_LABEL, IMAGENET_MEAN, IMAGENET_STD
from src.models import build_efficientnet, build_vit
from src.preprocess import remove_hair


def run_inference(image_path: str, model_name: str = "vit", checkpoint_path: str = None):
    p = Path(image_path)
    if not p.exists():
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    model_title = "Vision Transformer (ViT-B/16)" if model_name == "vit" else "EfficientNet-B3"
    print("\n=======================================================")
    print(" [*] Skin Lesion Classification -- Inference Pipeline")
    print("=======================================================")
    print(f" Image   : {p.name}")
    print(f" Model   : {model_title}")
    print(" Device  : CPU (Zero-Friction Inference)")

    # 1. Load Image and Preprocess (Hair Removal)
    t0 = time.perf_counter()
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        print(f"Error: Could not decode image '{image_path}'")
        sys.exit(1)

    clean_bgr = remove_hair(img_bgr, kernel_size=17, threshold=10, inpaint_radius=6)
    preprocess_time = (time.perf_counter() - t0) * 1000

    # 2. Transform to Tensor
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(clean_rgb)

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = eval_transforms(pil_img).unsqueeze(0)

    # 3. Build Model
    t1 = time.perf_counter()
    if model_name.lower() == "efficientnet":
        model = build_efficientnet(num_classes=7, pretrained=False)
    else:
        model = build_vit(num_classes=7, pretrained=False)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f" Checkpoint: Loaded from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
    else:
        print(" Checkpoint: Architecture validation mode (weights uninitialized or default)")

    model.eval()

    # 4. Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    infer_time = (time.perf_counter() - t1) * 1000

    top_idx = int(np.argmax(probs))
    top_code = IDX_TO_LABEL[top_idx].upper()
    top_name = CLASS_NAMES[top_idx]
    top_prob = probs[top_idx]

    # 5. Output Results Table
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
    print(f" Preprocessing   : {preprocess_time:.2f} ms (Black-Hat + TELEA Inpainting)")
    print(f" Inference Time  : {infer_time:.2f} ms")
    print("=======================================================")
    print(" Disclaimer: Research benchmark only -- not for clinical use.\n")


def main():
    parser = argparse.ArgumentParser(description="Zero-Friction Inference CLI for Skin Lesion Classification")
    parser.add_argument("--image", type=str, default="data/samples/sample_melanocytic_nevus.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "efficientnet"], help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to model checkpoint (.pth)")
    args = parser.parse_args()

    run_inference(image_path=args.image, model_name=args.model, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
