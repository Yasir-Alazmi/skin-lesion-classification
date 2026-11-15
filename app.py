"""
app.py -- Gradio research demo for the ConvNeXt-S ensemble skin-lesion classifier
================================================================================
Runs the real models (no simulated outputs) on an uploaded dermoscopic image: hair removal ->
resize (384 / 448) -> ImageNet norm -> flip TTA -> log-prob average + class bias.
Shows the top-3 class probabilities of the 7 HAM10000 classes.

Ensemble weights resolution (first that works):
  1. ``ENSEMBLE_DIR`` environment variable (directory with ensemble_config.json + .pt files),
  2. the repo directory ``models/ensemble/``,
  3. Hugging Face Hub: ``ENSEMBLE_REPO`` (default ``Yasir-Alazmi/skin-lesion-convnext-ensemble``).
The older single ViT-B/16 path (``resolve_weights``, used by ``demo.py --weights``) is kept.

Run locally:  python app.py        Deploy: see "Hugging Face Space" in README.md.
RESEARCH DEMO ONLY -- not a medical device, not clinically validated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import VIT_CKPT  # noqa: E402
from src.ensemble import DEFAULT_DIR, load_ensemble  # noqa: E402
from src.inference import load_model, predict_probs, top_k  # noqa: E402

MODEL_REPO = os.environ.get("MODEL_REPO", "Yasir-Alazmi/skin-lesion-vit-b16")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_vit_b16.pth")
ENSEMBLE_REPO = os.environ.get("ENSEMBLE_REPO", "Yasir-Alazmi/skin-lesion-convnext-ensemble")
SAMPLES_DIR = Path(__file__).parent / "data" / "samples"
EXAMPLES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

DISCLAIMER = (
    "**Research demo, not a medical device.** This model was trained on HAM10000 only and "
    "has not been clinically validated. On held-out data it detects about 72% of melanomas, "
    "so roughly one in four is missed and a wrong or reassuring answer is likely. Never use it to make or delay a medical "
    "decision; see a clinician for any skin concern."
)

_MODEL = None
_ENSEMBLE = None


def resolve_weights() -> str:
    """Path to the ViT-B/16 weights file (see module docstring for the order)."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    except Exception as exc:  # repo missing, offline, library absent
        print(f"Hub download unavailable ({type(exc).__name__}); using local {VIT_CKPT}")
        return VIT_CKPT


def resolve_ensemble_dir() -> str:
    """Directory holding ``ensemble_config.json`` and the member weights."""
    env = os.environ.get("ENSEMBLE_DIR")
    if env:
        return env
    if (DEFAULT_DIR / "ensemble_config.json").is_file() and any(DEFAULT_DIR.glob("*.pt")):
        return str(DEFAULT_DIR)
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(ENSEMBLE_REPO)
    local = [hf_hub_download(repo_id=ENSEMBLE_REPO, filename=f) for f in files if f.endswith((".pt", ".json"))]
    return str(Path(local[0]).parent)


def get_ensemble():
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = load_ensemble(resolve_ensemble_dir())
    return _ENSEMBLE


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(resolve_weights())
    return _MODEL


def classify(image: np.ndarray | None) -> dict[str, float]:
    """Gradio callback: RGB array -> {"Class name (code)": probability} for all classes."""
    if image is None:
        raise ValueError("Please upload or select a dermoscopic image.")
    probs = get_ensemble().predict_probs(image)
    full = {f"{t['name']} ({t['code']})": t["prob"] for t in top_k(probs, k=len(probs))}
    return full


def build_demo():
    import gradio as gr

    examples = [str(SAMPLES_DIR / f"sample_{c}.jpg") for c in EXAMPLES]
    examples = [p for p in examples if os.path.exists(p)]
    with gr.Blocks(title="Skin Lesion Classifier (research demo)") as demo:
        gr.Markdown("# Skin lesion classifier: ConvNeXt-S ensemble on HAM10000")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            inp = gr.Image(type="numpy", label="Dermoscopic image", image_mode="RGB")
            out = gr.Label(num_top_classes=3, label="Top-3 class probabilities")
        btn = gr.Button("Classify", variant="primary")
        btn.click(classify, inputs=inp, outputs=out)
        if examples:
            gr.Examples(
                examples=examples, inputs=inp, outputs=out, fn=classify, cache_examples=False
            )
        gr.Markdown(
            "Pipeline: hair removal (Black-Hat + TELEA) -> resize 384/448 -> ImageNet normalisation -> "
            "4 ConvNeXt-S models with flip TTA -> log-prob average + class bias. Reported test results "
            "(lesion-grouped split, 5-member ensemble): balanced accuracy 0.82, macro-F1 0.79, accuracy "
            "0.84. This demo runs a 4-member subset whose own score was not measured."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
