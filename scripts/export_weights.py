"""Export a slim, weights-only ViT-B/16 file for upload to the Hugging Face Hub.

Usage:  python scripts/export_weights.py <checkpoint.pth> [out.pth]
The training checkpoint already holds only the best weights; this keeps just
``model_state_dict`` so the uploaded file works with ``app.py`` unchanged.
"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "best_vit_b16.pth"
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({"model_state_dict": ckpt.get("model_state_dict", ckpt)}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
