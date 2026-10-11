# Models Directory

Trained model checkpoints are **not committed directly to Git** due to their size (`.pth` files exceed GitHub's 100 MB limit).

---

## Live Interactive Demo (Zero-Install)

You can test the full preprocessing, Stolz ABCD dermoscopic biomarker extraction, Grad-CAM / ViT patch attention maps, and 7-class HAM10000 inference live on **Hugging Face Spaces**:

> 🔬 **[Live Demo: Yasir-Alazmi/tashkhees-skin-lesion](https://huggingface.co/spaces/Yasir-Alazmi/tashkhees-skin-lesion)**

---

## Checkpoint Filenames (`src/config.py`)

When training locally via `python -m src.train`, checkpoints are saved automatically to `models/checkpoints/`:

| File | Model | 30-Epoch Benchmark Acc / AUC | 15-Epoch Focal-Loss Test AUC |
|------|-------|------------------------------|------------------------------|
| `best_efficientnet_b3.pth` | EfficientNet-B3 | 92.6% / 0.95 | 0.9333 (`N=1,503`) |
| `best_vit_b16.pth` | **ViT-B/16** | **94.2% / 0.96** | **0.9743 (`N=1,503`)** |

---

## Checkpoint Format

Each `.pth` checkpoint contains:

```python
{
    "epoch": int,                   # epoch at which checkpoint was saved
    "model_state_dict": dict,       # model weights
    "val_auc": float,               # validation macro AUC
    "val_acc": float,               # validation accuracy
}
```

### Loading a Checkpoint

```python
import torch
from src.models import build_vit
from src.config import DEVICE, VIT_CKPT

model = build_vit(num_classes=7, pretrained=False)
ckpt = torch.load(VIT_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']} | Val AUC: {ckpt['val_auc']:.4f}")
```

---

## Training Your Own Checkpoints

```bash
# Dry-run verification (no dataset required)
python -m src.train --model vit --dry-run

# Train EfficientNet-B3 — saves to models/checkpoints/best_efficientnet_b3.pth
python -m src.train --model efficientnet

# Train ViT-B/16 — saves to models/checkpoints/best_vit_b16.pth
python -m src.train --model vit --epochs 30 --batch-size 16 --lr 1e-4
```
