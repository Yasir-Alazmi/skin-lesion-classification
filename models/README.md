# Models Directory

Trained model checkpoints are **not committed to this repository** due to their large size (`.pth` files exceed GitHub's 100 MB limit).

---

## Downloading Pre-trained Checkpoints

The best checkpoints from our experiments are hosted on **Hugging Face Hub**:

> 🤗 Coming soon: `Yasir-Alazmi/skin-lesion-ham10000`

Once available, download using:

```python
from huggingface_hub import hf_hub_download

# EfficientNet-B3 checkpoint
eff_path = hf_hub_download(
    repo_id="Yasir-Alazmi/skin-lesion-ham10000",
    filename="efficientnet_b3_best.pth",
    local_dir="models/checkpoints/"
)

# ViT-B/16 checkpoint
vit_path = hf_hub_download(
    repo_id="Yasir-Alazmi/skin-lesion-ham10000",
    filename="vit_base_patch16_best.pth",
    local_dir="models/checkpoints/"
)
```

Or via CLI:
```bash
pip install huggingface-hub
huggingface-cli download Yasir-Alazmi/skin-lesion-ham10000 \
    --local-dir models/checkpoints/
```

---

## Checkpoint Format

Each `.pth` checkpoint contains:

```python
{
    "epoch": int,                   # epoch at which checkpoint was saved
    "model_state_dict": dict,       # model weights
    "optimizer_state_dict": dict,   # optimizer state
    "val_auc": float,               # validation macro AUC
    "val_loss": float,              # validation loss
    "config": dict,                 # training hyperparameters
}
```

### Loading a Checkpoint

```python
import torch
from src.models import build_vit
from src.config import DEVICE

model = build_vit(num_classes=7, pretrained=False)
ckpt = torch.load("models/checkpoints/vit_base_patch16_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']} | Val AUC: {ckpt['val_auc']:.4f}")
```

---

## Checkpoint Performance

| File | Model | Val AUC | Val Accuracy |
|------|-------|---------|-------------|
| `efficientnet_b3_best.pth` | EfficientNet-B3 | 0.95 | 92.6% |
| `vit_base_patch16_best.pth` | ViT-B/16 | **0.96** | **94.2%** |

---

## Training Your Own Checkpoints

```bash
# Train EfficientNet-B3 — saves to models/checkpoints/
python -m src.train --model efficientnet

# Train ViT-B/16
python -m src.train --model vit --epochs 30 --batch-size 16
```
