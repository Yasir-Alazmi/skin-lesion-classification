# Models Directory

Trained checkpoints are **not committed to Git** (a ViT-B/16 checkpoint is about 344 MB).

## Where the weights live

- Training writes the best-by-validation-macro-F1 checkpoint to `best_vit_b16.pth` / `best_efficientnet_b3.pth`
  (under `<SKIN_DATA_DIR>/checkpoints/` when that variable is set, otherwise `models/checkpoints/`).
- The demo (`app.py`) loads ViT-B/16 weights from `MODEL_PATH`, else from the Hugging Face model repo
  `Yasir-Alazmi/skin-lesion-vit-b16`, else from the local checkpoint above.
- To publish weights: `python scripts/export_weights.py best_vit_b16.pth` and upload the output to that repo.

Test-set results for these checkpoints (lesion-grouped split) are in [`../results/metrics.md`](../results/metrics.md).

## Checkpoint format

```python
{"epoch": int, "model_state_dict": dict, "val_f1": float, ...}   # a bare state_dict also loads
```

### Loading

```python
from src.inference import load_model
model = load_model("best_vit_b16.pth")   # ViT-B/16, CPU, eval mode
```

## Training

See "Reproduce" in the [README](../README.md): `scripts/train_all.ps1` trains both models on the committed split.
