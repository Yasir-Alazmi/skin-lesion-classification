"""Basic smoke tests — no GPU or dataset required."""
import importlib
import sys
import types


def test_src_imports():
    """All src modules must be importable (CPU-only, no data)."""
    modules = [
        "src",
        "src.config",
        "src.loss",
        "src.models",
        "src.preprocess",
        "src.visualize",
    ]
    for mod in modules:
        imported = importlib.import_module(mod)
        assert isinstance(imported, types.ModuleType), f"Failed to import {mod}"


def test_class_names_length():
    from src.config import CLASS_NAMES
    assert len(CLASS_NAMES) == 7, "Expected 7 HAM10000 classes"


def test_focal_loss_forward():
    """FocalLoss forward pass with dummy tensors."""
    import torch
    from src.loss import FocalLoss

    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 7)
    targets = torch.randint(0, 7, (8,))
    loss = criterion(logits, targets)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_model_output_shape_efficientnet():
    """EfficientNet-B3 should output (batch, 7) without GPU."""
    import torch
    from src.models import build_efficientnet

    model = build_efficientnet(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 7), f"Unexpected output shape: {out.shape}"


def test_model_output_shape_vit():
    """ViT-B/16 should output (batch, 7) without GPU."""
    import torch
    from src.models import build_vit

    model = build_vit(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 7), f"Unexpected output shape: {out.shape}"
