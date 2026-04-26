"""
models.py
=========
Model factory functions for EfficientNet-B3 and ViT-B/16.

Both builders use the ``timm`` library to load ImageNet pre-trained backbones
and replace the classification head with custom multi-layer heads tuned for
the 7-class HAM10000 task.

Functions
---------
build_efficientnet(num_classes, pretrained)
    EfficientNet-B3 with a Dropout→Linear→BN→ReLU→Dropout→Linear head.

build_vit(num_classes, pretrained)
    ViT-B/16 with a LayerNorm→Dropout→Linear→GELU→Dropout→Linear head.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# EfficientNet-B3
# ──────────────────────────────────────────────────────────────────────────────

class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 backbone with a custom classification head.

    Architecture
    ------------
    EfficientNet-B3 backbone (features, 1536-d global average pool)
      └─ Dropout(0.4)
         └─ Linear(1536 → 256)
            └─ BatchNorm1d(256)
               └─ ReLU()
                  └─ Dropout(0.3)
                     └─ Linear(256 → num_classes)

    Parameters
    ----------
    num_classes : int
        Output dimensionality (7 for HAM10000).
    pretrained : bool
        Load ImageNet-1k pre-trained weights from timm.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        super().__init__()

        # Load backbone (drop the original classifier head)
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,       # Remove head → output is feature vector
            global_pool="avg",   # Global average pooling
        )

        in_features = self.backbone.num_features  # 1536 for EfficientNet-B3

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (N, 1536)
        logits   = self.head(features) # (N, num_classes)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (head remains trainable)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_efficientnet(
    num_classes: int = 7,
    pretrained: bool = True,
) -> EfficientNetB3Classifier:
    """
    Factory function for the EfficientNet-B3 classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        If ``True``, initialises the backbone with ImageNet weights.

    Returns
    -------
    EfficientNetB3Classifier
    """
    model = EfficientNetB3Classifier(num_classes=num_classes, pretrained=pretrained)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Vision Transformer — ViT-B/16
# ──────────────────────────────────────────────────────────────────────────────

class ViTB16Classifier(nn.Module):
    """
    ViT-B/16 backbone with a custom classification head.

    Architecture
    ------------
    ViT-B/16 backbone (patch size 16, 768-d CLS token)
      └─ LayerNorm(768)
         └─ Dropout(0.4)
            └─ Linear(768 → 256)
               └─ GELU()
                  └─ Dropout(0.3)
                     └─ Linear(256 → num_classes)

    Parameters
    ----------
    num_classes : int
        Output dimensionality (7 for HAM10000).
    pretrained : bool
        Load ImageNet-21k → ImageNet-1k fine-tuned weights from timm.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        super().__init__()

        # Load ViT backbone (drop the original head)
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,    # Remove original head
        )

        in_features = self.backbone.num_features  # 768 for ViT-B/16

        # Custom classification head (no BatchNorm — ViT uses LayerNorm)
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (N, 768) — CLS token output
        logits   = self.head(features) # (N, num_classes)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (head remains trainable)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_vit(
    num_classes: int = 7,
    pretrained: bool = True,
) -> ViTB16Classifier:
    """
    Factory function for the ViT-B/16 classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        If ``True``, initialises the backbone with pre-trained weights from timm
        (ImageNet-21k → ImageNet-1k).

    Returns
    -------
    ViTB16Classifier
    """
    model = ViTB16Classifier(num_classes=num_classes, pretrained=pretrained)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts for a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Quick sanity check
    for name, build_fn in [("EfficientNet-B3", build_efficientnet), ("ViT-B/16", build_vit)]:
        model = build_fn(num_classes=7, pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        params = count_parameters(model)
        print(
            f"{name}: output shape {out.shape} | "
            f"total params {params['total']:,} | "
            f"trainable {params['trainable']:,}"
        )
