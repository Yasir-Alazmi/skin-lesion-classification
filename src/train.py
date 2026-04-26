"""
train.py
========
Training loop with early stopping, learning-rate scheduling, and two-phase
fine-tuning (frozen backbone → full fine-tune) for the HAM10000 pipeline.

Key components
--------------
EarlyStopping
    Monitors a metric (default: val AUC) and halts training when no
    improvement is seen for `patience` consecutive epochs.

train_one_epoch(model, loader, optimizer, criterion, device)
    Single epoch of forward + backward with gradient clipping.

evaluate_epoch(model, loader, criterion, device)
    Computes loss and accuracy over a DataLoader (no grad).

train_model(model, train_loader, val_loader, config, …)
    Full training loop:
     * Phase 1 (epochs 0–UNFREEZE_EPOCH-1): only the head is trained.
     * Phase 2 (epoch UNFREEZE_EPOCH onward): backbone unfrozen at LR×0.1.
     * AdamW optimiser, ReduceLROnPlateau scheduler on val AUC.
     * Saves the best checkpoint to models/checkpoints/.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    GRAD_CLIP,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    UNFREEZE_EPOCH,
    WEIGHT_DECAY,
    EFFICIENTNET_CKPT,
    VIT_CKPT,
)
from src.evaluate import evaluate


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement to wait before stopping.
    min_delta : float
        Minimum change in the monitored metric that qualifies as improvement.
    mode : {'max', 'min'}
        Whether larger ('max') or smaller ('min') values of the metric are
        considered better.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode

        self.counter       = 0
        self.best_value    = float("-inf") if mode == "max" else float("inf")
        self.early_stop    = False

    def __call__(self, value: float) -> bool:
        """
        Update internal state with the latest metric value.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self) -> None:
        """Reset internal counter and best value."""
        self.counter       = 0
        self.best_value    = float("-inf") if self.mode == "max" else float("inf")
        self.early_stop    = False


# ──────────────────────────────────────────────────────────────────────────────
# Single-epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = DEVICE,
    grad_clip: float = GRAD_CLIP,
) -> dict[str, float]:
    """
    Run one full pass over the training DataLoader.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Loss function (FocalLoss or CrossEntropyLoss).
    device : torch.device
    grad_clip : float
        Maximum L2 norm for gradient clipping.

    Returns
    -------
    dict with keys 'loss' and 'accuracy'
    """
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients (especially for ViT)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        bs            = images.size(0)
        running_loss += loss.item() * bs
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += bs

        pbar.set_postfix(loss=f"{running_loss / total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """
    Compute loss and accuracy on a validation or test DataLoader.

    Returns
    -------
    dict with keys 'loss', 'accuracy', 'auc'
    """
    results = evaluate(model, loader, criterion, device)
    return {
        "loss":     results["loss"],
        "accuracy": results["accuracy"],
        "auc":      results["auc"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "model",
    num_epochs: int = 30,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = 7,
    checkpoint_path: str | None = None,
    device: torch.device = DEVICE,
    criterion: nn.Module | None = None,
) -> dict[str, list[float]]:
    """
    Train a model with two-phase fine-tuning and early stopping.

    Phase 1 (epochs 0 … UNFREEZE_EPOCH − 1)
        The backbone is frozen; only the classification head is trained.

    Phase 2 (epochs UNFREEZE_EPOCH … num_epochs − 1)
        The backbone is unfrozen.  The backbone's learning rate is set to
        ``lr × 0.1`` to protect the pre-trained features; the head keeps
        the original ``lr``.

    Parameters
    ----------
    model : nn.Module
        Must expose ``freeze_backbone()`` and ``unfreeze_backbone()`` methods.
    train_loader, val_loader : DataLoader
    model_name : str
        Used for progress messages and the default checkpoint filename.
    num_epochs : int
    lr : float
        Initial head learning rate.
    weight_decay : float
    patience : int
        EarlyStopping patience.
    checkpoint_path : str or None
        Where to save the best model weights (.pth).  Defaults to the path
        defined in ``config.py`` for the model name.
    device : torch.device
    criterion : nn.Module or None
        Loss function.  Defaults to ``FocalLoss(gamma=2.0)``.

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc',
              'lr_history'.  Each value is a list of per-epoch floats.
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    if criterion is None:
        from src.loss import FocalLoss
        criterion = FocalLoss(gamma=2.0).to(device)

    if checkpoint_path is None:
        if "efficientnet" in model_name.lower():
            checkpoint_path = EFFICIENTNET_CKPT
        else:
            checkpoint_path = VIT_CKPT

    model = model.to(device)

    # Phase 1: freeze backbone, train only the head
    model.freeze_backbone()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler   = ReduceLROnPlateau(
        optimizer,
        mode="max",       # maximise val AUC
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )
    early_stop  = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
        "val_auc":    [],
        "lr_history": [],
    }

    best_val_auc   = float("-inf")
    best_weights   = copy.deepcopy(model.state_dict())
    phase_switched = False

    print(f"\n{'='*60}")
    print(f"  Training: {model_name} on {device}")
    print(f"  Epochs: {num_epochs} | Batch size: {train_loader.batch_size}")
    print(f"  Backbone unfreezes at epoch {UNFREEZE_EPOCH}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ── Phase 2 transition ───────────────────────────────────────────────
        if epoch == UNFREEZE_EPOCH and not phase_switched:
            print(f"\n[Epoch {epoch+1}] Unfreezing backbone — switching to full fine-tune")
            model.unfreeze_backbone()

            # Rebuild optimiser: backbone gets lr×0.1, head keeps lr
            head_params     = list(model.head.parameters())
            backbone_params = [
                p for p in model.backbone.parameters() if p.requires_grad
            ]
            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": lr * 0.1},
                    {"params": head_params,     "lr": lr},
                ],
                weight_decay=weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=LR_FACTOR,
                patience=LR_PATIENCE,
                min_lr=LR_MIN,
            )
            phase_switched = True

        # ── Train one epoch ──────────────────────────────────────────────────
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate_epoch(model, val_loader, criterion, device)

        val_auc  = val_metrics["auc"]
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[-1]["lr"]  # head LR

        # ── Record history ───────────────────────────────────────────────────
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_auc)
        history["lr_history"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} || "
            f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
            f"AUC: {val_auc:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # ── Save best checkpoint ─────────────────────────────────────────────
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_weights,
                    "val_auc": best_val_auc,
                    "val_acc": val_metrics["accuracy"],
                },
                checkpoint_path,
            )
            print(f"  ✔ New best AUC {best_val_auc:.4f} — checkpoint saved.")

        # ── Early stopping ───────────────────────────────────────────────────
        if early_stop(val_auc):
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"\nTraining complete.  Best val AUC: {best_val_auc:.4f}")
    return history
