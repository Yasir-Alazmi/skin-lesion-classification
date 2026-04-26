"""
evaluate.py
===========
Evaluation utilities for the HAM10000 classification models.

Functions
---------
evaluate(model, loader, criterion, device)
    Runs inference over a DataLoader and returns loss, accuracy, macro AUC,
    raw probabilities, predictions, and ground-truth labels.

full_evaluation(model, loader, criterion, class_names, device)
    Extends ``evaluate()`` with per-class F1, sensitivity, specificity, and
    the full confusion matrix.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from src.config import DEVICE, NUM_CLASSES, CLASS_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Core inference pass
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> dict:
    """
    Run inference over an entire DataLoader without computing gradients.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
        Loss function used only for computing the scalar loss value.
    device : torch.device

    Returns
    -------
    dict with keys:
        'loss'       – float, mean loss over all samples
        'accuracy'   – float, overall accuracy
        'auc'        – float, macro one-vs-rest AUC
        'probs'      – np.ndarray (N, C), softmax probabilities
        'preds'      – np.ndarray (N,), predicted class indices
        'labels'     – np.ndarray (N,), ground-truth class indices
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    running_loss = 0.0
    total        = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            logits  = model(images)
            loss    = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            total        += images.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Concatenate
    logits_cat = torch.cat(all_logits, dim=0)         # (N, C)
    labels_cat = torch.cat(all_labels, dim=0).numpy() # (N,)

    probs = torch.softmax(logits_cat, dim=-1).numpy()  # (N, C)
    preds = logits_cat.argmax(dim=-1).numpy()           # (N,)

    avg_loss = running_loss / total
    accuracy = accuracy_score(labels_cat, preds)

    # Macro one-vs-rest AUC (requires binarised labels)
    labels_bin = label_binarize(labels_cat, classes=list(range(NUM_CLASSES)))
    auc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")

    return {
        "loss":     avg_loss,
        "accuracy": accuracy,
        "auc":      auc,
        "probs":    probs,
        "preds":    preds,
        "labels":   labels_cat,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation (per-class metrics + confusion matrix)
# ──────────────────────────────────────────────────────────────────────────────

def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str] = CLASS_NAMES,
    device: torch.device = DEVICE,
) -> dict:
    """
    Extended evaluation that adds per-class sensitivity, specificity, and F1.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
    class_names : list of str
        Human-readable label names in class-index order.
    device : torch.device

    Returns
    -------
    dict with all keys from ``evaluate()`` plus:
        'f1_macro'         – float, macro F1
        'f1_per_class'     – np.ndarray (C,), per-class F1
        'sensitivity'      – np.ndarray (C,), recall per class
        'specificity'      – np.ndarray (C,), specificity per class
        'confusion_matrix' – np.ndarray (C, C)
        'per_class_auc'    – np.ndarray (C,), one-vs-rest AUC per class
        'class_names'      – list of str
    """
    base = evaluate(model, loader, criterion, device)
    preds  = base["preds"]
    labels = base["labels"]
    probs  = base["probs"]

    num_classes = len(class_names)

    # ── Macro F1 ──────────────────────────────────────────────────────────────
    f1_macro    = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # ── Per-class sensitivity (TPR) and specificity (TNR) ────────────────────
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp                 # actual positives missed
        fp = cm[:, c].sum() - tp                 # false positives
        tn = cm.sum() - tp - fn - fp             # true negatives

        sensitivity[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── Per-class AUC ─────────────────────────────────────────────────────────
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    per_class_auc = np.zeros(num_classes)
    for c in range(num_classes):
        try:
            per_class_auc[c] = roc_auc_score(labels_bin[:, c], probs[:, c])
        except ValueError:
            per_class_auc[c] = float("nan")

    # ── Summary table print ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Class':<22} {'AUC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8}")
    print("-"*70)
    for i, name in enumerate(class_names):
        print(
            f"{name:<22} "
            f"{per_class_auc[i]:>8.4f} "
            f"{f1_per_class[i]:>8.4f} "
            f"{sensitivity[i]:>8.4f} "
            f"{specificity[i]:>8.4f}"
        )
    print("-"*70)
    print(
        f"{'MACRO / OVERALL':<22} "
        f"{base['auc']:>8.4f} "
        f"{f1_macro:>8.4f} "
        f"{sensitivity.mean():>8.4f} "
        f"{specificity.mean():>8.4f}"
    )
    print("="*70)
    print(f"Overall Accuracy : {base['accuracy']:.4f}")
    print(f"Overall Loss     : {base['loss']:.4f}")

    return {
        **base,
        "f1_macro":        f1_macro,
        "f1_per_class":    f1_per_class,
        "sensitivity":     sensitivity,
        "specificity":     specificity,
        "confusion_matrix": cm,
        "per_class_auc":   per_class_auc,
        "class_names":     class_names,
    }
