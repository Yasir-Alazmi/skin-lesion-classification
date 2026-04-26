"""
visualize.py
============
Plotting utilities for the HAM10000 skin-lesion classification project.

All functions save figures to the ``outputs/`` directory defined in
``config.py`` and also return the Matplotlib ``Figure`` object so callers
can display or embed them in notebooks.

Functions
---------
plot_training_history(eff_hist, vit_hist)
    Side-by-side loss/accuracy/AUC curves for both models.

plot_confusion_matrix(cm, class_names, model_name)
    Normalised heatmap of the confusion matrix.

plot_roc_curves(results, class_names, model_name)
    Per-class and macro-average ROC curves.

plot_sens_spec(results, class_names, model_name)
    Grouped bar chart of per-class sensitivity and specificity.

plot_model_comparison(eff_results, vit_results)
    Radar / grouped-bar comparison of the two models across all metrics.

show_predictions(model, dataset, class_names, device, n)
    Grid of n sample images with true and predicted labels.
"""

from __future__ import annotations

import os
import math
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch
from torch.utils.data import Dataset

from src.config import OUTPUTS_DIR, NUM_CLASSES, CLASS_NAMES

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = {
    "efficientnet": "#4C72B0",
    "vit":          "#DD8452",
    "correct":      "#2ECC71",
    "wrong":        "#E74C3C",
}

def _save(fig: plt.Figure, filename: str) -> str:
    """Save a figure to OUTPUTS_DIR and return the full path."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def _style() -> None:
    """Apply a consistent seaborn style to all plots."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Training history
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    eff_hist: dict[str, list[float]],
    vit_hist: dict[str, list[float]],
    save: bool = True,
) -> plt.Figure:
    """
    Plot training / validation loss, accuracy, and AUC for both models.

    Parameters
    ----------
    eff_hist : dict
        History dict returned by ``train_model()`` for EfficientNet-B3.
    vit_hist : dict
        History dict returned by ``train_model()`` for ViT-B/16.
    save : bool
        Whether to save the figure to disk.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History — EfficientNet-B3 vs ViT-B/16", fontsize=14, fontweight="bold")

    metrics = [
        ("loss",    "Loss",     "train_loss", "val_loss"),
        ("acc",     "Accuracy", "train_acc",  "val_acc"),
        ("auc",     "Val AUC",  None,         "val_auc"),
    ]

    for ax, (_, ylabel, train_key, val_key) in zip(axes, metrics):
        for hist, label, color in [
            (eff_hist, "EfficientNet-B3", _PALETTE["efficientnet"]),
            (vit_hist, "ViT-B/16",        _PALETTE["vit"]),
        ]:
            epochs = range(1, len(hist[val_key]) + 1)
            ax.plot(epochs, hist[val_key], color=color, label=f"{label} val", linewidth=2)
            if train_key and train_key in hist:
                ax.plot(
                    epochs, hist[train_key],
                    color=color, linestyle="--", alpha=0.55,
                    label=f"{label} train",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        _save(fig, "training_history.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """
    Plot a normalised confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray, shape (C, C)
        Confusion matrix (counts).
    class_names : list of str
    model_name : str
    save : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    # Overlay raw counts in a smaller font
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j + 0.5, i + 0.72,
                f"({cm[i, j]})",
                ha="center", va="center",
                fontsize=7, color="grey",
            )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        _save(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. ROC curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: dict,
    class_names: list[str] = CLASS_NAMES,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """
    Plot per-class and macro-average ROC curves.

    Parameters
    ----------
    results : dict
        Output of ``full_evaluation()`` containing 'probs' and 'labels'.
    class_names : list of str
    model_name : str
    save : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()
    probs  = results["probs"]   # (N, C)
    labels = results["labels"]  # (N,)
    n_cls  = len(class_names)

    labels_bin = label_binarize(labels, classes=list(range(n_cls)))

    # Compute ROC for each class
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for c in range(n_cls):
        fpr_dict[c], tpr_dict[c], _ = roc_curve(labels_bin[:, c], probs[:, c])
        roc_auc_dict[c] = auc(fpr_dict[c], tpr_dict[c])

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(n_cls)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_cls):
        mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
    mean_tpr /= n_cls
    macro_auc = auc(all_fpr, mean_tpr)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))

    for c in range(n_cls):
        ax.plot(
            fpr_dict[c], tpr_dict[c],
            color=cmap(c),
            lw=1.5,
            label=f"{class_names[c]} (AUC={roc_auc_dict[c]:.3f})",
        )

    ax.plot(
        all_fpr, mean_tpr,
        color="black", lw=2.5, linestyle="--",
        label=f"Macro Avg (AUC={macro_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if save:
        _save(fig, f"roc_curves_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sensitivity / Specificity bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_sens_spec(
    results: dict,
    class_names: list[str] = CLASS_NAMES,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart of per-class sensitivity and specificity.

    Parameters
    ----------
    results : dict
        Output of ``full_evaluation()``.
    class_names : list of str
    model_name : str
    save : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()
    sensitivity = results["sensitivity"]
    specificity = results["specificity"]
    n_cls = len(class_names)

    x     = np.arange(n_cls)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, sensitivity, width, label="Sensitivity (TPR)",
                   color="#3498DB", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, specificity, width, label="Specificity (TNR)",
                   color="#E67E22", alpha=0.85, edgecolor="white")

    # Annotate bars
    for bar in itertools.chain(bars1, bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Per-Class Sensitivity & Specificity — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save:
        _save(fig, f"sens_spec_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Model comparison (grouped bar)
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    eff_results: dict,
    vit_results: dict,
    save: bool = True,
) -> plt.Figure:
    """
    Head-to-head grouped bar comparison of key metrics.

    Parameters
    ----------
    eff_results : dict
        Output of ``full_evaluation()`` for EfficientNet-B3.
    vit_results : dict
        Output of ``full_evaluation()`` for ViT-B/16.
    save : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()
    metric_labels = ["Accuracy", "Macro F1", "Macro AUC",
                     "Mean Sens.", "Mean Spec."]
    eff_vals = [
        eff_results["accuracy"],
        eff_results["f1_macro"],
        eff_results["auc"],
        eff_results["sensitivity"].mean(),
        eff_results["specificity"].mean(),
    ]
    vit_vals = [
        vit_results["accuracy"],
        vit_results["f1_macro"],
        vit_results["auc"],
        vit_results["sensitivity"].mean(),
        vit_results["specificity"].mean(),
    ]

    x     = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width / 2, eff_vals, width,
                   label="EfficientNet-B3", color=_PALETTE["efficientnet"],
                   alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, vit_vals, width,
                   label="ViT-B/16", color=_PALETTE["vit"],
                   alpha=0.85, edgecolor="white")

    for bar in itertools.chain(bars1, bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — EfficientNet-B3 vs ViT-B/16",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save:
        _save(fig, "model_comparison.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Sample predictions grid
# ──────────────────────────────────────────────────────────────────────────────

def show_predictions(
    model: torch.nn.Module,
    dataset: Dataset,
    class_names: list[str] = CLASS_NAMES,
    device: torch.device | None = None,
    n: int = 12,
    save: bool = True,
) -> plt.Figure:
    """
    Display a grid of n images with their true and predicted labels.

    Parameters
    ----------
    model : nn.Module
    dataset : Dataset
        A ``HAMDataset`` instance (or any map-style dataset returning
        (tensor, int) pairs).
    class_names : list of str
    device : torch.device or None
    n : int
        Number of images to show.
    save : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    from src.config import DEVICE as _DEV
    if device is None:
        device = _DEV

    _style()
    model.eval()

    # Collect n samples
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    # ImageNet de-normalisation for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            tensor, true_label = dataset[idx]
            inp = tensor.unsqueeze(0).to(device)
            logit = model(inp)
            pred_label = logit.argmax(dim=1).item()

            # De-normalise for visualisation
            img_np = tensor.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np * std + mean, 0, 1)

            color = _PALETTE["correct"] if pred_label == true_label else _PALETTE["wrong"]
            ax.imshow(img_np)
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
            ax.axis("off")

    # Hide any unused axes
    for ax in axes[len(indices):]:
        ax.set_visible(False)

    plt.suptitle("Sample Predictions (green=correct, red=wrong)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        _save(fig, "sample_predictions.png")
    return fig
