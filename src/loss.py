"""
loss.py
=======
Loss functions and sampling utilities for imbalanced classification.

Contents
--------
FocalLoss
    Multi-class focal loss (Lin et al., 2017) that down-weights well-classified
    examples so the model focuses on hard, minority-class samples.

make_weighted_sampler(labels)
    Builds a ``WeightedRandomSampler`` that over-samples minority classes so
    each mini-batch has a roughly balanced class distribution.

References
----------
Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
Focal Loss for Dense Object Detection. ICCV 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    FL(p_t) = -α_t * (1 − p_t)^γ * log(p_t)

    For C-class classification the loss is computed via log-softmax so that
    numerical precision is maintained.

    Parameters
    ----------
    alpha : Tensor or None
        Per-class weighting vector of shape (C,).  If ``None``, no class
        weighting is applied (uniform α = 1 for all classes).
    gamma : float
        Focusing parameter.  γ = 0 reduces to standard cross-entropy;
        larger values focus more on hard examples.
    reduction : {'mean', 'sum', 'none'}
        How to reduce the per-sample loss values.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor, shape (N, C)
            Raw (unnormalised) class scores from the model.
        targets : Tensor, shape (N,)
            Ground-truth class indices in [0, C).

        Returns
        -------
        Tensor
            Focal loss scalar (if reduction != 'none') or per-sample vector.
        """
        # Standard cross-entropy with log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)           # (N, C)
        probs     = torch.exp(log_probs)                    # (N, C)

        # Gather the log-probability and probability of the true class
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        p_t     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # (N,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(self.gamma)

        # Apply optional per-class α weighting
        if self.alpha is not None:
            alpha_t      = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        # Element-wise focal loss
        loss = -focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction='{self.reduction}'"


def build_focal_loss(
    class_counts: list[int] | None = None,
    gamma: float = 2.0,
    device: torch.device | None = None,
) -> FocalLoss:
    """
    Convenience factory that creates a ``FocalLoss`` with optional inverse-
    frequency class weights derived from ``class_counts``.

    Parameters
    ----------
    class_counts : list of int or None
        Number of samples per class.  If provided, per-class α is set to
        1 / count, normalised so the weights sum to the number of classes.
    gamma : float
        Focal loss γ parameter.
    device : torch.device or None
        Where to place the α buffer.

    Returns
    -------
    FocalLoss
    """
    alpha = None
    if class_counts is not None:
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)  # normalise
        alpha = weights

    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    if device is not None and alpha is not None:
        loss_fn = loss_fn.to(device)
    return loss_fn


# ──────────────────────────────────────────────────────────────────────────────
# Weighted sampler
# ──────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Create a ``WeightedRandomSampler`` that assigns each sample a weight
    inversely proportional to its class frequency.

    This ensures that every class is seen roughly equally often during
    training, without discarding any samples.

    Parameters
    ----------
    labels : list of int
        Integer class label for every sample in the training set (must be
        0-indexed and dense, i.e. range from 0 to C-1).

    Returns
    -------
    WeightedRandomSampler
        Suitable for passing directly to a ``DataLoader`` as ``sampler=``.
    """
    labels_array = np.array(labels)
    class_counts  = np.bincount(labels_array)

    # Weight per class: inverse of frequency
    class_weights = 1.0 / class_counts.astype(np.float64)

    # Assign weight to each sample
    sample_weights = class_weights[labels_array]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )
    return sampler
