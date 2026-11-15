"""
train.py
========
Training loop with early stopping, learning-rate scheduling, and two-phase
fine-tuning (frozen backbone → full fine-tune) for the HAM10000 pipeline.

Key components
--------------
EarlyStopping
    Monitors a metric (val macro-F1 in ``train_model``) and halts training when no
    improvement is seen for `patience` consecutive epochs.

train_one_epoch(model, loader, optimizer, criterion, device)
    Single epoch of forward + backward with gradient clipping.

evaluate_epoch(model, loader, criterion, device)
    Computes loss and accuracy over a DataLoader (no grad).

train_model(model, train_loader, val_loader, config, …)
    Full training loop:
     * Phase 1 (epochs 0–UNFREEZE_EPOCH-1): only the head is trained.
     * Phase 2 (epoch UNFREEZE_EPOCH onward): backbone unfrozen at LR×0.1.
     * AdamW optimiser, mixed precision (fp16 autocast + GradScaler) on CUDA.
     * ReduceLROnPlateau, early stopping and checkpointing follow val macro-F1.
     * The test set is never touched here; see ``final_test_evaluation``.
     * Best checkpoint goes to CHECKPOINTS_DIR (outside git when SKIN_DATA_DIR is set).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    EFFICIENTNET_CKPT,
    GRAD_CLIP,
    LOGS_DIR,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    UNFREEZE_EPOCH,
    VIT_CKPT,
    WEIGHT_DECAY,
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
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.early_stop = False

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
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self) -> None:
        """Reset internal counter and best value."""
        self.counter = 0
        self.best_value = float("-inf") if self.mode == "max" else float("inf")
        self.early_stop = False


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
    scaler: torch.amp.GradScaler | None = None,
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
    scaler : GradScaler or None
        When given and enabled, the forward pass runs under fp16 autocast and the
        loss is scaled (mixed precision).  ``None`` trains in plain fp32.

    Returns
    -------
    dict with keys 'loss' and 'accuracy'
    """
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False, disable=not sys.stderr.isatty())
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Gradient clipping prevents exploding gradients (especially for ViT)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs

        pbar.set_postfix(loss=f"{running_loss / total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """
    Compute loss, accuracy, macro-F1 and AUC on a validation DataLoader.

    Returns
    -------
    dict with keys 'loss', 'accuracy', 'f1_macro', 'auc'
    """
    results = evaluate(model, loader, criterion, device)
    return {
        "loss": results["loss"],
        "accuracy": results["accuracy"],
        "f1_macro": results["f1_macro"],
        "auc": results["auc"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────────────────────────────────────


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _full_finetune_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    """Backbone gets lr x 0.1, head keeps lr."""
    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def write_status(path: str | None, **fields) -> None:
    """Merge ``fields`` into the JSON status file (atomic), stamping ``updated``."""
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    cur.update(fields)
    cur["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def last_checkpoint_path(checkpoint_path: str) -> str:
    """``best_<tag>.pth`` -> ``last_<tag>.pth`` (same directory)."""
    d, name = os.path.split(checkpoint_path)
    return os.path.join(d, name.replace("best_", "last_", 1) if "best_" in name else "last_" + name)


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
    unfreeze_epoch: int = UNFREEZE_EPOCH,
    use_amp: bool | None = None,
    log_path: str | None = None,
    resume: bool = False,
    status_path: str | None = None,
) -> dict:
    """
    Train a model with two-phase fine-tuning and early stopping.

    Model selection (best checkpoint, LR schedule, early stopping) uses the
    validation macro-F1.  The test set is not an argument: it cannot influence
    training.

    Phase 1 (epochs 0 … unfreeze_epoch − 1)
        The backbone is frozen; only the classification head is trained.

    Phase 2 (epochs unfreeze_epoch … num_epochs − 1)
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
        EarlyStopping patience (in epochs, measured on val macro-F1).
    checkpoint_path : str or None
        Where to save the best model weights (.pth).  Defaults to the path
        defined in ``config.py`` for the model name.
    device : torch.device
    criterion : nn.Module or None
        Loss function.  Defaults to ``FocalLoss(gamma=2.0)``.
    unfreeze_epoch : int
        Epoch index at which the backbone is unfrozen (0 = no frozen phase).
    use_amp : bool or None
        Mixed precision; ``None`` enables it on CUDA only.
    log_path : str or None
        If given, the per-epoch history is rewritten there as JSON after each epoch.
    resume : bool
        Continue from ``last_<tag>.pth`` (next to the best checkpoint) when it exists:
        model, optimizer, scaler, scheduler, early-stopping state, history and RNG
        state are restored, so the run continues exactly as if never interrupted.
        That file is written (atomically) after every epoch regardless of this flag.
    status_path : str or None
        If given, ``status.json`` is updated after every epoch (see ``write_status``).

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_auc',
              'lr_history', 'epoch_seconds', 'best_epoch'.
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

    if use_amp is None:
        use_amp = device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):  # torch >= 2.3
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    else:  # torch 2.1 / 2.2 (the version pinned in pyproject.toml and CI)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = model.to(device)

    def _new_scheduler(opt: torch.optim.Optimizer) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            opt,
            mode="max",  # maximise val macro-F1
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LR_MIN,
        )

    phase_switched = unfreeze_epoch <= 0
    if phase_switched:
        model.unfreeze_backbone()
        optimizer = _full_finetune_optimizer(model, lr, weight_decay)
    else:
        # Phase 1: freeze backbone, train only the head
        model.freeze_backbone()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    scheduler = _new_scheduler(optimizer)
    early_stop = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")

    history: dict = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
        "lr_history": [],
        "epoch_seconds": [],
        "best_epoch": 0,
    }

    best_val_f1 = float("-inf")
    best_weights = _cpu_state(model)
    start_epoch = 0
    last_path = last_checkpoint_path(checkpoint_path)

    if resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        start_epoch = int(st["epoch"])
        # Optimizer layout at save time: frozen head-only until the epoch loop switches.
        if unfreeze_epoch <= 0 or start_epoch > unfreeze_epoch:
            model.unfreeze_backbone()
            optimizer = _full_finetune_optimizer(model, lr, weight_decay)
            scheduler = _new_scheduler(optimizer)
            phase_switched = True
        model.load_state_dict(st["model_state_dict"])
        optimizer.load_state_dict(st["optimizer_state_dict"])
        scheduler.load_state_dict(st["scheduler_state_dict"])
        scaler.load_state_dict(st["scaler_state_dict"])
        early_stop.__dict__.update(st["early_stop"])
        history = st["history"]
        best_val_f1 = st["best_val_f1"]
        best_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)[
            "model_state_dict"
        ]
        _set_rng_state(st["rng"])
        print(f"Resumed from {last_path} after epoch {start_epoch}.", flush=True)

    print(f"\n{'='*60}")
    print(f"  Training: {model_name} on {device} (AMP={use_amp})")
    print(f"  Epochs: {num_epochs} | Batch size: {train_loader.batch_size}")
    print(f"  Backbone unfreezes at epoch {unfreeze_epoch} | selection: val macro-F1")
    print(f"{'='*60}", flush=True)

    # A run that already early-stopped before the interruption has nothing left to train.
    for epoch in range(start_epoch, 0 if early_stop.early_stop else num_epochs):
        epoch_start = time.time()

        # ── Phase 2 transition ───────────────────────────────────────────────
        if epoch == unfreeze_epoch and not phase_switched:
            print(f"\n[Epoch {epoch+1}] Unfreezing backbone - switching to full fine-tune")
            model.unfreeze_backbone()
            optimizer = _full_finetune_optimizer(model, lr, weight_decay)
            scheduler = _new_scheduler(optimizer)
            phase_switched = True

        # ── Train one epoch ──────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)

        val_f1 = val_metrics["f1_macro"]
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[-1]["lr"]  # head LR

        # ── Record history ───────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_metrics["auc"])
        history["lr_history"].append(current_lr)
        history["epoch_seconds"].append(epoch_time)

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} || "
            f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_f1:.4f} | AUC: {val_metrics['auc']:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            + (
                f" | GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                if device.type == "cuda"
                else ""
            ),
            flush=True,
        )

        # ── Save best checkpoint (by val macro-F1) ───────────────────────────
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_weights = _cpu_state(model)
            history["best_epoch"] = epoch + 1
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_name": model_name,
                    "model_state_dict": best_weights,
                    "val_f1": best_val_f1,
                    "val_auc": val_metrics["auc"],
                    "val_acc": val_metrics["accuracy"],
                },
                checkpoint_path,
            )
            print(f"  New best val macro-F1 {best_val_f1:.4f} - checkpoint saved.", flush=True)

        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(json.dumps(history, indent=2))

        # ── Early stopping ───────────────────────────────────────────────────
        stop = early_stop(val_f1)

        # Resume point (atomic, so a kill mid-write never corrupts it)
        tmp_last = last_path + ".tmp"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "early_stop": dict(early_stop.__dict__),
                "history": history,
                "best_val_f1": best_val_f1,
                "rng": _rng_state(),
            },
            tmp_last,
        )
        os.replace(tmp_last, last_path)
        write_status(
            status_path,
            model=model_name,
            phase="training",
            epoch=epoch + 1,
            num_epochs=num_epochs,
            val_f1=val_f1,
            best_val_f1=best_val_f1,
            best_epoch=history["best_epoch"],
            epoch_seconds=epoch_time,
            done=False,
        )
        if stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.", flush=True)
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"\nTraining complete.  Best val macro-F1: {best_val_f1:.4f}")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Final (one-shot) test evaluation
# ──────────────────────────────────────────────────────────────────────────────


def final_test_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: str,
    out_dir: str,
    model_title: str,
    device: torch.device = DEVICE,
) -> dict:
    """
    Evaluate the best-by-validation checkpoint on the test set, once.

    Writes ``test_predictions.csv`` (image_id, label, pred, p_<class>...) and
    ``test_metrics.json`` into ``out_dir``.
    """
    import numpy as np
    import pandas as pd

    from src.config import CLASS_NAMES, IDX_TO_LABEL
    from src.evaluate import full_evaluation
    from src.loss import FocalLoss

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    res = full_evaluation(model, test_loader, FocalLoss(gamma=2.0).to(device), device=device)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "image_id": test_loader.dataset.df["image_id"].to_numpy(),
            "label": [IDX_TO_LABEL[i] for i in res["labels"]],
            "pred": [IDX_TO_LABEL[i] for i in res["preds"]],
        }
    )
    for c, name in IDX_TO_LABEL.items():
        frame[f"p_{name}"] = res["probs"][:, c]
    frame.to_csv(out / "test_predictions.csv", index=False)

    metrics = {
        "model": model_title,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "val_f1_at_checkpoint": float(ckpt["val_f1"]),
        "n_test": int(len(frame)),
        "accuracy": float(res["accuracy"]),
        "f1_macro": float(res["f1_macro"]),
        "auc_macro": float(res["auc"]),
        "per_class": {
            name: {
                "f1": float(res["f1_per_class"][i]),
                "sensitivity": float(res["sensitivity"][i]),
                "specificity": float(res["specificity"][i]),
                "auc": float(res["per_class_auc"][i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": np.asarray(res["confusion_matrix"]).tolist(),
    }
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _run_one(args, model_key: str, train_loader, val_loader, test_loader) -> None:
    """Train one architecture, then evaluate its best-by-validation checkpoint once."""
    from src.config import CHECKPOINTS_DIR, RUNS_DIR
    from src.models import build_efficientnet, build_vit

    model = (
        build_vit(pretrained=True) if model_key == "vit" else build_efficientnet(pretrained=True)
    )
    model_title = "ViT-B/16" if model_key == "vit" else "EfficientNet-B3"
    tag = "vit_b16" if model_key == "vit" else "efficientnet_b3"
    if args.smoke:
        tag += "_smoke"
    run_dir = args.out_dir or os.path.join(RUNS_DIR, tag)
    if args.out_dir and args.model == "both":
        run_dir = os.path.join(args.out_dir, tag)
    ckpt = os.path.join(CHECKPOINTS_DIR, f"best_{tag}.pth")
    os.makedirs(LOGS_DIR, exist_ok=True)
    status = args.status_file
    if (
        args.resume
        and not args.smoke
        and os.path.exists(os.path.join(run_dir, "test_metrics.json"))
    ):
        print(f"{model_title}: already trained and tested - skipping.", flush=True)
        return
    write_status(status, model=model_title, phase="starting", epoch=0, done=False)

    history = train_model(
        model,
        train_loader,
        val_loader,
        model_name=model_title,
        num_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=ckpt,
        unfreeze_epoch=args.unfreeze_epoch,
        use_amp=False if args.no_amp else None,
        log_path=os.path.join(run_dir, "history.json"),
        resume=args.resume,
        status_path=status,
    )
    if args.smoke:
        mean_t = sum(history["epoch_seconds"]) / len(history["epoch_seconds"])
        n_train = len(train_loader.dataset)
        print(f"[smoke] {n_train} train imgs, {mean_t:.1f}s/epoch")
        return

    # The test set is touched exactly once, after training has finished.
    write_status(status, model=model_title, phase="testing", done=False)
    metrics = final_test_evaluation(model, test_loader, ckpt, run_dir, model_title)
    print(
        f"\nTEST ({model_title}): acc={metrics['accuracy']:.4f} "
        f"macro-F1={metrics['f1_macro']:.4f} macro-AUC={metrics['auc_macro']:.4f}",
        flush=True,
    )
    write_status(
        status,
        model=model_title,
        phase="tested",
        test_f1_macro=metrics["f1_macro"],
        test_accuracy=metrics["accuracy"],
        done=False,
    )


def main() -> None:
    """CLI entrypoint for training EfficientNet-B3 and/or ViT-B/16 on HAM10000."""
    import argparse

    import pandas as pd

    from src.cache import build_cache
    from src.config import (
        BATCH_SIZE,
        CACHE_DIR,
        HAM_IMAGES_DIR_1,
        HAM_IMAGES_DIR_2,
        HAM_METADATA_CSV,
        NUM_EPOCHS,
        SEED,
        SPLITS_CSV,
        set_seed,
    )
    from src.dataset import build_loaders
    from src.models import build_efficientnet, build_vit

    parser = argparse.ArgumentParser(description="Train HAM10000 Skin Lesion Classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "efficientnet", "both"],
        help="Model architecture to train ('both' = EfficientNet-B3 then ViT-B/16)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early-stopping patience")
    parser.add_argument("--unfreeze-epoch", type=int, default=UNFREEZE_EPOCH)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Cache-building processes, and DataLoader workers with --loader workers",
    )
    parser.add_argument(
        "--loader",
        choices=["auto", "gpu", "workers"],
        default="auto",
        help="gpu = batches from RAM with on-device augmentation (default when the cache "
        "exists); workers = classic multi-process DataLoader",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--resume", action="store_true", help="Continue from last_<model>.pth if present"
    )
    parser.add_argument(
        "--status-file", type=str, default=None, help="JSON progress file updated every epoch"
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Run output directory")
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        metavar="N",
        help="Debug: train on a class-balanced subset of ~N images (no test evaluation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify CLI and model instantiation without requiring the full HAM10000 dataset",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.dry_run:
        m = (
            build_vit(pretrained=False)
            if args.model == "vit"
            else build_efficientnet(pretrained=False)
        )
        n_params = sum(p.numel() for p in m.parameters())
        print(f"[Dry-Run OK] Initialized {args.model} ({n_params:,} params)")
        return

    if not os.path.exists(HAM_METADATA_CSV):
        raise FileNotFoundError(
            f"HAM10000 metadata not found at {HAM_METADATA_CSV}. "
            "Download dataset first or run with --dry-run."
        )

    # Speed over bit-exact reproducibility: autotune conv kernels (the splits and the
    # seeds stay fixed; only float summation order may differ run to run).
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(HAM_METADATA_CSV)
    print("Preparing image cache (hair removal + resize, one-off)...", flush=True)
    build_cache(
        df["image_id"].tolist(),
        [HAM_IMAGES_DIR_1, HAM_IMAGES_DIR_2],
        CACHE_DIR,
        hair_removal=True,
        workers=max(args.workers, 4),
    )
    train_loader, val_loader, test_loader = build_loaders(
        df,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.workers,
        splits_csv=None if args.smoke else SPLITS_CSV,
        cache_dir=CACHE_DIR,
        smoke_n=args.smoke or None,
        gpu_pipeline={"auto": None, "gpu": True, "workers": False}[args.loader],
    )
    models = ["efficientnet", "vit"] if args.model == "both" else [args.model]
    for key in models:
        _run_one(args, key, train_loader, val_loader, test_loader)
    if not args.smoke:
        write_status(args.status_file, phase="finished", done=True)


if __name__ == "__main__":
    main()
