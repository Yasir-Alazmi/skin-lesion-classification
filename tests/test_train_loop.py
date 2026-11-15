"""CPU-only checks of the training loop's model-selection behaviour."""

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import NUM_CLASSES
from src.train import train_model


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 16)
        self.head = nn.Linear(16, NUM_CLASSES)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def _loader(n=70):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 8, generator=g)
    y = torch.arange(n) % NUM_CLASSES
    return DataLoader(TensorDataset(x, y), batch_size=10)


def test_train_model_selects_by_val_f1_and_saves_checkpoint(tmp_path):
    ckpt = tmp_path / "best.pth"
    hist = train_model(
        _Tiny(),
        _loader(),
        _loader(),
        model_name="tiny",
        num_epochs=3,
        unfreeze_epoch=1,
        checkpoint_path=str(ckpt),
        device=torch.device("cpu"),
        log_path=str(tmp_path / "h.json"),
    )
    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=True)
    assert saved["val_f1"] == hist["val_f1"][saved["epoch"] - 1]
    assert len(hist["val_f1"]) == len(hist["epoch_seconds"]) >= 1
    assert (tmp_path / "h.json").exists()


def _shuffled_loader(n=70):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 8, generator=g)
    y = torch.arange(n) % NUM_CLASSES
    return DataLoader(TensorDataset(x, y), batch_size=10, shuffle=True)


def _run(tmp_path, epochs, resume, seed=123):
    torch.manual_seed(seed)  # model init + shuffle order come from the global RNG
    model = _Tiny()
    hist = train_model(
        model,
        _shuffled_loader(),
        _loader(),
        model_name="tiny",
        num_epochs=epochs,
        unfreeze_epoch=2,
        patience=50,
        checkpoint_path=str(tmp_path / "best_tiny.pth"),
        device=torch.device("cpu"),
        resume=resume,
        status_path=str(tmp_path / "status.json"),
    )
    return model, hist


def test_resume_continues_identically(tmp_path):
    ref_dir, cut_dir = tmp_path / "ref", tmp_path / "cut"
    ref_dir.mkdir()
    cut_dir.mkdir()
    ref_model, ref = _run(ref_dir, epochs=5, resume=False)

    _run(cut_dir, epochs=3, resume=False)  # "interrupted" after epoch 3 (past the unfreeze)
    assert (cut_dir / "last_tiny.pth").exists()
    # Garbage RNG/model state from an unrelated run must not leak into the resumed run.
    torch.manual_seed(999)
    res_model, res = _run(cut_dir, epochs=5, resume=True, seed=999)

    for key in ("train_loss", "val_f1", "val_loss", "lr_history"):
        assert res[key] == ref[key], key
    assert res["best_epoch"] == ref["best_epoch"]
    for a, b in zip(ref_model.state_dict().values(), res_model.state_dict().values()):
        assert torch.equal(a, b)
    st = json.loads((cut_dir / "status.json").read_text())
    assert st["epoch"] == 5 and st["model"] == "tiny" and "updated" in st
