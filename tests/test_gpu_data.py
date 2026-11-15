"""CPU checks of the batch-level (GPU-capable) data pipeline."""

import numpy as np
import pandas as pd
import torch

from src.gpu_data import GPUBatchLoader, augment, normalize, to_float


def _batch(n=8, size=32):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g)


def test_augment_keeps_shape_range_and_changes_pixels():
    torch.manual_seed(0)
    x = _batch()
    y = augment(x.clone())
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1
    assert not torch.allclose(x, y)


def test_hue_roundtrip_is_identity():
    from src.gpu_data import _hsv_to_rgb, _rgb_to_hsv

    x = _batch()
    assert torch.allclose(_hsv_to_rgb(_rgb_to_hsv(x)), x, atol=1e-5)


def test_loader_eval_matches_totensor_normalize_and_train_is_random():
    imgs = np.random.default_rng(0).integers(0, 256, (10, 16, 16, 3), dtype=np.uint8)
    df = pd.DataFrame({"image_id": [f"i{k}" for k in range(10)], "dx": ["nv"] * 10})
    labels = [k % 7 for k in range(10)]
    ev = GPUBatchLoader(imgs, range(10), labels, df, 4, torch.device("cpu"))
    batches = list(ev)
    assert [len(b[0]) for b in batches] == [4, 4, 2]
    expect = normalize(to_float(torch.from_numpy(imgs[:4])))
    assert torch.allclose(batches[0][0], expect)
    assert batches[0][1].tolist() == labels[:4]

    tr = GPUBatchLoader(imgs, range(10), labels, df, 4, torch.device("cpu"), train=True)
    assert len(tr) == 3 and len(list(tr)) == 3
    a, b = next(iter(tr))[0], next(iter(tr))[0]
    assert a.shape == (4, 3, 16, 16) and not torch.equal(a, b)
