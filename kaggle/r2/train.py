"""
Round 1 sweep for HAM10000 (lesion-grouped split). Runs on Kaggle (2 x T4).

- Same split as the repo: results/splits.csv (train / val / test), uploaded as a Kaggle dataset.
- Model selection by VALIDATION macro-F1 only. Test probabilities are saved but test metrics are
  NOT computed or printed here (the test set is looked at once, at the very end, by scripts/report.py).
- GPU-side augmentation (flip, rotation, scale, colour jitter) on cached uint8 tensors.
- Optional metadata branch (age, sex, localization).
- Two worker processes, one per GPU; each runs its share of CONFIGS.
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "MODE": "full",             # "measure": 40 steps per config to time them; "full": real training
    "MEASURE_STEPS": 40,
    "EPOCHS": 30,
    "PATIENCE": 8,
    "BATCH": 32,
    "LR": 1e-4,
    "WD": 1e-4,
    "WARMUP_EPOCHS": 1,
    "LABEL_SMOOTH": 0.0,
    "FOCAL_GAMMA": 2.0,
    "HAIR": True,               # black-hat + inpaint hair removal, as in the repo
    "MAX_HOURS": 8.0,
    "SEED": 42,
    "RUNS": [                   # (name, timm model, image size, use metadata, seed)
        ["cnx384_s42", "convnext_small.fb_in22k_ft_in1k_384", 384, False, 42],
        ["cnx448_s42", "convnext_small.fb_in22k_ft_in1k_384", 448, False, 42],
        ["cnx384_s7", "convnext_small.fb_in22k_ft_in1k_384", 384, False, 7],
        ["cnx448_s7", "convnext_small.fb_in22k_ft_in1k_384", 448, False, 7],
    ],
}
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./out")
OUT.mkdir(parents=True, exist_ok=True)
INPUT = Path("/kaggle/input")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def find(name, root=INPUT):
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


# ---------------------------------------------------------------- data
def remove_hair(bgr):
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, mask, 6, cv2.INPAINT_TELEA)


def _load_one(args):
    import cv2
    path, size, hair = args
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    if hair:
        bgr = remove_hair(bgr)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_tables():
    meta = pd.read_csv(find("HAM10000_metadata.csv"))
    splits = pd.read_csv(find("skin_splits_lesion_grouped.csv"))
    df = splits.merge(meta[["image_id", "age", "sex", "localization"]], on="image_id", how="left")
    assert len(df) == len(splits) == 10015, (len(df), len(splits))
    assert df.split.isin(["train", "val", "test"]).all()
    return df


def image_paths(df):
    idx = {}
    for p in INPUT.rglob("ISIC_*.jpg"):
        idx[p.stem] = p
    missing = [i for i in df.image_id if i not in idx]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    return [idx[i] for i in df.image_id]


def build_meta(df):
    tr = df[df.split == "train"]
    age_mean, age_std = tr.age.mean(), tr.age.std()
    age = df.age.fillna(age_mean).astype(float)
    feats = [((age - age_mean) / age_std).values[:, None], df.age.isna().values[:, None].astype(float)]
    for col in ("sex", "localization"):
        cats = sorted(tr[col].fillna("unknown").unique())
        v = df[col].fillna("unknown")
        feats.append(np.stack([(v == c).values for c in cats], 1).astype(float))
    return np.concatenate(feats, 1).astype(np.float32)


_CACHE = {}


def get_images(df, size):
    key = size
    if key in _CACHE:
        return _CACHE[key]
    from multiprocessing import Pool
    paths = image_paths(df)
    t = time.time()
    arr = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    with Pool(max(2, (os.cpu_count() or 2) // 2)) as pool:
        for i, im in enumerate(pool.imap(_load_one, [(p, size, CONFIG["HAIR"]) for p in paths], chunksize=32)):
            arr[i] = im
    log(f"cached {arr.shape} at {size}px in {time.time() - t:.0f}s")
    _CACHE.clear()
    _CACHE[key] = arr
    return arr


# ---------------------------------------------------------------- model
def make_model(name, meta_dim):
    import timm
    import torch
    import torch.nn as nn
    backbone = timm.create_model(name, pretrained=True, num_classes=0)
    feat = backbone.num_features

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU(), nn.Dropout(0.1)) if meta_dim else None
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(feat + (64 if meta_dim else 0), len(CLASSES))

        def forward(self, x, m=None):
            f = self.backbone(x)
            if self.meta is not None:
                f = torch.cat([f, self.meta(m)], 1)
            return self.head(self.drop(f))

    return Net(), backbone


def pretrained_cfg(backbone):
    c = backbone.pretrained_cfg
    return list(c.get("mean", (0.485, 0.456, 0.406))), list(c.get("std", (0.229, 0.224, 0.225)))


# ---------------------------------------------------------------- aug / loss / metrics
def gpu_augment(x, gen):
    """x: float tensor (B,3,H,W) in [0,1]. Random flips, rotation, scale, brightness/contrast."""
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    dev = x.device
    ang = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * math.radians(30)
    sc = 1.0 / (0.85 + torch.rand(B, device=dev, generator=gen) * 0.30)
    fx = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos * fx, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos * fy, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * 0.2
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * 0.2
    mean = x.mean((1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean * b).clamp(0, 1)


def focal_loss(logits, y, alpha, gamma, smooth):
    import torch
    import torch.nn.functional as F
    logp = F.log_softmax(logits, 1)
    if smooth > 0:
        n = logits.shape[1]
        tgt = torch.full_like(logp, smooth / (n - 1)).scatter_(1, y[:, None], 1 - smooth)
        ce = -(tgt * logp).sum(1)
        pt = (tgt * logp.exp()).sum(1)
    else:
        ce = F.nll_loss(logp, y, reduction="none")
        pt = logp.gather(1, y[:, None]).squeeze(1).exp()
    return (alpha[y] * (1 - pt) ** gamma * ce).mean()


def metrics(y, prob):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    pred = prob.argmax(1)
    mel = CLASSES.index("mel")
    out = {
        "acc": float((pred == y).mean()),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mel_sens": float(((pred == mel) & (y == mel)).sum() / max((y == mel).sum(), 1)),
    }
    try:
        out["macro_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out


@__import__("torch").no_grad()
def predict(model, X, M, mean, std, dev, bs=64, tta=False):
    import torch
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255
        xb = (xb - mean) / std
        mb = torch.from_numpy(M[i:i + bs]).to(dev) if M is not None else None
        views = [xb, xb.flip(3), xb.flip(2), xb.flip(2).flip(3)] if tta else [xb]
        with torch.autocast("cuda", dtype=torch.float16):
            ps = [torch.softmax(model(v, mb).float(), 1) for v in views]
        out.append(torch.stack(ps).mean(0).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- one run
def run_one(run, df, dev):
    import torch
    name, model_name, size, use_meta, seed = run
    cfg = CONFIG
    measure = cfg["MODE"] == "measure"
    torch.manual_seed(seed); np.random.seed(seed)
    X = get_images(df, size)
    M_all = build_meta(df) if use_meta else None
    y_all = df.dx.map({c: i for i, c in enumerate(CLASSES)}).values
    sp = df.split.values
    tri, vai, tei = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    model, backbone = make_model(model_name, M_all.shape[1] if use_meta else 0)
    mean, std = pretrained_cfg(backbone)
    mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1); std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    model.to(dev)
    counts = np.bincount(y_all[tri], minlength=len(CLASSES))
    alpha = torch.tensor((counts.sum() / (len(CLASSES) * counts)) ** 0.5, dtype=torch.float32, device=dev)
    alpha = alpha / alpha.mean()
    w = (1.0 / counts)[y_all[tri]]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(tri), replacement=True,
                                                     generator=torch.Generator().manual_seed(seed))
    bs = cfg["BATCH"] if size <= 384 else 24   # 448px at batch 32 risks T4 OOM (R1 peak 10.2 GB at 384)
    steps_per_epoch = math.ceil(len(tri) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    total = cfg["EPOCHS"] * steps_per_epoch
    warm = cfg["WARMUP_EPOCHS"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    res = {"name": name, "model": model_name, "size": size, "meta": use_meta, "history": []}
    best, best_ep, bad = -1.0, -1, 0
    odir = OUT / name; odir.mkdir(exist_ok=True)
    for ep in range(cfg["EPOCHS"]):
        model.train()
        t_ep = time.time()
        order = tri[np.fromiter(iter(sampler), dtype=np.int64, count=len(tri))]
        tot_loss = 0.0
        n_steps = cfg["MEASURE_STEPS"] if measure else steps_per_epoch
        for s in range(n_steps):
            ids = order[s * bs:(s + 1) * bs]
            if len(ids) < 2:
                break
            xb = torch.from_numpy(X[ids]).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255
            xb = (gpu_augment(xb, gen) - mean) / std
            yb = torch.from_numpy(y_all[ids]).to(dev)
            mb = torch.from_numpy(M_all[ids]).to(dev) if use_meta else None
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xb, mb)
            loss = focal_loss(logits.float(), yb, alpha, cfg["FOCAL_GAMMA"], cfg["LABEL_SMOOTH"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot_loss += float(loss)
        torch.cuda.synchronize()
        dt = time.time() - t_ep
        if measure:
            sps = dt / n_steps
            res.update({"s_per_step": sps, "steps_per_epoch": steps_per_epoch,
                        "est_epoch_s": sps * steps_per_epoch, "est_total_h": sps * steps_per_epoch * cfg["EPOCHS"] / 3600,
                        "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
            log(name, "MEASURED", {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items() if k.startswith(("s_per", "est", "peak"))})
            break
        pv = predict(model, X[vai], M_all[vai] if use_meta else None, mean, std, dev)
        mv = metrics(y_all[vai], pv)
        mv.update({"epoch": ep, "train_loss": tot_loss / n_steps, "epoch_s": dt})
        res["history"].append(mv)
        log(name, f"ep{ep}", {k: round(v, 4) for k, v in mv.items()})
        if mv["macro_f1"] > best:
            best, best_ep, bad = mv["macro_f1"], ep, 0
            torch.save(model.state_dict(), odir / "best.pt")
            np.save(odir / "val_probs.npy", pv)
        else:
            bad += 1
            if bad >= cfg["PATIENCE"]:
                log(name, "early stop"); break
        if (time.time() - T0) / 3600 > cfg["MAX_HOURS"]:
            log(name, "time guard"); break
    if not measure:
        model.load_state_dict(torch.load(odir / "best.pt", map_location=dev))
        pt = predict(model, X[tei], M_all[tei] if use_meta else None, mean, std, dev)
        np.save(odir / "test_probs.npy", pt)      # saved, NOT scored here
        np.save(odir / "val_probs_tta.npy", predict(model, X[vai], None, mean, std, dev, tta=True))
        np.save(odir / "test_probs_tta.npy", predict(model, X[tei], None, mean, std, dev, tta=True))
        res["val_tta"] = metrics(y_all[vai], np.load(odir / "val_probs_tta.npy"))
        np.save(odir / "val_idx.npy", vai); np.save(odir / "test_idx.npy", tei)
        res["best_epoch"] = best_ep
        res["val_best"] = res["history"][best_ep]
    (odir / "result.json").write_text(json.dumps(res, indent=1))
    del model, opt
    torch.cuda.empty_cache()
    return res


def worker(gpu, runs, q):
    import torch
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)
    df = load_tables()
    for run in runs:
        try:
            q.put(run_one(run, df, dev))
        except Exception:
            log(run[0], "FAILED\n" + traceback.format_exc())
            q.put({"name": run[0], "error": traceback.format_exc()[-800:]})


def main():
    import torch
    import torch.multiprocessing as mp
    n_gpu = torch.cuda.device_count()
    log("GPUs:", n_gpu, "mode:", CONFIG["MODE"], "| input roots:", sorted(p.name for p in INPUT.iterdir())[:10])
    runs = CONFIG["RUNS"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    shards = [runs[i::max(n_gpu, 1)] for i in range(max(n_gpu, 1))]
    procs = [ctx.Process(target=worker, args=(g, shards[g], q)) for g in range(max(n_gpu, 1))]
    for p in procs:
        p.start()
    results = [q.get() for _ in runs]
    for p in procs:
        p.join()
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    log("DONE", [(r["name"], r.get("val_best", {}).get("macro_f1"), r.get("val_tta", {}).get("macro_f1")) for r in results])


if __name__ == "__main__":
    # script is re-imported by spawned workers; only the parent runs main()
    main()
