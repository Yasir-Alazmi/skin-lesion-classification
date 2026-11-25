"""
Export the ConvNeXt-S ensemble members to ONNX and check them against PyTorch.

    python scripts/export_onnx.py --model-dir models/ensemble

What it writes:
- ``<model-dir>/<member>.onnx`` for each member, next to the .pt weights. Git ignores these files.
- ``results/onnx_export.json`` with the parity and CPU latency numbers.

Parity is checked only on the demo images in ``data/samples``, never on the test set. The test set
is not re-used (see README). The script runs on CPU only.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serving.model import load_onnx_ensemble, onnx_name  # noqa: E402
from src.ensemble import load_ensemble  # noqa: E402
from src.preprocess import read_image_bgr  # noqa: E402

OPSET = 17


def export_members(model_dir: Path) -> list[dict]:
    cfg = json.loads((model_dir / "ensemble_config.json").read_text(encoding="utf-8"))
    ens = load_ensemble(model_dir, device="cpu")
    out = []
    for (model, size), m in zip(ens.members, cfg["members"]):
        path = model_dir / onnx_name(m["file"])
        dummy = torch.randn(1, 3, size, size)
        torch.onnx.export(model, dummy, str(path), opset_version=OPSET, input_names=["image"],
                          output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False)
        out.append({"member": m["file"], "onnx": path.name, "size": size,
                    "onnx_mb": round(path.stat().st_size / 2**20, 1)})
        print(f"exported {path.name} ({out[-1]['onnx_mb']} MB)")
    return out


def timed(fn, images, warmup: int = 1) -> tuple[list[np.ndarray], list[float]]:
    for img in images[:warmup]:
        fn(img)
    outs, times = [], []
    for img in images:
        t0 = time.perf_counter()
        outs.append(fn(img))
        times.append((time.perf_counter() - t0) * 1000.0)
    return outs, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "ensemble"))
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    model_dir = Path(args.model_dir)

    members = export_members(model_dir)
    if args.threads:
        torch.set_num_threads(args.threads)

    samples = sorted((ROOT / "data" / "samples").glob("*.jpg"))
    images = [read_image_bgr(p)[:, :, ::-1].copy() for p in samples]  # BGR -> RGB

    torch_ens = load_ensemble(model_dir, device="cpu")
    onnx_ens = load_onnx_ensemble(model_dir, threads=args.threads)
    t_out, t_ms = timed(torch_ens.predict_probs, images)
    o_out, o_ms = timed(onnx_ens.predict_probs, images)

    diffs = [float(np.abs(a - b).max()) for a, b in zip(t_out, o_out)]
    agree = sum(int(np.argmax(a) == np.argmax(b)) for a, b in zip(t_out, o_out))
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "what": "ONNX export of the 4-member ConvNeXt-S ensemble; parity vs PyTorch on data/samples (not the test set)",
        "opset": OPSET,
        "members": members,
        "parity": {"images": len(images), "max_abs_prob_diff": max(diffs),
                   "mean_max_abs_prob_diff": float(np.mean(diffs)), "top1_agreement": f"{agree}/{len(images)}"},
        "cpu_latency_ms_per_image": {
            "note": "full request: 4 members x 4 flip views, incl. hair removal; first call excluded as warm-up",
            "pytorch_median": round(statistics.median(t_ms), 1),
            "onnxruntime_median": round(statistics.median(o_ms), 1),
            "pytorch_mean": round(statistics.mean(t_ms), 1),
            "onnxruntime_mean": round(statistics.mean(o_ms), 1),
        },
        "environment": {"cpu": platform.processor(), "python": platform.python_version(),
                        "torch": torch.__version__, "onnxruntime": __import__("onnxruntime").__version__,
                        "threads": args.threads or "default"},
    }
    out = ROOT / "results" / "onnx_export.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["parity"], indent=2))
    print(json.dumps(report["cpu_latency_ms_per_image"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
