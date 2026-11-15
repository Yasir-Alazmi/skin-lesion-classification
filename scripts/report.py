"""Recompute every reported metric from the committed prediction CSVs.

Reads ``results/{vit,efficientnet}_test_predictions.csv`` and ``results/splits.csv``,
computes accuracy, balanced accuracy, macro-F1, macro-AUC and per-class
sensitivity / specificity / F1 / AUC, and attaches 95% bootstrap confidence
intervals.  The bootstrap resamples whole *lesions* (``lesion_id``), never single
images, so the several photographs of one lesion cannot make the intervals
look tighter than they are.

Outputs (all under ``results/``): ``metrics.json``, ``metrics.md`` and the figures
(confusion matrices, ROC curves, sensitivity/specificity, training history, model
comparison) drawn with ``src/visualize.py``.

Usage (repo root):  python scripts/report.py [--n-boot 1000] [--seed 42] [--no-figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, LABEL_MAP  # noqa: E402

CODES = list(LABEL_MAP)  # class-index order: nv, mel, bkl, bcc, akiec, vasc, df
N_CLS = len(CODES)
MODELS = {
    "ViT-B/16": ("vit", "vit_b16"),
    "EfficientNet-B3": ("efficientnet", "efficientnet_b3"),
}
OVERALL = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]
PER_CLASS = ["sensitivity", "specificity", "f1", "auc"]


def load_predictions(path: Path, splits: pd.DataFrame) -> pd.DataFrame:
    """Predictions joined with lesion_id; checks they cover exactly the test split."""
    df = pd.read_csv(path)
    test_ids = set(splits.loc[splits["split"] == "test", "image_id"])
    if set(df["image_id"]) != test_ids or len(df) != len(test_ids):
        raise ValueError(f"{path.name}: image_ids do not match the test split of splits.csv")
    df = df.merge(splits[["image_id", "lesion_id"]], on="image_id", how="left")
    return df.sort_values("image_id").reset_index(drop=True)


def compute_metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict:
    """All metrics for one set of predictions, as flat numpy values."""
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    row = cm.sum(axis=1).astype(float)  # support per true class
    col = cm.sum(axis=0).astype(float)  # predicted per class
    fn, fp = row - tp, col - tp
    tn = n - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(row > 0, tp / row, np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        f1 = np.where(row + col > 0, 2 * tp / (row + col), 0.0)
    auc = np.full(N_CLS, np.nan)
    for c in range(N_CLS):
        pos = y == c
        if 0 < pos.sum() < n:
            auc[c] = roc_auc_score(pos, probs[:, c])
    return {
        "accuracy": tp.sum() / n,
        "balanced_accuracy": np.nanmean(sens),
        "macro_f1": f1.mean(),
        "macro_auc": np.nanmean(auc),
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }


def flat(m: dict) -> np.ndarray:
    """Metric dict -> one vector (overall metrics, then each per-class block)."""
    return np.concatenate([[m[k] for k in OVERALL]] + [m[k] for k in PER_CLASS])


def unflat(v: np.ndarray) -> dict:
    out = {k: v[..., i] for i, k in enumerate(OVERALL)}
    for j, k in enumerate(PER_CLASS):
        s = len(OVERALL) + j * N_CLS
        out[k] = v[..., s : s + N_CLS]
    return out


def lesion_bootstrap(
    lesion_ids: np.ndarray, per_model: dict[str, tuple], n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    """Metric vectors for ``n_boot`` lesion-level resamples, shared by all models.

    Using the same resamples for every model makes paired differences valid.
    """
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(lesion_ids, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    out = {name: [] for name in per_model}
    for _ in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[k] for k in draw])
        for name, (y, pred, probs) in per_model.items():
            out[name].append(flat(compute_metrics(y[idx], pred[idx], probs[idx])))
    return {name: np.stack(v) for name, v in out.items()}


def ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return lo, hi


def pack(point: dict, boot_vec: np.ndarray) -> dict:
    """JSON-ready dict of value + 95% CI for every metric."""
    lo_v, hi_v = ci(boot_vec)
    lo, hi = unflat(lo_v), unflat(hi_v)

    def cell(v, a, b):
        return {
            "value": round(float(v), 4),
            "ci_low": round(float(a), 4),
            "ci_high": round(float(b), 4),
        }

    res = {k: cell(point[k], lo[k], hi[k]) for k in OVERALL}
    res["per_class"] = {
        code: {k: cell(point[k][i], lo[k][i], hi[k][i]) for k in PER_CLASS}
        for i, code in enumerate(CODES)
    }
    res["confusion_matrix"] = point["confusion_matrix"].tolist()
    return res


def fmt(c: dict, pct: bool = False) -> str:
    if pct:
        return f"{c['value']*100:.1f}% [{c['ci_low']*100:.1f}, {c['ci_high']*100:.1f}]"
    return f"{c['value']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def write_markdown(path: Path, metrics: dict) -> None:
    meta = metrics["meta"]
    names = list(metrics["models"])
    lines = [
        "# Test-set metrics",
        "",
        f"Lesion-grouped test split: {meta['n_images']:,} images from {meta['n_lesions']:,} "
        f"lesions, evaluated once. Values are point estimates with 95% bootstrap confidence "
        f"intervals in brackets ({meta['n_boot']:,} resamples of lesions, seed {meta['seed']}).",
        "Generated by `scripts/report.py` from `results/*_test_predictions.csv`; do not edit.",
        "",
        "## Overall",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro-F1",
        "macro_auc": "Macro-AUC (one-vs-rest)",
    }
    for k in OVERALL:
        lines.append(
            f"| {labels[k]} | " + " | ".join(fmt(metrics["models"][n][k]) for n in names) + " |"
        )
    if "comparison" in metrics:
        c = metrics["comparison"]
        diffs = ", ".join(f"{labels[k]} {fmt(c[k])}" for k in ("macro_f1", "macro_auc"))
        lines += [
            "",
            f"Paired difference ({c['a']} minus {c['b']}, same lesion resamples): {diffs}.",
        ]
    for n in names:
        lines += [
            "",
            f"## Per-class: {n}",
            "",
            "| Class | Support | Sensitivity | Specificity | F1 | AUC |",
            "|---|---|---|---|---|---|",
        ]
        cm = np.array(metrics["models"][n]["confusion_matrix"])
        for i, code in enumerate(CODES):
            pc = metrics["models"][n]["per_class"][code]
            lines.append(
                f"| {CLASS_NAMES[i]} ({code}) | {int(cm[i].sum())} | {fmt(pc['sensitivity'])} | "
                f"{fmt(pc['specificity'])} | {fmt(pc['f1'])} | {fmt(pc['auc'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(out_dir: Path, data: dict, points: dict, history_dir: Path) -> list[str]:
    """Regenerate result figures with src/visualize.py; returns the files written."""
    import matplotlib.pyplot as plt

    from src import visualize as viz

    written: list[str] = []

    def save(fig, name: str) -> None:
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    results = {}
    for name, (y, _, probs) in data.items():
        p = points[name]
        results[name] = {
            "probs": probs,
            "labels": y,
            "sensitivity": np.nan_to_num(p["sensitivity"]),
            "specificity": np.nan_to_num(p["specificity"]),
            "accuracy": p["accuracy"],
            "f1_macro": p["macro_f1"],
            "auc": p["macro_auc"],
            "confusion_matrix": p["confusion_matrix"],
        }
        tag = MODELS[name][1]
        save(
            viz.plot_confusion_matrix(p["confusion_matrix"], model_name=name, save=False),
            f"confusion_matrix_{tag}.png",
        )
        save(
            viz.plot_roc_curves(results[name], model_name=name, save=False), f"roc_curves_{tag}.png"
        )
        save(
            viz.plot_sens_spec(results[name], model_name=name, save=False),
            f"sensitivity_specificity_{tag}.png",
        )

    save(
        viz.plot_model_comparison(results["EfficientNet-B3"], results["ViT-B/16"], save=False),
        "model_comparison.png",
    )

    hist = {}
    for name, (_, tag) in MODELS.items():
        f = history_dir / f"history_{tag}.json"
        if f.exists():
            hist[name] = json.loads(f.read_text())
    if len(hist) == 2:
        save(
            viz.plot_training_history(hist["EfficientNet-B3"], hist["ViT-B/16"], save=False),
            "training_history.png",
        )
    else:
        print("history_*.json missing in results/: training_history.png not regenerated")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    res = args.results_dir

    splits = pd.read_csv(res / "splits.csv")
    data, lesions = {}, None
    for name, (short, _) in MODELS.items():
        df = load_predictions(res / f"{short}_test_predictions.csv", splits)
        y = df["label"].map(LABEL_MAP).to_numpy()
        pred = df["pred"].map(LABEL_MAP).to_numpy()
        probs = df[[f"p_{c}" for c in CODES]].to_numpy(dtype=float)
        data[name] = (y, pred, probs)
        if lesions is None:
            lesions = df["lesion_id"].to_numpy()
        elif not np.array_equal(lesions, df["lesion_id"].to_numpy()):
            raise ValueError("models are not evaluated on the same ordered image set")

    points = {n: compute_metrics(*d) for n, d in data.items()}
    boots = lesion_bootstrap(lesions, data, args.n_boot, args.seed)

    metrics = {
        "meta": {
            "n_images": int(len(lesions)),
            "n_lesions": int(len(np.unique(lesions))),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "resampling_unit": "lesion_id",
            "ci": "95% percentile bootstrap",
        },
        "models": {n: pack(points[n], boots[n]) for n in data},
    }
    a, b = "ViT-B/16", "EfficientNet-B3"
    diff = boots[a] - boots[b]
    d_point = flat(points[a]) - flat(points[b])
    d_lo, d_hi = ci(diff)
    metrics["comparison"] = {"a": a, "b": b}
    for i, k in enumerate(OVERALL):
        metrics["comparison"][k] = {
            "value": round(float(d_point[i]), 4),
            "ci_low": round(float(d_lo[i]), 4),
            "ci_high": round(float(d_hi[i]), 4),
        }

    (res / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_markdown(res / "metrics.md", metrics)
    print(f"wrote {res / 'metrics.json'} and {res / 'metrics.md'}")
    for n in data:
        print(n, {k: fmt(metrics["models"][n][k]) for k in OVERALL})

    if not args.no_figures:
        for f in make_figures(res, data, points, res):
            print("figure:", f)


if __name__ == "__main__":
    main()
