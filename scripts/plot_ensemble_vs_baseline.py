"""
Plot the v2.1 ensemble against the v2.0 ViT-B/16 baseline.

    python scripts/plot_ensemble_vs_baseline.py

Reads only ``results/final_r2_metrics.json`` and writes ``results/ensemble_vs_baseline.png``.
No model is run and the test set is not touched. Error bars are the 95% lesion-level bootstrap
CIs stored in that file (available for the ensemble only). The file has no melanoma sensitivity
for the baseline, so that bar is left out rather than taken from elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "final_r2_metrics.json"
OUT = ROOT / "results" / "ensemble_vs_baseline.png"

METRICS = [("bal_acc", "Balanced accuracy"), ("macro_f1", "Macro-F1"), ("mel_sens", "Melanoma sensitivity")]
ENS_COLOR = "#2a6f97"
BASE_COLOR = "#a9a9a9"


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    ens, base, ci = data["ensemble_biased"], data["baseline_vit224"], data["ci95"]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    width = 0.36
    for i, (key, _) in enumerate(METRICS):
        v = ens[key]
        err = None
        if key in ci:
            lo, hi = ci[key]
            err = [[v - lo], [hi - v]]
        ax.bar(i - width / 2, v, width, color=ENS_COLOR, yerr=err, capsize=4,
               label="ConvNeXt-S ensemble + bias (5 models)" if i == 0 else None)
        ax.text(i - width / 2, (ci[key][1] if key in ci else v) + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
        if key in base:
            b = base[key]
            ax.bar(i + width / 2, b, width, color=BASE_COLOR, label="ViT-B/16 baseline" if i == 0 else None)
            ax.text(i + width / 2, b + 0.015, f"{b:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test set, 1,503 images, lesion-grouped; 95% bootstrap CIs", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.text(0.01, 0.01, "Error bars: ensemble only. Baseline melanoma sensitivity is not stored in "
             "final_r2_metrics.json, so it is not shown.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
