"""Final evaluation of the R2 ensemble (run ONCE). Config fixed before looking at the test set:
members = 4 ConvNeXt-S runs from R2 (with 4-view flip TTA) + 1 ConvNeXt-S 384 run from R1 (no TTA);
log-prob average; per-class log-bias tuned on the FULL validation set (coordinate ascent for balanced accuracy,
grid [-1.5, 1.5], 3 passes). Bias tuning was validated by lesion-disjoint cross-fitting inside validation.
"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score as bas, f1_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
R1, R2 = str(ROOT / "outputs/kaggle_runs/r1_full") + "/", str(ROOT / "outputs/kaggle_runs/r2_full") + "/"
C = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
s = pd.read_csv(ROOT / "results/splits.csv")
vai, tei = np.where(s.split.values == "val")[0], np.where(s.split.values == "test")[0]
yall = s.dx.map({c: i for i, c in enumerate(C)}).values
lesion = s.lesion_id.values
def lg(p): return np.log(np.clip(p, 1e-6, 1))
members = [(R2 + n + "/val_probs_tta.npy", R2 + n + "/test_probs_tta.npy") for n in ["cnx384_s42", "cnx384_s7", "cnx448_s42", "cnx448_s7"]]
members.append((R1 + "cnx384/val_probs.npy", R1 + "cnx384/test_probs.npy"))
for n in ["cnx384_s42", "cnx448_s42"]:
    assert (np.load(R2 + n + "/test_idx.npy") == tei).all() and (np.load(R2 + n + "/val_idx.npy") == vai).all()
assert (np.load(R1 + "cnx384/test_idx.npy") == tei).all()
LV = np.mean([lg(np.load(v)) for v, _ in members], 0)
LT = np.mean([lg(np.load(t)) for _, t in members], 0)
yv, yt = yall[vai], yall[tei]
def tune(l, yy, iters=3, grid=np.linspace(-1.5, 1.5, 13)):
    b = np.zeros(7)
    for _ in range(iters):
        for k in range(7):
            best = (-1, 0)
            for g in grid:
                bb = b.copy(); bb[k] = g; v = bas(yy, (l + bb).argmax(1))
                if v > best[0] + 1e-9: best = (v, g)
            b[k] = best[1]
    return b
bias = tune(LV, yv)
def norm(l): e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def metrics(y, pred, prob):
    mel = C.index("mel")
    return dict(acc=float((pred == y).mean()), bal_acc=float(bas(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                macro_auc=float(roc_auc_score(y, prob, multi_class="ovr")), mel_sens=float(((pred == mel) & (y == mel)).sum() / (y == mel).sum()))
prob = norm(LT + bias)
pred = prob.argmax(1)
res = {"bias": bias.tolist(), "ensemble_biased": metrics(yt, pred, prob), "ensemble_unbiased": metrics(yt, LT.argmax(1), norm(LT))}
# baseline (ViT-B/16, previous pipeline) from committed predictions
bl = pd.read_csv(ROOT / "results/vit_test_predictions.csv")
print("baseline csv columns:", list(bl.columns)[:12])
# lesion bootstrap CIs and paired difference vs baseline if aligned by image_id
ids = s.image_id.values[tei]
bcol = [c for c in bl.columns if c.lower() in ("pred", "prediction", "y_pred", "predicted")]
if "image_id" in bl.columns and bcol:
    bl = bl.set_index("image_id").loc[ids]
    bp = bl[bcol[0]].map({c: i for i, c in enumerate(C)}).values if bl[bcol[0]].dtype == object else bl[bcol[0]].values
    res["baseline_vit224"] = {"acc": float((bp == yt).mean()), "bal_acc": float(bas(yt, bp)), "macro_f1": float(f1_score(yt, bp, average="macro"))}
rng = np.random.default_rng(42)
ul = np.unique(lesion[tei]); idx = {l: np.where(lesion[tei] == l)[0] for l in ul}
boot = {"acc": [], "bal_acc": [], "macro_f1": [], "mel_sens": [], "d_bal_acc": [], "d_macro_f1": []}
for _ in range(1000):
    ix = np.concatenate([idx[l] for l in rng.choice(ul, len(ul))])
    boot["acc"].append((pred[ix] == yt[ix]).mean()); boot["bal_acc"].append(bas(yt[ix], pred[ix]))
    boot["macro_f1"].append(f1_score(yt[ix], pred[ix], average="macro"))
    boot["mel_sens"].append(((pred[ix] == 1) & (yt[ix] == 1)).sum() / max((yt[ix] == 1).sum(), 1))
    if "baseline_vit224" in res:
        boot["d_bal_acc"].append(bas(yt[ix], pred[ix]) - bas(yt[ix], bp[ix])); boot["d_macro_f1"].append(f1_score(yt[ix], pred[ix], average="macro") - f1_score(yt[ix], bp[ix], average="macro"))
res["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items() if v}
rec = {C[k]: float(((pred == k) & (yt == k)).sum() / (yt == k).sum()) for k in range(7)}
res["per_class_sensitivity"] = rec
out = ROOT / "results/final_r2_metrics.json"
assert not out.exists(), "final test evaluation was already run; do not repeat"
out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
