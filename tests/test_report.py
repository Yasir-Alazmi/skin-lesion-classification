"""Tests for the bootstrap/metric helpers in scripts/report.py (no data files needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

import report  # noqa: E402


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 7, n)
    probs = rng.dirichlet(np.ones(7), n)
    probs[np.arange(n), y] += 1.0  # make the true class most likely
    probs /= probs.sum(1, keepdims=True)
    return y, probs.argmax(1), probs


def test_compute_metrics_matches_sklearn():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y, pred, probs = _toy()
    m = report.compute_metrics(y, pred, probs)
    assert np.isclose(m["accuracy"], accuracy_score(y, pred))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, pred))
    assert np.isclose(m["macro_f1"], f1_score(y, pred, average="macro"))
    assert m["confusion_matrix"].sum() == len(y)


def test_lesion_bootstrap_is_seeded_and_groups_images():
    y, pred, probs = _toy(200)
    lesions = np.repeat(np.arange(100), 2)  # two images per lesion
    a = report.lesion_bootstrap(lesions, {"m": (y, pred, probs)}, 20, seed=42)["m"]
    b = report.lesion_bootstrap(lesions, {"m": (y, pred, probs)}, 20, seed=42)["m"]
    assert a.shape[0] == 20 and np.array_equal(a, b, equal_nan=True)
    lo, hi = report.ci(a)
    assert (lo[:4] <= hi[:4]).all()
