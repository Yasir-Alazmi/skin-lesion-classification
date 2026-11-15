"""Tests for the lesion-grouped, class-stratified train/val/test split."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.split import SPLIT_NAMES, load_or_create_splits, make_lesion_split, split_frames

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]


def _synthetic_metadata(lesions_per_class: int = 60) -> pd.DataFrame:
    """Every class has many lesions; lesions have 1 to 3 images each."""
    rows = []
    for c_idx, cls in enumerate(CLASSES):
        for k in range(lesions_per_class):
            lesion = f"HAM_{cls}_{k:03d}"
            for j in range(1 + (k % 3)):
                rows.append({"lesion_id": lesion, "image_id": f"ISIC_{c_idx}{k:03d}{j}", "dx": cls})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    return _synthetic_metadata()


def test_zero_lesion_overlap_between_any_two_splits(meta):
    train, val, test = split_frames(make_lesion_split(meta, seed=42))
    sets = {
        "train": set(train["lesion_id"]),
        "val": set(val["lesion_id"]),
        "test": set(test["lesion_id"]),
    }
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_every_image_assigned_exactly_once(meta):
    out = make_lesion_split(meta, seed=42)
    assert len(out) == len(meta)
    assert out["image_id"].is_unique
    assert set(out["image_id"]) == set(meta["image_id"])
    assert set(out["split"]) == set(SPLIT_NAMES)
    assert out["split"].notna().all()


def test_deterministic_for_fixed_seed_and_input_order(meta):
    a = make_lesion_split(meta, seed=42)
    b = make_lesion_split(meta, seed=42)
    shuffled = meta.sample(frac=1.0, random_state=1)
    c = make_lesion_split(shuffled, seed=42)
    key = ["image_id", "split"]
    assert a[key].equals(b[key])
    assert a[key].reset_index(drop=True).equals(c[key].reset_index(drop=True))


def test_different_seed_changes_assignment(meta):
    a = make_lesion_split(meta, seed=42).set_index("image_id")["split"]
    b = make_lesion_split(meta, seed=7).set_index("image_id")["split"]
    assert not a.equals(b.loc[a.index])


def test_each_class_present_in_every_split(meta):
    out = make_lesion_split(meta, seed=42)
    for name in SPLIT_NAMES:
        assert set(out.loc[out["split"] == name, "dx"]) == set(CLASSES), name


def test_ratios_are_close_to_70_15_15(meta):
    out = make_lesion_split(meta, seed=42)
    frac = out["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.04
    assert abs(frac["val"] - 0.15) < 0.04
    assert abs(frac["test"] - 0.15) < 0.04


def test_naive_image_level_split_leaks_but_ours_does_not():
    """Tiny frame where an image-level stratified split provably leaks lesions."""
    rows = []
    for cls in CLASSES:
        for k in range(20):
            lesion = f"HAM_{cls}_{k:02d}"
            for j in range(4):  # 4 images per lesion makes leakage near certain
                rows.append({"lesion_id": lesion, "image_id": f"{lesion}_img{j}", "dx": cls})
    df = pd.DataFrame(rows)

    # Naive baseline: what the original pipeline did (stratify on dx only).
    train_val, naive_test = train_test_split(df, test_size=0.15, stratify=df["dx"], random_state=42)
    naive_leak = set(train_val["lesion_id"]) & set(naive_test["lesion_id"])
    assert len(naive_leak) > 0, "negative case is only meaningful if the naive split leaks"

    train, val, test = split_frames(make_lesion_split(df, seed=42))
    assert set(train["lesion_id"]).isdisjoint(test["lesion_id"])
    assert set(train["lesion_id"]).isdisjoint(val["lesion_id"])
    assert set(val["lesion_id"]).isdisjoint(test["lesion_id"])


def test_missing_class_in_a_split_raises():
    df = pd.DataFrame(
        {
            "lesion_id": [f"L{i}" for i in range(6)],
            "image_id": [f"I{i}" for i in range(6)],
            "dx": ["nv", "nv", "nv", "nv", "nv", "mel"],
        }
    )
    with pytest.raises(ValueError):
        make_lesion_split(df, seed=42)


def test_lesion_with_conflicting_labels_raises():
    df = pd.DataFrame({"lesion_id": ["A", "A"], "image_id": ["i1", "i2"], "dx": ["nv", "mel"]})
    with pytest.raises(ValueError):
        make_lesion_split(df, seed=42)


def test_load_or_create_splits_writes_audit_csv_and_reuses_it(tmp_path, meta):
    path = tmp_path / "splits.csv"
    first = load_or_create_splits(meta, path, seed=42)
    assert path.exists()
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["image_id", "lesion_id", "dx", "split"]
    assert len(saved) == len(meta)

    again = load_or_create_splits(meta, path, seed=42)
    assert first[["image_id", "split"]].to_numpy().tolist() == (
        again[["image_id", "split"]].to_numpy().tolist()
    )


def test_load_or_create_rejects_stale_csv(tmp_path, meta):
    path = tmp_path / "splits.csv"
    load_or_create_splits(meta, path, seed=42)
    smaller = meta.iloc[: len(meta) // 2]
    with pytest.raises(ValueError):
        load_or_create_splits(smaller, path, seed=42)
