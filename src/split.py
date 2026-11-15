"""
split.py
========
Lesion-grouped, class-stratified train / val / test split for HAM10000.

HAM10000 holds several images of the same physical lesion (``lesion_id``).
Splitting images independently lets near-duplicates of a lesion land in both
train and test, which inflates test metrics.  Every function here assigns a
whole ``lesion_id`` to exactly one split.

The split is produced with ``StratifiedGroupKFold`` over 20 folds (5 % each):
3 folds -> test (15 %), 3 folds -> val (15 %), 14 folds -> train (70 %).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
SPLIT_COLUMNS: list[str] = ["image_id", "lesion_id", "dx", "split"]

_N_FOLDS = 20  # each fold is 5 % of the data
_TEST_FOLDS = 3  # 15 %
_VAL_FOLDS = 3  # 15 %


def _validate(df: pd.DataFrame) -> None:
    missing = {"image_id", "lesion_id", "dx"} - set(df.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if df["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    if (df.groupby("lesion_id")["dx"].nunique() > 1).any():
        raise ValueError("a lesion_id carries more than one diagnosis")


def make_lesion_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign every image to train / val / test with no ``lesion_id`` shared
    between splits, stratified by ``dx`` at the lesion level.

    The input row order does not matter: rows are sorted by ``image_id`` first,
    so the result is a pure function of (metadata content, seed).

    Returns
    -------
    DataFrame with columns ``image_id, lesion_id, dx, split`` sorted by image_id.

    Raises
    ------
    ValueError
        On malformed metadata or if any class is absent from any split.
    """
    _validate(df)
    data = df[["image_id", "lesion_id", "dx"]].sort_values("image_id").reset_index(drop=True)

    if data["lesion_id"].nunique() < _N_FOLDS:
        raise ValueError(f"need at least {_N_FOLDS} distinct lesions to split")

    sgkf = StratifiedGroupKFold(n_splits=_N_FOLDS, shuffle=True, random_state=seed)
    split = pd.Series(index=data.index, dtype=object)
    for fold, (_, idx) in enumerate(sgkf.split(data, data["dx"], groups=data["lesion_id"])):
        if fold < _TEST_FOLDS:
            split.iloc[idx] = "test"
        elif fold < _TEST_FOLDS + _VAL_FOLDS:
            split.iloc[idx] = "val"
        else:
            split.iloc[idx] = "train"
    data["split"] = split

    classes = set(data["dx"])
    for name in SPLIT_NAMES:
        present = set(data.loc[data["split"] == name, "dx"])
        if present != classes:
            raise ValueError(f"split '{name}' lacks classes: {sorted(classes - present)}")

    assert data["split"].notna().all()
    return data[SPLIT_COLUMNS]


def split_frames(splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames from a split table."""
    return tuple(  # type: ignore[return-value]
        splits[splits["split"] == name].reset_index(drop=True) for name in SPLIT_NAMES
    )


def load_or_create_splits(
    df: pd.DataFrame, path: str | Path, seed: int = 42, overwrite: bool = False
) -> pd.DataFrame:
    """
    Load the audited split CSV at ``path`` if it exists, else create and save it.

    An existing CSV is rejected (``ValueError``) if it does not describe exactly
    the images, lesions and labels of ``df`` or if it leaks lesions across splits,
    so a stale or hand-edited file can never be used silently.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        saved = pd.read_csv(path, dtype=str)
        _check_saved(saved, df)
        return saved
    out = make_lesion_split(df, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _check_saved(saved: pd.DataFrame, df: pd.DataFrame) -> None:
    if list(saved.columns) != SPLIT_COLUMNS:
        raise ValueError(f"splits CSV must have columns {SPLIT_COLUMNS}")
    cols = ["image_id", "lesion_id", "dx"]
    ref = df[cols].astype(str).sort_values("image_id").reset_index(drop=True)
    cur = saved[cols].astype(str).sort_values("image_id").reset_index(drop=True)
    if not ref.to_numpy().tolist() == cur.to_numpy().tolist():
        raise ValueError("splits CSV does not match the metadata; re-create with overwrite=True")
    if not set(saved["split"]) <= set(SPLIT_NAMES):
        raise ValueError("splits CSV has unknown split names")
    lesion_splits = saved.groupby("lesion_id")["split"].nunique()
    if (lesion_splits > 1).any():
        raise ValueError("splits CSV leaks lesion_id across splits")
