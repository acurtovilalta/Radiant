"""Patient-level cross-validation splits.

Outer folds: K-fold over patients (stratified by class for classification). Inside each training
fold, a patient-level validation partition is held out for early stopping and hyperparameter
selection; the test fold is only used once, for the selected model.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedGroupKFold


def _group_kfold(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_groups, test_groups in splitter.split(unique_groups):
        test_mask = np.isin(groups, unique_groups[test_groups])
        folds.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return folds


def patient_kfold(
    groups: np.ndarray, labels: np.ndarray | None, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_val_indices, test_indices) per fold; stratified when `labels` is given."""
    groups = np.asarray(groups)
    if labels is None:
        return _group_kfold(groups, n_splits, seed)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.zeros(len(groups)), np.asarray(labels), groups)]


def patient_holdout(
    indices: np.ndarray, groups: np.ndarray, labels: np.ndarray | None, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split `indices` into (train, val) with about `val_fraction` of the patients in val."""
    n_splits = max(2, int(round(1.0 / val_fraction)))
    sub_groups = np.asarray(groups)[indices]
    sub_labels = None if labels is None else np.asarray(labels)[indices]
    train_pos, val_pos = patient_kfold(sub_groups, sub_labels, n_splits, seed)[0]
    return indices[train_pos], indices[val_pos]


def check_train_val_test(train: np.ndarray, val: np.ndarray, test: np.ndarray, groups: np.ndarray) -> None:
    """Raise if any of the three partitions is empty or if they share an image or a patient."""
    groups = np.asarray(groups)
    parts = {"train": train, "val": val, "test": test}
    for name, idx in parts.items():
        if len(idx) == 0:
            raise ValueError(f"The {name} partition is empty.")
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if set(parts[a].tolist()) & set(parts[b].tolist()):
                raise ValueError(f"The {a} and {b} partitions share images.")
            if set(groups[parts[a]]) & set(groups[parts[b]]):
                raise ValueError(f"The {a} and {b} partitions share patients.")
