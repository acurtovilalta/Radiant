"""Evaluate a frozen RADIANT X-ray encoder on one downstream task (DT1-DT7).

Protocol (per task config in `downstream/configs/`):
  - patient-level K-fold cross-validation (stratified by class for classification); every fold
    has three patient-disjoint partitions: train, val (held out of the training folds) and test;
  - inside each training fold, a patient-level validation partition is held out; a random
    hyperparameter search trains one head per trial on the rest, early-stopping on validation,
    and the trial with the best validation score is evaluated once on the test fold;
  - the trial candidates are drawn from the task's seeded search space, so every encoder
    evaluated on a task sees the same candidates.

Outputs in `--out-dir`:
  predictions.csv   one row per image: image_id, patient_id, fold, label + prediction, or Dice + pixel counts
  folds.csv         per-fold test metric and the selected hyperparameters
  summary.json      mean +/- SD (population SD) over folds (the reported values) and the pooled out-of-fold metric

Usage:
    python -m downstream.evaluate --task downstream/configs/dt1_bone_tumor_malignancy.yaml \
        --checkpoint outputs/radiant/resnet50_scratch/checkpoints/final.pt \
        --out-dir outputs/downstream/radiant_resnet50_scratch/dt1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import ManifestDataset
from .encoder import FrozenXRayEncoder
from .engine import (
    balanced_accuracy,
    extract_pooled_features,
    mean_absolute_error,
    mean_dice,
    predict_linear_head,
    predict_segmentation,
    train_linear_head,
    train_segmentation,
)
from .splits import check_train_val_test, patient_holdout, patient_kfold


def load_task(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_trials(search: dict, seed: int) -> list[dict]:
    """`n_trials` random draws from the search space (same draws for every encoder)."""
    rng = np.random.default_rng(seed)
    keys = [key for key in search if key != "n_trials"]
    return [{key: rng.choice(search[key]).item() for key in keys} for _ in range(int(search["n_trials"]))]


def task_metric(task: dict, frame: pd.DataFrame) -> float:
    """The task's reported metric on a set of out-of-fold predictions."""
    if task["type"] == "classification":
        return balanced_accuracy(frame["label"].to_numpy(), frame["prediction"].to_numpy())
    if task["type"] == "regression":
        return mean_absolute_error(frame["label"].to_numpy(), frame["prediction"].to_numpy())
    return mean_dice(frame["dice"].to_numpy())


def evaluate(task: dict, checkpoint: str | Path, out_dir: Path, device: torch.device, max_folds: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(task.get("seed", 123))
    set_seed(seed)

    dataset = ManifestDataset(task["manifest"], task["type"], int(task["image_size"]))
    encoder = FrozenXRayEncoder(checkpoint).to(device)
    groups = dataset.patient_ids
    stratify_labels = dataset.labels if task["type"] == "classification" else None
    folds = patient_kfold(groups, stratify_labels, int(task.get("n_folds", 5)), seed)
    trials = sample_trials(task["search"], seed)

    features = targets = None
    if task["type"] != "segmentation":
        features, targets = extract_pooled_features(encoder, dataset, device, num_workers=int(task.get("num_workers", 8)))

    prediction_frames, fold_rows = [], []
    for fold, (train_val_idx, test_idx) in enumerate(folds):
        if max_folds is not None and fold >= max_folds:
            break
        train_idx, val_idx = patient_holdout(train_val_idx, groups, stratify_labels, float(task.get("val_fraction", 0.15)), seed + fold)
        check_train_val_test(train_idx, val_idx, test_idx, groups)
        print(f"[{task['task']}] fold {fold + 1}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} images", flush=True)
        best, best_trial = None, None
        for trial_index, hparams in enumerate(trials):
            set_seed(seed + 1000 * fold + trial_index)
            if task["type"] == "segmentation":
                result = train_segmentation(encoder, dataset, train_idx, val_idx, task, hparams, device)
            else:
                result = train_linear_head(features, targets, train_idx, val_idx, task, hparams, device)
            print(f"[{task['task']}] fold {fold + 1} trial {trial_index + 1}/{len(trials)} {hparams} val={result.val_score:.4f}", flush=True)
            if best is None or result.val_score > best.val_score:
                best, best_trial = result, hparams

        if task["type"] == "segmentation":
            outputs = predict_segmentation(best, dataset, test_idx, task, device)
        else:
            outputs = predict_linear_head(best, features[test_idx], task, device)
        frame = pd.DataFrame({"image_id": dataset.image_ids[test_idx], "patient_id": groups[test_idx], "fold": fold})
        if task["type"] != "segmentation":
            frame["label"] = targets[test_idx].numpy()
        for key, values in outputs.items():
            frame[key] = values
        prediction_frames.append(frame)
        fold_rows.append({"fold": fold, task["metric"]: task_metric(task, frame), "val_score": best.val_score, **best_trial, **best.extras})
        print(f"[{task['task']}] fold {fold + 1}: test {task['metric']} = {fold_rows[-1][task['metric']]:.4f}", flush=True)
        if task["type"] == "segmentation":
            del best  # free the decoder before the next fold

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(out_dir / "folds.csv", index=False)
    values = fold_table[task["metric"]].to_numpy()
    summary = {
        "task": task["task"],
        "name": task["name"],
        "checkpoint": str(checkpoint),
        "metric": task["metric"],
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=0)),
        "pooled_oof": task_metric(task, predictions),
        "n_folds": len(values),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Task config, e.g. downstream/configs/dt1_bone_tumor_malignancy.yaml.")
    parser.add_argument("--checkpoint", required=True, help="RADIANT pretraining checkpoint (final.pt, the final epoch).")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-folds", type=int, default=None, help="Only run the first N folds (smoke tests).")
    args = parser.parse_args()
    evaluate(load_task(args.task), args.checkpoint, args.out_dir, torch.device(args.device), args.max_folds)


if __name__ == "__main__":
    main()
