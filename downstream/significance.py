"""Paired bootstrap over patients between two encoders evaluated on the same task.

Both models' out-of-fold predictions (`predictions.csv` from `downstream.evaluate`) must cover the
same images with the same fold assignment; this is checked.

  - Delta is the difference in fold-mean performance, A - B: the task metric is computed on each
    test fold and averaged over the folds, as in the reported mean over five folds.
  - Each bootstrap resample draws patients with replacement within every test fold (the same draw
    for both models, so the comparison is paired), recomputes both models' fold-mean metric and
    takes the difference.
  - Reported: the observed Delta, the 95% percentile interval (unadjusted) and the two-sided
    p = 2 (b + 1) / (B + 1), capped at 1, where B is the number of resamples and b the number of
    resampled differences on the other side of zero (b = min(#{d* <= 0}, #{d* >= 0})).

Holm-Bonferroni correction is applied afterwards, across the tasks of a comparison family
(e.g. RADIANT vs. RADIANT-X on DT1-DT7), with `--holm` on the per-task result files.

Usage:
    # one task
    python -m downstream.significance --task downstream/configs/dt1_bone_tumor_malignancy.yaml \
        --a outputs/downstream/radiant_resnet50_scratch/dt1 \
        --b outputs/downstream/radiant_x_resnet50_scratch/dt1 \
        --out outputs/downstream/significance/radiant_vs_radiant_x/dt1.json

    # Holm-Bonferroni over one comparison family (adds `p_holm` to every file, prints a table)
    python -m downstream.significance --holm outputs/downstream/significance/radiant_vs_radiant_x/dt*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _weighted_metric(task: dict, frame: pd.DataFrame, weights: np.ndarray) -> float:
    """Task metric on a set of images, where `weights[i]` is how often row i was drawn."""
    if task["type"] == "classification":
        labels, correct = frame["label"].to_numpy(), (frame["label"] == frame["prediction"]).to_numpy()
        recalls = [
            np.sum(weights * correct * (labels == c)) / np.sum(weights * (labels == c))
            for c in np.unique(labels)
            if np.sum(weights * (labels == c)) > 0
        ]
        return float(np.mean(recalls))
    if task["type"] == "regression":
        return float(np.sum(weights * np.abs(frame["label"] - frame["prediction"])) / np.sum(weights))
    # Segmentation: mean of the per-image Dice scores.
    return float(np.sum(weights * frame["dice"].to_numpy()) / np.sum(weights))


def _fold_mean(task: dict, folds: list[pd.DataFrame], fold_weights: list[np.ndarray]) -> float:
    return float(np.mean([_weighted_metric(task, frame, weights) for frame, weights in zip(folds, fold_weights)]))


def paired_bootstrap(task: dict, frame_a: pd.DataFrame, frame_b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    frame_a = frame_a.sort_values("image_id").reset_index(drop=True)
    frame_b = frame_b.sort_values("image_id").reset_index(drop=True)
    if not frame_a["image_id"].equals(frame_b["image_id"]):
        raise ValueError("The two models were not evaluated on the same images.")
    if not frame_a["fold"].equals(frame_b["fold"]):
        raise ValueError("The two models' fold assignments differ.")

    fold_ids = sorted(frame_a["fold"].unique())
    folds_a = [frame_a[frame_a["fold"] == fold].reset_index(drop=True) for fold in fold_ids]
    folds_b = [frame_b[frame_b["fold"] == fold].reset_index(drop=True) for fold in fold_ids]
    # Patients of each test fold and, for every row, the index of its patient within the fold.
    fold_patients = [
        np.unique(frame["patient_id"].astype(str).to_numpy(), return_inverse=True) for frame in folds_a
    ]

    ones = [np.ones(len(frame)) for frame in folds_a]
    observed = _fold_mean(task, folds_a, ones) - _fold_mean(task, folds_b, ones)

    rng = np.random.default_rng(seed)
    differences = np.empty(n_boot)
    for index in range(n_boot):
        weights = []
        for patients, row_patient in fold_patients:
            counts = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
            weights.append(counts[row_patient].astype(np.float64))
        differences[index] = _fold_mean(task, folds_a, weights) - _fold_mean(task, folds_b, weights)

    b = int(min(np.sum(differences <= 0.0), np.sum(differences >= 0.0)))
    p_value = min(1.0, 2.0 * (b + 1) / (n_boot + 1))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "task": task["task"],
        "metric": task["metric"],
        "difference_a_minus_b": float(observed),
        "ci95": [float(low), float(high)],
        "p_value": float(p_value),
        "n_boot": int(n_boot),
        "n_folds": len(fold_ids),
        "n_patients": int(sum(len(patients) for patients, _ in fold_patients)),
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    With the m p-values sorted ascending, p_(i) is adjusted to max_{j <= i} min(1, (m - j + 1) p_(j)).
    """
    m = len(p_values)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def apply_holm(paths: list[Path]) -> list[dict]:
    """Add `p_holm` to the per-task result files of one comparison family (rewritten in place)."""
    results = [json.loads(path.read_text()) for path in paths]
    for result, path, p_holm in zip(results, paths, holm_bonferroni([r["p_value"] for r in results])):
        result["p_holm"] = p_holm
        result["holm_family_size"] = len(results)
        path.write_text(json.dumps(result, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holm", nargs="+", type=Path, default=None, metavar="RESULT_JSON",
                        help="Holm-Bonferroni over the per-task result files of one comparison family.")
    parser.add_argument("--task")
    parser.add_argument("--a", type=Path, help="Output dir of model A (downstream.evaluate).")
    parser.add_argument("--b", type=Path, help="Output dir of model B.")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.holm:
        for result in apply_holm(args.holm):
            low, high = result["ci95"]
            print(f"{result['task']}\tdelta={result['difference_a_minus_b']:+.4f} [{low:+.4f}, {high:+.4f}]"
                  f"\tp={result['p_value']:.4f}\tp_holm={result['p_holm']:.4f}")
        return
    if not (args.task and args.a and args.b):
        parser.error("--task, --a and --b are required (or use --holm).")

    with open(args.task) as handle:
        task = yaml.safe_load(handle)
    result = paired_bootstrap(
        task, pd.read_csv(args.a / "predictions.csv"), pd.read_csv(args.b / "predictions.csv"), args.n_boot, args.seed
    )
    result |= {"model_a": str(args.a), "model_b": str(args.b)}
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
