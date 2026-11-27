"""Downstream evaluation: patient-level splits, bootstrap and segmentation decoder."""

import numpy as np
import pandas as pd
import torch

from downstream.engine import _segmentation_counts, dice_loss, mean_dice
from downstream.models import MultiScaleSegDecoder
from downstream.significance import holm_bonferroni, paired_bootstrap
from downstream.splits import check_train_val_test, patient_holdout, patient_kfold


def test_patient_kfold_never_splits_a_patient() -> None:
    groups = np.repeat(np.arange(40), 2)
    labels = np.repeat(np.arange(40) % 2, 2)
    for stratify in (labels, None):
        folds = patient_kfold(groups, stratify, n_splits=5, seed=0)
        assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(80))
        for train, test in folds:
            assert not set(groups[train]) & set(groups[test])
            inner_train, val = patient_holdout(train, groups, stratify, 0.15, seed=1)
            assert not set(groups[inner_train]) & set(groups[val])
            assert set(inner_train) | set(val) == set(train)
            check_train_val_test(inner_train, val, test, groups)


def test_bootstrap_of_identical_models_is_null() -> None:
    frame = pd.DataFrame(
        {"image_id": [f"im{i}" for i in range(30)], "patient_id": [f"p{i // 2}" for i in range(30)],
         "fold": [i % 5 for i in range(30)], "label": [i % 2 for i in range(30)], "prediction": [(i // 3) % 2 for i in range(30)]}
    )
    result = paired_bootstrap({"task": "DT2", "metric": "balanced_accuracy", "type": "classification"}, frame, frame.copy(), n_boot=200, seed=0)
    assert result["difference_a_minus_b"] == 0.0
    assert result["p_value"] == 1.0


def test_multiscale_decoder_resnet_and_vit_shapes() -> None:
    resnet_decoder = MultiScaleSegDecoder(2048, {"layer3": 1024, "layer2": 512}, {}, output_size=224)
    logits = resnet_decoder(torch.randn(2, 2048, 7, 7), {"layer3": torch.randn(2, 1024, 14, 14), "layer2": torch.randn(2, 512, 28, 28)}, {})
    assert logits.shape == (2, 1, 224, 224)

    vit_decoder = MultiScaleSegDecoder(768, {}, {"block_4": 768}, output_size=224)
    logits = vit_decoder(torch.randn(2, 768, 14, 14), {}, {"block_4": torch.randn(2, 768, 14, 14)})
    assert logits.shape == (2, 1, 224, 224)


def test_dice_is_thresholded_per_image_and_averaged() -> None:
    masks = torch.zeros(2, 1, 8, 8)
    masks[:, :, :4] = 1.0
    perfect_logits = (masks * 2 - 1) * 50.0
    assert float(dice_loss(perfect_logits, masks)) < 1e-4
    # Image 0: probability 0.6 on the foreground -> binarized to a perfect prediction (Dice 1).
    # Image 1: probability 0.4 on the foreground -> binarized to an empty prediction (Dice 0).
    probs = masks.clone()
    probs[0] *= 0.6
    probs[1] *= 0.4
    counts = _segmentation_counts(probs, masks)
    assert counts["dice"].tolist() == [1.0, 0.0]
    assert abs(mean_dice(counts["dice"]) - 0.5) < 1e-6
    # Empty prediction on an empty mask scores 1.
    assert _segmentation_counts(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))["dice"].tolist() == [1.0]


def test_bootstrap_p_value_and_fold_mean_difference() -> None:
    rows = 40
    frame_b = pd.DataFrame(
        {"image_id": [f"im{i}" for i in range(rows)], "patient_id": [f"p{i}" for i in range(rows)],
         "fold": [i % 2 for i in range(rows)], "label": [i % 2 for i in range(rows)], "prediction": [0] * rows}
    )
    frame_a = frame_b.assign(prediction=frame_b["label"])  # A is perfect, B always predicts class 0
    result = paired_bootstrap({"task": "DT2", "metric": "balanced_accuracy", "type": "classification"}, frame_a, frame_b, n_boot=99, seed=0)
    # Each fold contains one class only, so every fold's balanced accuracy is 1.0 (A) or 1.0 / 0.0 (B).
    expected = np.mean([1.0, 1.0]) - np.mean([1.0, 0.0])
    assert abs(result["difference_a_minus_b"] - expected) < 1e-9
    # Every resampled difference is > 0, so b = 0 and p = 2 / (B + 1).
    assert abs(result["p_value"] - 2.0 / 100.0) < 1e-12


def test_holm_bonferroni() -> None:
    # Sorted: 0.01 * 3 = 0.03; 0.02 * 2 = 0.04; 0.04 * 1 = 0.04 (monotone, capped at 1).
    assert np.allclose(holm_bonferroni([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert np.allclose(holm_bonferroni([0.5, 0.6]), [1.0, 1.0])
    # Minimum attainable value of the paper: 7 tasks, B = 10,000 -> 7 * 2 / 10,001 = 0.0014.
    assert abs(holm_bonferroni([2 / 10_001] * 7)[0] - 0.0014) < 1e-4
