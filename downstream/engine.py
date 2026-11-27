"""Training and evaluation of the downstream heads on a frozen encoder.

Classification / regression: the frozen encoder's pooled features are extracted once per dataset
(there is no augmentation), then a linear head is trained on them. Segmentation: the decoder is
trained on the encoder's feature maps, computed on the fly because training images are augmented.
Every run early-stops on its validation partition and keeps the best validation epoch.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .models import build_segmenter

def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.float16) if device.type == "cuda" else nullcontext()


@dataclass
class RunResult:
    """Outcome of one training run (one hyperparameter trial on one fold)."""

    val_score: float  # selection score on the validation partition (higher is better)
    model: nn.Module
    extras: dict = field(default_factory=dict)  # e.g. regression target scaling


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(balanced_accuracy_score(labels, predictions))


def mean_absolute_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(targets) - np.asarray(predictions))))


DICE_THRESHOLD = 0.5


def mean_dice(dice: np.ndarray) -> float:
    """Dice score of a set of images: the mean of the per-image Dice scores."""
    return float(np.mean(np.asarray(dice, dtype=np.float64)))


def _segmentation_counts(probs: torch.Tensor, masks: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-image Dice of the binarized prediction (probability >= 0.5), plus its pixel counts.

    `tp`, `pred_sum` and `mask_sum` are the per-image true-positive, predicted and ground-truth
    foreground pixel counts; `dice` = 2 tp / (pred_sum + mask_sum), and 1 when the prediction and
    the mask are both empty.
    """
    predicted = (probs >= DICE_THRESHOLD).to(masks.dtype)
    tp = (predicted * masks).sum(dim=(1, 2, 3))
    pred_sum = predicted.sum(dim=(1, 2, 3))
    mask_sum = masks.sum(dim=(1, 2, 3))
    denominator = pred_sum + mask_sum
    dice = torch.where(denominator > 0, 2.0 * tp / denominator.clamp_min(1.0), torch.ones_like(denominator))
    return {
        "tp": tp.numpy(),
        "pred_sum": pred_sum.numpy(),
        "mask_sum": mask_sum.numpy(),
        "dice": dice.numpy(),
    }


# --------------------------------------------------------------------------------------
# Classification / regression on cached features
# --------------------------------------------------------------------------------------


@torch.no_grad()
def extract_pooled_features(encoder, dataset, device: torch.device, batch_size: int = 64, num_workers: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features, targets = [], []
    for images, batch_targets in loader:
        with _autocast(device):
            features.append(encoder.pooled(images.to(device, non_blocking=True)).float().cpu())
        targets.append(batch_targets)
    return torch.cat(features), torch.cat(targets)


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    return counts.sum() / (num_classes * counts)


def train_linear_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task: dict,
    hparams: dict,
    device: torch.device,
) -> RunResult:
    """Linear head with cross-entropy (classification) or MSE on standardized targets (regression).

    Early stopping and the kept epoch follow `training.checkpoint_metric`: `val_loss` or
    `val_metric` (balanced accuracy / MAE on the validation partition).
    """
    is_classification = task["type"] == "classification"
    training = task["training"]
    extras: dict = {}
    if is_classification:
        num_classes = int(task["num_classes"])
        head = nn.Linear(features.shape[1], num_classes).to(device)
        weight = _class_weights(targets[train_idx], num_classes).to(device) if task.get("class_weighting", False) else None
        criterion = nn.CrossEntropyLoss(weight=weight)
        scaled = targets
    else:
        head = nn.Linear(features.shape[1], 1).to(device)
        criterion = nn.MSELoss()
        mean = float(targets[train_idx].mean())
        std = float(targets[train_idx].std().clamp_min(1e-6))
        extras.update(target_mean=mean, target_std=std)
        scaled = (targets - mean) / std

    train_loader = DataLoader(
        TensorDataset(features[train_idx], scaled[train_idx]), batch_size=int(hparams["batch_size"]), shuffle=True
    )
    val_x, val_y = features[val_idx].to(device), scaled[val_idx].to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))

    def val_scores() -> tuple[float, float]:
        head.eval()
        with torch.no_grad():
            outputs = head(val_x)
            if is_classification:
                loss = float(criterion(outputs, val_y))
                metric = balanced_accuracy(val_y.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy())
            else:
                outputs = outputs.squeeze(1)
                loss = float(criterion(outputs, val_y))
                metric = -mean_absolute_error(
                    (val_y * extras["target_std"] + extras["target_mean"]).cpu().numpy(),
                    (outputs * extras["target_std"] + extras["target_mean"]).cpu().numpy(),
                )
        return loss, metric  # metric: higher is better (negative MAE for regression)

    use_loss = training.get("checkpoint_metric", "val_loss") == "val_loss"
    min_delta = float(training.get("min_delta", 0.0))
    best_key, best_metric, best_state, waited = float("inf"), float("-inf"), None, 0
    for _ in range(int(training["max_epochs"])):
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = head(batch_x)
            loss = criterion(outputs, batch_y) if is_classification else criterion(outputs.squeeze(1), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_loss, val_metric = val_scores()
        key = val_loss if use_loss else -val_metric
        if key < best_key - min_delta:
            best_key, best_metric, best_state, waited = key, val_metric, copy.deepcopy(head.state_dict()), 0
        else:
            waited += 1
            if waited >= int(training["patience"]):
                break
    head.load_state_dict(best_state)
    return RunResult(val_score=best_metric, model=head, extras=extras)


@torch.no_grad()
def predict_linear_head(result: RunResult, features: torch.Tensor, task: dict, device: torch.device) -> dict[str, np.ndarray]:
    head = result.model.eval()
    outputs = head(features.to(device)).float().cpu()
    if task["type"] == "classification":
        probs = torch.softmax(outputs, dim=1).numpy()
        predictions = {"prediction": probs.argmax(axis=1)}
        predictions.update({f"prob_{c}": probs[:, c] for c in range(probs.shape[1])})
        return predictions
    values = outputs.squeeze(1) * result.extras["target_std"] + result.extras["target_mean"]
    return {"prediction": values.numpy()}


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def dice_loss(logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss, 1 - Dice(sigmoid(logits), masks), per image and averaged over the batch."""
    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


@torch.no_grad()
def _segmentation_outputs(model, loader, device, loss_fn) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss, count, all_probs, all_masks = 0.0, 0, [], []
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with _autocast(device):
            logits = model(images)
        logits = logits.float()
        total_loss += float(loss_fn(logits, masks)) * masks.shape[0]
        count += masks.shape[0]
        all_probs.append(torch.sigmoid(logits).cpu())
        all_masks.append(masks.cpu())
    return total_loss / max(1, count), torch.cat(all_probs), torch.cat(all_masks)


def train_segmentation(
    encoder,
    dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task: dict,
    hparams: dict,
    device: torch.device,
) -> RunResult:
    """Decoder on the frozen encoder with soft Dice loss and augmented training images.

    `training.checkpoint_metric` is `val_loss` or `val_dice` (mean per-image Dice, threshold 0.5).
    """
    training = task["training"]
    model = build_segmenter(encoder, int(task["image_size"]), task["decoder"], device)
    loss_fn = dice_loss

    dataset.augment_indices = set(int(i) for i in train_idx) if task.get("augment", True) else set()
    loader_kwargs = {"batch_size": int(hparams["batch_size"]), "num_workers": int(task.get("num_workers", 8)), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, drop_last=len(train_idx) > int(hparams["batch_size"]), **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(decoder_params, lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    use_loss = training.get("checkpoint_metric", "val_loss") == "val_loss"
    best_key, best = float("inf"), None
    waited = 0
    for _ in range(int(training["max_epochs"])):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with _autocast(device):
                logits = model(images)
            loss = loss_fn(logits.float(), masks)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        val_loss, probs, masks = _segmentation_outputs(model, val_loader, device, loss_fn)
        val_dice = mean_dice(_segmentation_counts(probs, masks)["dice"])
        key = val_loss if use_loss else -val_dice
        if key < best_key:
            best_key, waited = key, 0
            best = (val_dice, copy.deepcopy(model.decoder.state_dict()))
        else:
            waited += 1
            if waited >= int(training["patience"]):
                break
    dataset.augment_indices = set()
    val_dice, decoder_state = best
    model.decoder.load_state_dict(decoder_state)
    return RunResult(val_score=val_dice, model=model)


@torch.no_grad()
def predict_segmentation(result: RunResult, dataset, test_idx: np.ndarray, task: dict, device: torch.device) -> dict[str, np.ndarray]:
    """Per-image Dice (threshold 0.5) and pixel counts on the test images."""
    loader = DataLoader(Subset(dataset, test_idx), batch_size=8, shuffle=False, num_workers=int(task.get("num_workers", 8)))
    _, probs, masks = _segmentation_outputs(result.model, loader, device, lambda logits, masks: logits.sum() * 0.0)
    return _segmentation_counts(probs, masks)
