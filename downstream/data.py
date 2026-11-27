"""Manifest-backed downstream datasets and preprocessing.

Every task reads one CSV manifest (built by `downstream/prepare_manifests.py`) with the columns

    image_path   path to the radiograph (absolute, or relative to the manifest's folder)
    patient_id   patient identifier; all CV splits are patient-level
    label        class index (classification) or target value (regression)
    mask_path    binary mask PNG (segmentation only; same resolution as the image)

Radiographs are converted to grayscale, resized to `image_size` x `image_size` and normalized per
image (zero mean, unit variance).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def per_image_normalize(image: torch.Tensor) -> torch.Tensor:
    return (image - image.mean()) / (image.std() + 1e-6)


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(per_image_normalize),
        ]
    )


def mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


def augment_image_mask_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Training-only, mask-synchronized augmentation used for the segmentation tasks.

    Horizontal flip and a small rotation are applied identically to image and mask; brightness and
    contrast jitter touch the image only. No vertical flip (radiographs have an up/down orientation).
    """
    if random.random() < 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    angle = random.uniform(-10.0, 10.0)
    image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
    mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
    if random.random() < 0.5:
        image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
    return image, mask


class ManifestDataset(Dataset):
    """Images (+ labels or masks) listed in a task manifest.

    `__getitem__` returns `(image, target)`: a class index (long), a regression value (float) or a
    binary mask `[1, H, W]` (float). Segmentation samples whose index is in `augment_indices` are
    augmented with `augment_image_mask_pair`; all others are deterministic.
    """

    def __init__(self, manifest_path: str | Path, task_type: str, image_size: int) -> None:
        if task_type not in {"classification", "regression", "segmentation"}:
            raise ValueError(f"Unknown task type {task_type!r}.")
        self.manifest_path = Path(manifest_path)
        self.task_type = task_type
        self.frame = pd.read_csv(self.manifest_path)
        required = {"image_path", "patient_id"} | ({"mask_path"} if task_type == "segmentation" else {"label"})
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{self.manifest_path} is missing columns {sorted(missing)}.")
        self.frame["patient_id"] = self.frame["patient_id"].astype(str)
        self.image_transform = image_transform(image_size)
        self.mask_transform = mask_transform(image_size)
        self.augment_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve(self, path: str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.manifest_path.parent / path

    @property
    def image_ids(self) -> np.ndarray:
        return self.frame["image_path"].astype(str).to_numpy()

    @property
    def patient_ids(self) -> np.ndarray:
        return self.frame["patient_id"].to_numpy()

    @property
    def labels(self) -> np.ndarray | None:
        return None if self.task_type == "segmentation" else self.frame["label"].to_numpy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(self._resolve(row["image_path"])).convert("L")
        if self.task_type == "segmentation":
            mask = Image.open(self._resolve(row["mask_path"])).convert("L")
            if index in self.augment_indices:
                image, mask = augment_image_mask_pair(image, mask)
            return self.image_transform(image), (self.mask_transform(mask) > 0.5).float()
        if self.task_type == "classification":
            target = torch.tensor(int(row["label"]), dtype=torch.long)
        else:
            target = torch.tensor(float(row["label"]), dtype=torch.float32)
        return self.image_transform(image), target
