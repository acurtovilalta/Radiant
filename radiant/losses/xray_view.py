"""X-ray view/plane prediction head and loss, used in the warm-up stage only."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def normalize_metadata_label(value: Any) -> str:
    """Normalize a metadata label into a stable, case-insensitive token."""
    if value is None:
        return ""
    return str(value).strip().lower()


def encode_categorical_labels(labels: list[Any], mapping: dict[str, int], device: torch.device) -> torch.Tensor:
    """Map string labels to integer ids, using -1 for missing/unknown labels."""
    if not labels:
        return torch.empty(0, dtype=torch.long, device=device)
    encoded = [mapping.get(normalize_metadata_label(label), -1) for label in labels]
    return torch.tensor(encoded, dtype=torch.long, device=device)


class XRayViewHead(nn.Module):
    """Two-layer MLP predicting the X-ray view/plane from the X-ray embedding.

    With `num_classes == 0` (no view labels in the manifest) the head is empty and the loss is 0.
    """

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        hidden = max(64, int(embedding_dim))
        self.network = (
            nn.Sequential(nn.Linear(int(embedding_dim), hidden), nn.GELU(), nn.Linear(hidden, self.num_classes))
            if self.num_classes > 0
            else None
        )

    def forward(self, embedding: torch.Tensor, view_labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over samples with a known view (label -1 = unknown, ignored)."""
        if self.network is None:
            return embedding.sum() * 0.0
        if embedding.ndim != 2:
            raise ValueError(f"embedding must be [batch, dim], got {tuple(embedding.shape)}.")
        valid = view_labels >= 0
        if not torch.any(valid):
            return embedding.sum() * 0.0
        return F.cross_entropy(self.network(embedding[valid]), view_labels[valid].long())
