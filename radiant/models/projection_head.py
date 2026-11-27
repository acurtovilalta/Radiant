"""Shared MLP projection head and embedding normalization."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Small MLP projection head from pooled backbone features to the embedding space."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project pooled features into an embedding space."""
        return self.net(features)


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """L2-normalize an embedding tensor along the feature dimension."""
    return F.normalize(embedding, dim=-1, eps=1e-6)
