"""Learned MRI 3D-to-Kx2D projection bridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from .projection_head import normalize_embedding


@dataclass(slots=True)
class MRIProjectionBridgeConfig:
    """Configuration for MRIProjectionBridge."""

    in_channels: int
    num_projections: int = 4
    attention_temperature: float = 1.0
    # "learned_attention" (default): K learnable depth-attention queries, as below.
    # "global_average" (ablation A1): a fixed uniform (1/D) weighting over depth, i.e. plain global
    # average pooling, always a single projection (K=1) regardless of num_projections. The
    # bridge-dependent losses (projection MIL, diversity) stay wired up on that single embedding.
    pooling_mode: str = "learned_attention"


@dataclass(slots=True)
class MRIProjection2DHeadConfig:
    """Configuration for MRIProjection2DHead."""

    in_channels: int
    embedding_dim: int = 256


class MRIProjectionBridge(nn.Module):
    """Project 3D MRI feature maps into K learned 2D feature projections.

    Input:
        F_m: [B, C, D, H, W]
    Outputs:
        P: [B, K, C, H, W]
        A: [B, K, D]
    """

    def __init__(self, config: MRIProjectionBridgeConfig) -> None:
        super().__init__()
        if int(config.in_channels) <= 0:
            raise ValueError(f"in_channels must be positive, got {config.in_channels}.")
        if int(config.num_projections) <= 0:
            raise ValueError(f"num_projections must be positive, got {config.num_projections}.")
        if float(config.attention_temperature) <= 0.0:
            raise ValueError(
                "attention_temperature must be > 0.0, "
                f"got {config.attention_temperature}."
            )
        if config.pooling_mode not in {"learned_attention", "global_average"}:
            raise ValueError(
                "pooling_mode must be 'learned_attention' or 'global_average', "
                f"got {config.pooling_mode!r}."
            )

        self.config = config
        self.in_channels = int(config.in_channels)
        self.attention_temperature = float(config.attention_temperature)
        self.pooling_mode = str(config.pooling_mode)

        if self.pooling_mode == "global_average":
            # Global average pooling has no learned notion of "K distinct
            # projections" -- every projection would be the same uniform average,
            # so this mode always collapses to a single embedding (K=1) regardless
            # of config.num_projections, and there are no depth_queries to learn.
            self.num_projections = 1
            self.depth_queries = None
        else:
            self.num_projections = int(config.num_projections)
            # K learnable depth-attention queries over per-slice channel descriptors.
            self.depth_queries = nn.Parameter(
                torch.randn(self.num_projections, self.in_channels) * (1.0 / max(1, self.in_channels) ** 0.5)
            )

    def forward(self, mri_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute K soft depth projections from 3D MRI features."""
        if mri_features.ndim != 5:
            raise ValueError(
                "mri_features must have shape [B, C, D, H, W], "
                f"got {tuple(mri_features.shape)}."
            )
        batch_size, channels, depth, _, _ = mri_features.shape
        if channels != self.in_channels:
            raise ValueError(
                "Channel mismatch for MRIProjectionBridge: "
                f"expected C={self.in_channels}, got C={channels}."
            )
        if depth <= 0:
            raise ValueError("Depth dimension D must be positive.")

        if self.pooling_mode == "global_average":
            # Fixed uniform weighting over depth: [B, 1, D]. Not a function of
            # mri_features -- deliberately not learned, so this is a "clean" GAP
            # baseline rather than an attention mechanism that merely starts
            # near-uniform.
            attention_weights = mri_features.new_full((batch_size, 1, depth), 1.0 / float(depth))
            projected_features = mri_features.mean(dim=2, keepdim=False).unsqueeze(1)
            if projected_features.shape != (batch_size, 1, channels, mri_features.shape[3], mri_features.shape[4]):
                raise RuntimeError(
                    "Internal shape error while building GAP-projected features: "
                    f"got {tuple(projected_features.shape)}."
                )
            return projected_features, attention_weights

        if self.num_projections > depth:
            diversity_floor = (self.num_projections / depth - 1) / (self.num_projections - 1)
            raise ValueError(
                "num_projections (K) must not exceed the depth dimension (D): "
                f"K={self.num_projections}, D={depth}. With K > D, the attention "
                f"diversity floor (K/D - 1)/(K - 1) = {diversity_floor:.4f} forces "
                "degenerate depth-attention overlap."
            )

        # [B, D, C]: per-depth slice tokens from spatially pooled feature maps.
        depth_tokens = mri_features.mean(dim=(3, 4)).permute(0, 2, 1)
        logits = torch.einsum("bdc,kc->bkd", depth_tokens, self.depth_queries)
        attention_weights = torch.softmax(logits / self.attention_temperature, dim=-1)

        if attention_weights.shape != (batch_size, self.num_projections, depth):
            raise RuntimeError(
                "Internal shape error while building attention weights: "
                f"expected {(batch_size, self.num_projections, depth)}, "
                f"got {tuple(attention_weights.shape)}."
            )

        # Weighted sum along depth: [B, K, C, H, W].
        projected_features = torch.einsum("bkd,bcdhw->bkchw", attention_weights, mri_features)
        if projected_features.shape != (batch_size, self.num_projections, channels, mri_features.shape[3], mri_features.shape[4]):
            raise RuntimeError(
                "Internal shape error while building projected features: "
                f"got {tuple(projected_features.shape)}."
            )
        return projected_features, attention_weights


class MRIProjection2DHead(nn.Module):
    """Shared 2D head for K MRI projections.

    Input:
        projected_features: [B, K, C, H, W]
    Output:
        z_m_proj: [B, K, d]
    """

    def __init__(self, config: MRIProjection2DHeadConfig) -> None:
        super().__init__()
        if int(config.in_channels) <= 0:
            raise ValueError(f"in_channels must be positive, got {config.in_channels}.")
        if int(config.embedding_dim) <= 0:
            raise ValueError(f"embedding_dim must be positive, got {config.embedding_dim}.")

        self.config = config
        self.in_channels = int(config.in_channels)
        self.embedding_dim = int(config.embedding_dim)

        self.conv1x1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(self.in_channels, self.embedding_dim)

    def forward(self, projected_features: torch.Tensor) -> torch.Tensor:
        if projected_features.ndim != 5:
            raise ValueError(
                "projected_features must have shape [B, K, C, H, W], "
                f"got {tuple(projected_features.shape)}."
            )
        batch_size, num_projections, channels, _, _ = projected_features.shape
        if channels != self.in_channels:
            raise ValueError(
                "Channel mismatch for MRIProjection2DHead: "
                f"expected C={self.in_channels}, got C={channels}."
            )

        flattened = projected_features.reshape(batch_size * num_projections, channels, projected_features.shape[3], projected_features.shape[4])
        hidden = self.conv1x1(flattened)
        hidden = self.bn(hidden)
        hidden = self.relu(hidden)
        pooled = self.pool(hidden).flatten(1)
        embeddings = self.projection(pooled)
        output = embeddings.reshape(batch_size, num_projections, self.embedding_dim)
        if output.shape != (batch_size, num_projections, self.embedding_dim):
            raise RuntimeError(
                "Internal shape error while building MRI projection embeddings: "
                f"got {tuple(output.shape)}."
            )
        return normalize_embedding(output)
