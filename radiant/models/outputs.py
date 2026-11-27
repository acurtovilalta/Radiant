"""Shared output container for X-ray and MRI encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class EncoderOutput:
    """Container for encoder features and the L2-normalized embedding."""

    feature_map: torch.Tensor
    pooled_feature: torch.Tensor
    embedding: torch.Tensor  # L2-normalized projector output, used by SwAV and all cross-modal losses
    intermediate_feature_maps: dict[str, torch.Tensor]
    projection: torch.Tensor  # raw (unnormalized) projector output
    mri_proj_feats: torch.Tensor | None = None
    mri_proj_embs: torch.Tensor | None = None
    mri_depth_attn: torch.Tensor | None = None
