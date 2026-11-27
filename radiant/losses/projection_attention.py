"""Diversity regularizer for the MRI projection bridge's depth attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def projection_attention_regularization(mri_depth_attn: torch.Tensor) -> dict[str, torch.Tensor]:
    """Penalize redundant depth-attention distributions across the K projection queries.

    `diversity_loss` is the mean squared cosine similarity between the attention distributions of
    every pair of distinct queries (0 when all K distributions are orthogonal); `pairwise_cosine`
    is the mean (unsquared) cosine, logged as a diagnostic.
    """
    if mri_depth_attn.ndim != 3:
        raise ValueError(
            f"mri_depth_attn must have shape [B, K, D], got {tuple(mri_depth_attn.shape)}."
        )
    num_heads = mri_depth_attn.shape[1]
    eps = 1e-8

    attn = mri_depth_attn.clamp_min(eps)
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    normalized = F.normalize(attn, p=2, dim=-1, eps=eps)
    cosine = torch.matmul(normalized, normalized.transpose(1, 2))

    if num_heads < 2:
        zero = cosine.sum() * 0.0
        return {"diversity_loss": zero, "pairwise_cosine": zero}
    off_diagonal_mask = ~torch.eye(num_heads, device=cosine.device, dtype=torch.bool).unsqueeze(0)
    off_diagonal_cosine = cosine.masked_select(off_diagonal_mask)
    return {
        "diversity_loss": off_diagonal_cosine.square().mean(),
        "pairwise_cosine": off_diagonal_cosine.mean(),
    }
