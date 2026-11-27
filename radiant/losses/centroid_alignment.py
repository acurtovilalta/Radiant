"""Cross-modal prototype-centroid alignment.

For every anatomy family present in both the X-ray and the MRI batch, each prototype's centroid is
computed per modality as the code-weighted mean of that modality's embeddings. The loss pulls the
X-ray and MRI centroids of the same prototype together, so that a prototype means the same anatomy
in both modalities. Prototypes with (almost) no assigned mass in either modality are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .swav import SwAVAssignmentOutput


@dataclass(slots=True)
class CentroidAlignmentConfig:
    """`centroid_alignment` block of the configs (its weight is `lambda_centroid_alignment`)."""

    normalize_centroids: bool = True  # L2-normalize centroids before comparing them
    min_prototype_mass: float = 1e-4  # prototypes with less assigned mass in either modality are skipped
    distance: str = "mse"  # "mse" or "l1"


def _masked_codes(assignments: torch.Tensor, counts: torch.Tensor, prototype_count: int) -> torch.Tensor:
    """First `prototype_count` code columns, with each sample's padding slots zeroed."""
    mask = torch.arange(prototype_count, device=assignments.device).unsqueeze(0) < counts.unsqueeze(1)
    return assignments[:, :prototype_count] * mask.to(assignments.dtype)


def _centroids(embeddings: torch.Tensor, codes: torch.Tensor, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Code-weighted mean embedding per prototype ([P, d]) and the total code mass per prototype ([P])."""
    codes = codes.float()
    mass = codes.sum(dim=0)
    centroids = (codes.t() @ embeddings.float()) / mass.unsqueeze(1).clamp_min(1e-12)
    if normalize:
        centroids = F.normalize(centroids, dim=-1, eps=1e-6)
    return centroids, mass


def centroid_alignment_loss(
    xray_embeddings: torch.Tensor,
    xray_assignments: SwAVAssignmentOutput,
    mri_embeddings: torch.Tensor,
    mri_assignments: SwAVAssignmentOutput,
    config: CentroidAlignmentConfig,
) -> torch.Tensor:
    """Mean over shared anatomy families of the masked centroid distance (0 if no family is shared)."""
    if config.distance not in ("mse", "l1"):
        raise ValueError(f"Unsupported centroid distance '{config.distance}'. Use 'mse' or 'l1'.")

    def indices_by_family(family_names: list[str]) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = {}
        for index, family in enumerate(family_names):
            grouped.setdefault(family, []).append(index)
        return grouped

    xray_indices = indices_by_family(xray_assignments.family_names)
    mri_indices = indices_by_family(mri_assignments.family_names)
    shared_families = sorted(set(xray_indices) & set(mri_indices))
    if not shared_families:
        return xray_embeddings.new_tensor(0.0)

    family_losses = []
    for family in shared_families:
        x_idx = torch.tensor(xray_indices[family], dtype=torch.long, device=xray_embeddings.device)
        m_idx = torch.tensor(mri_indices[family], dtype=torch.long, device=mri_embeddings.device)
        x_counts = xray_assignments.family_prototype_counts.index_select(0, x_idx)
        m_counts = mri_assignments.family_prototype_counts.index_select(0, m_idx)
        # All samples of one family share one bank, so their counts are identical.
        prototype_count = int(torch.minimum(x_counts[0], m_counts[0]).item())

        x_codes = _masked_codes(xray_assignments.family_assignments.index_select(0, x_idx), x_counts, prototype_count)
        m_codes = _masked_codes(mri_assignments.family_assignments.index_select(0, m_idx), m_counts, prototype_count)
        x_centroids, x_mass = _centroids(xray_embeddings.index_select(0, x_idx), x_codes, config.normalize_centroids)
        m_centroids, m_mass = _centroids(mri_embeddings.index_select(0, m_idx), m_codes, config.normalize_centroids)

        # Mean distance over the valid prototypes only, written without boolean indexing
        # (which would force a device sync); it is 0 when no prototype is valid.
        valid = ((x_mass >= config.min_prototype_mass) & (m_mass >= config.min_prototype_mass)).to(x_centroids.dtype)
        diff = x_centroids - m_centroids
        per_element = diff.square() if config.distance == "mse" else diff.abs()
        denominator = (valid.sum() * x_centroids.shape[-1]).clamp_min(1.0)
        family_losses.append((per_element * valid.unsqueeze(-1)).sum() / denominator)

    return torch.stack(family_losses).mean()
