from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.losses import CentroidAlignmentConfig, SwAVAssignmentOutput, centroid_alignment_loss


def _codes(family_names: list[str], assignments: torch.Tensor, counts: list[int]) -> SwAVAssignmentOutput:
    return SwAVAssignmentOutput(
        family_names=family_names,
        family_assignments=assignments,
        family_prototype_counts=torch.tensor(counts, dtype=torch.long),
        queue_sizes=torch.zeros(len(family_names), dtype=torch.long),
    )


def test_identical_modalities_give_zero_loss() -> None:
    embeddings = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    codes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = centroid_alignment_loss(
        xray_embeddings=embeddings,
        xray_assignments=_codes(["Knee", "Knee"], codes, [2, 2]),
        mri_embeddings=embeddings.clone(),
        mri_assignments=_codes(["Knee", "Knee"], codes, [2, 2]),
        config=CentroidAlignmentConfig(),
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_no_shared_family_gives_zero_loss() -> None:
    codes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = centroid_alignment_loss(
        xray_embeddings=torch.randn(2, 3),
        xray_assignments=_codes(["Knee", "Knee"], codes, [2, 2]),
        mri_embeddings=torch.randn(2, 3),
        mri_assignments=_codes(["Chest", "Chest"], codes, [2, 2]),
        config=CentroidAlignmentConfig(),
    )
    assert float(loss) == 0.0


def test_mismatched_centroids_give_positive_loss_with_gradient() -> None:
    xray = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    mri = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [1.0, 0.0, 0.0]])
    x_codes = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
    m_codes = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    loss = centroid_alignment_loss(
        xray_embeddings=xray,
        xray_assignments=_codes(["Knee"] * 3, x_codes, [2, 2, 2]),
        mri_embeddings=mri,
        mri_assignments=_codes(["Knee"] * 3, m_codes, [2, 2, 2]),
        config=CentroidAlignmentConfig(normalize_centroids=True),
    )
    loss.backward()
    assert float(loss) > 0.0
    assert xray.grad is not None and xray.grad.abs().sum() > 0


def test_prototypes_without_mass_are_ignored() -> None:
    # Prototype 1 gets no code mass in the MRI batch, so only prototype 0 is compared, and
    # its centroids coincide -> loss 0 even though the X-ray prototype-1 centroid differs.
    xray = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mri = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = centroid_alignment_loss(
        xray_embeddings=xray,
        xray_assignments=_codes(["Knee", "Knee"], torch.tensor([[1.0, 0.0], [0.0, 1.0]]), [2, 2]),
        mri_embeddings=mri,
        mri_assignments=_codes(["Knee", "Knee"], torch.tensor([[1.0, 0.0], [1.0, 0.0]]), [2, 2]),
        config=CentroidAlignmentConfig(min_prototype_mass=1e-4),
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_unknown_distance_raises() -> None:
    codes = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="Unsupported centroid distance"):
        centroid_alignment_loss(
            xray_embeddings=torch.randn(1, 2),
            xray_assignments=_codes(["Knee"], codes, [2]),
            mri_embeddings=torch.randn(1, 2),
            mri_assignments=_codes(["Knee"], codes, [2]),
            config=CentroidAlignmentConfig(distance="cosine"),
        )
