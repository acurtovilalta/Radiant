from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.losses import SwAVAssigner, SwAVAssignmentConfig, SwAVLossConfig, swav_loss
from radiant.models import AnatomyPrototypeBanks, PrototypeBankConfig


def _build_banks(embedding_dim: int = 8) -> AnatomyPrototypeBanks:
    return AnatomyPrototypeBanks(
        PrototypeBankConfig(
            embedding_dim=embedding_dim,
            prototypes_per_family={
                "Distal Arm": 5,
                "Proximal Arm": 5,
                "Hip": 5,
                "Foot": 5,
                "Knee": 5,
                "Chest": 5,
                "Thigh": 5,
                "Distal Leg": 5,
            },
        )
    )


def test_swav_loss_is_symmetric_under_view_swap() -> None:
    torch.manual_seed(0)
    embeddings_a = torch.randn(4, 8)
    embeddings_b = torch.randn(4, 8)
    families = ["Knee", "Knee", "Distal Arm", "Distal Arm"]
    config = SwAVLossConfig(temperature=0.2, num_assignment_crops=2)

    assigner_ab = SwAVAssigner(_build_banks(embedding_dim=8), SwAVAssignmentConfig(sinkhorn_iterations=25))
    loss_ab = swav_loss(
        projections=[embeddings_a, embeddings_b],
        anatomy_family=families,
        assigner=assigner_ab,
        config=config,
        update_queue=False,
    ).loss

    assigner_ba = SwAVAssigner(_build_banks(embedding_dim=8), SwAVAssignmentConfig(sinkhorn_iterations=25))
    assigner_ba.load_state_dict(assigner_ab.state_dict(), strict=False)
    loss_ba = swav_loss(
        projections=[embeddings_b, embeddings_a],
        anatomy_family=families,
        assigner=assigner_ba,
        config=config,
        update_queue=False,
    ).loss

    assert torch.allclose(loss_ab, loss_ba, atol=1e-6)


def test_swav_loss_returns_expected_tensor_shapes_with_queue() -> None:
    torch.manual_seed(1)
    projections = [torch.randn(3, 8) for _ in range(4)]
    families = ["Knee", "Knee", "Chest"]
    assigner = SwAVAssigner(
        _build_banks(embedding_dim=8),
        SwAVAssignmentConfig(
            sinkhorn_iterations=25,
            use_feature_queue=True,
            queue_size=6,
            queue_start_steps=1,
        ),
    )

    first = swav_loss(
        projections=projections,
        anatomy_family=families,
        assigner=assigner,
        config=SwAVLossConfig(temperature=0.2, num_assignment_crops=2),
        update_queue=True,
    )
    second = swav_loss(
        projections=projections,
        anatomy_family=families,
        assigner=assigner,
        config=SwAVLossConfig(temperature=0.2, num_assignment_crops=2),
        update_queue=True,
    )

    assert first.loss.ndim == 0
    assert len(second.assignment_outputs) == 2
    assert second.assignment_outputs[0].family_assignments.shape == (3, 5)
    assert second.assignment_outputs[0].queue_sizes.tolist() == [4, 4, 2]
    assert second.assignment_outputs[1].queue_sizes.tolist() == [6, 6, 3]


def test_swav_one_batch_overfit_sanity_check() -> None:
    torch.manual_seed(2)
    banks = _build_banks(embedding_dim=6)
    assigner = SwAVAssigner(
        banks,
        SwAVAssignmentConfig(
            sinkhorn_iterations=30,
            use_feature_queue=False,
        ),
    )
    projections = [
        torch.nn.Parameter(torch.randn(4, 6)),
        torch.nn.Parameter(torch.randn(4, 6)),
        torch.nn.Parameter(torch.randn(4, 6)),
        torch.nn.Parameter(torch.randn(4, 6)),
    ]
    optimizer = torch.optim.Adam(list(banks.parameters()) + projections, lr=0.05)
    families = ["Knee", "Knee", "Distal Arm", "Distal Arm"]
    config = SwAVLossConfig(temperature=0.2, num_assignment_crops=2)

    with torch.no_grad():
        initial_loss = swav_loss(
            projections=[parameter.detach() for parameter in projections],
            anatomy_family=families,
            assigner=assigner,
            config=config,
            update_queue=False,
        ).loss.item()

    final_loss = initial_loss
    for _ in range(40):
        optimizer.zero_grad()
        loss = swav_loss(
            projections=projections,
            anatomy_family=families,
            assigner=assigner,
            config=config,
            update_queue=False,
        ).loss
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    assert final_loss < initial_loss


def test_queue_updates_preserve_assignment_gradients_for_joint_loss() -> None:
    banks = _build_banks()
    assigner = SwAVAssigner(
        banks, SwAVAssignmentConfig(use_feature_queue=True, queue_size=4, queue_start_steps=0),
    )
    families = ["Knee", "Knee", "Chest", "Chest"]
    projections = [torch.nn.Parameter(torch.randn(4, 8)) for _ in range(3)]
    for family in ("Knee", "Chest"):
        assigner.queue.enqueue(family, torch.randn(4, 8))
    output = swav_loss(projections, families, assigner, SwAVLossConfig(), update_queue=True)
    # Centroid alignment also differentiates through crop-0 assignment codes.
    loss = output.loss + output.assignment_outputs[0].family_assignments.square().sum()
    loss.backward()
    for projection in projections:
        assert projection.grad is not None
        assert torch.isfinite(projection.grad).all()
    gradient = banks.get_family_bank("Knee").prototypes.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
