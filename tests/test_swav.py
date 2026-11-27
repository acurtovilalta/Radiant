from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.losses import FamilyFeatureQueue, SwAVAssigner, SwAVAssignmentConfig, sinkhorn_knopp
from radiant.models import AnatomyPrototypeBanks, PrototypeBankConfig


def test_sinkhorn_knopp_returns_column_stochastic_assignments() -> None:
    # Cosine-similarity logits, as in training (L2-normalized embeddings and prototypes).
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.nn.functional.normalize(torch.randn(6, 16, generator=generator), dim=1)
    prototypes = torch.nn.functional.normalize(torch.randn(4, 16, generator=generator), dim=1)
    logits = embeddings @ prototypes.t()
    assignments = sinkhorn_knopp(logits, epsilon=0.05, num_iters=200)

    assert assignments.shape == (6, 4)
    assert torch.allclose(assignments.sum(dim=1), torch.ones(6), atol=1e-4)
    column_mass = assignments.sum(dim=0)
    expected = torch.full_like(column_mass, 6.0 / 4.0)
    assert torch.allclose(column_mass, expected, atol=2e-2)


def test_family_feature_queue_keeps_latest_embeddings() -> None:
    queue = FamilyFeatureQueue(
        families=("Distal Arm", "Chest"),
        embedding_dim=3,
        queue_size=2,
    )
    queue.enqueue("Distal Arm", torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    queue.enqueue("Distal Arm", torch.tensor([[3.0, 0.0, 0.0]]))
    stored = queue.get("Distal Arm")

    assert stored.shape == (2, 3)
    assert torch.allclose(stored, torch.tensor([[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))


def test_swav_assigner_routes_samples_per_family() -> None:
    prototype_banks = AnatomyPrototypeBanks(
        PrototypeBankConfig(
            embedding_dim=8,
            prototypes_per_family={
                "Distal Arm": 5,
                "Proximal Arm": 6,
                "Hip": 7,
                "Foot": 8,
                "Knee": 4,
                "Chest": 3,
                "Thigh": 3,
                "Distal Leg": 3,
            },
        )
    )
    assigner = SwAVAssigner(
        prototype_banks=prototype_banks,
        config=SwAVAssignmentConfig(
            sinkhorn_iterations=5,
            use_feature_queue=False,
        ),
    )
    embeddings = torch.randn(3, 8)
    output = assigner(
        embeddings=embeddings,
        anatomy_family=["Knee", "Distal Arm", "Chest"],
    )

    assert output.family_assignments.shape == (3, 5)
    assert output.family_prototype_counts.tolist() == [4, 5, 3]
    assert output.family_names == ["Knee", "Distal Arm", "Chest"]
    assert torch.allclose(output.family_assignments[0, :4].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(output.family_assignments[1, :5].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(output.family_assignments[2, :3].sum(), torch.tensor(1.0), atol=1e-4)


def test_swav_assigner_uses_feature_queue_after_warmup() -> None:
    prototype_banks = AnatomyPrototypeBanks(
        PrototypeBankConfig(
            embedding_dim=6,
            prototypes_per_family=4,
        )
    )
    assigner = SwAVAssigner(
        prototype_banks=prototype_banks,
        config=SwAVAssignmentConfig(
            sinkhorn_iterations=4,
            use_feature_queue=True,
            queue_size=3,
            queue_start_steps=1,
        ),
    )

    first = assigner(
        embeddings=torch.randn(2, 6),
        anatomy_family=["Knee", "Knee"],
        update_queue=True,
    )
    second = assigner(
        embeddings=torch.randn(2, 6),
        anatomy_family=["Knee", "Knee"],
        update_queue=True,
    )

    assert first.queue_sizes.tolist() == [0, 0]
    assert second.queue_sizes.tolist() == [2, 2]
    # Queued rows only join the Sinkhorn batch; the codes returned are still for the 2 current samples.
    assert second.family_assignments.shape == (2, 4)
