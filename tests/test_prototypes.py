from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.models.prototypes import AnatomyPrototypeBanks, PrototypeBankConfig


def test_family_specific_banks_return_expected_shapes() -> None:
    module = AnatomyPrototypeBanks(
        PrototypeBankConfig(
            embedding_dim=32,
            prototypes_per_family={
                "Distal Arm": 8,
                "Proximal Arm": 6,
                "Hip": 10,
                "Foot": 12,
                "Knee": 5,
                "Chest": 4,
                "Thigh": 4,
                "Distal Leg": 4,
            },
        )
    )
    embeddings = torch.randn(3, 32)
    output = module(
        embeddings=embeddings,
        anatomy_family=["Distal Arm", "Knee", "Chest"],
    )

    assert output.family_logits.shape == (3, 8)
    assert output.family_assignments.shape == (3, 8)
    assert output.family_prototypes.shape == (3, 8, 32)
    assert output.family_names == ["Distal Arm", "Knee", "Chest"]


def test_family_id_routing_matches_family_names() -> None:
    module = AnatomyPrototypeBanks(
        PrototypeBankConfig(
            embedding_dim=24,
            prototypes_per_family=5,
            families=("Hip", "Knee", "UNKNOWN"),
        )
    )
    embeddings = torch.randn(2, 24)
    family_ids = torch.tensor([1, 0], dtype=torch.long)
    output = module(embeddings=embeddings, anatomy_family_ids=family_ids)

    assert output.family_names == ["Knee", "Hip"]
    assert output.family_ids.tolist() == [1, 0]
