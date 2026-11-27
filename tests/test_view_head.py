from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.losses.xray_view import XRayViewHead, encode_categorical_labels


def test_encode_categorical_labels_maps_unknown_to_minus_one() -> None:
    encoded = encode_categorical_labels(["Frontal", "lateral", "", "oblique"], {"frontal": 0, "lateral": 1}, torch.device("cpu"))
    assert encoded.tolist() == [0, 1, -1, -1]


def test_view_loss_trains_head_and_embedding() -> None:
    head = XRayViewHead(embedding_dim=8, num_classes=3)
    embedding = torch.randn(4, 8, requires_grad=True)
    loss = head(embedding, torch.tensor([0, 1, 2, -1]))
    loss.backward()
    assert float(loss) > 0.0
    assert embedding.grad is not None
    assert embedding.grad[3].abs().sum() == 0.0  # unknown view contributes nothing


def test_view_loss_is_zero_when_all_labels_missing() -> None:
    head = XRayViewHead(embedding_dim=8, num_classes=3)
    assert float(head(torch.randn(4, 8), torch.full((4,), -1))) == 0.0


def test_view_loss_is_zero_without_view_classes() -> None:
    head = XRayViewHead(embedding_dim=8, num_classes=0)
    assert head.network is None
    assert float(head(torch.randn(4, 8), torch.zeros(4, dtype=torch.long))) == 0.0
