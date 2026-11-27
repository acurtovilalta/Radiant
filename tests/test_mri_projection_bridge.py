from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.models import (
    MRIProjection2DHead,
    MRIProjection2DHeadConfig,
    MRIProjectionBridge,
    MRIProjectionBridgeConfig,
)


def test_mri_projection_bridge_output_shapes() -> None:
    bridge = MRIProjectionBridge(MRIProjectionBridgeConfig(in_channels=64, num_projections=4))
    features = torch.randn(2, 64, 8, 16, 16)
    projected, attention = bridge(features)

    assert projected.shape == (2, 4, 64, 16, 16)
    assert attention.shape == (2, 4, 8)


def test_mri_projection_bridge_matches_eq1_and_eq2() -> None:
    """Eq. 1: alpha = softmax_d(q_j^T t_d / tau_p), t_d = spatial mean of V at depth d.
    Eq. 2: P_j = sum_d alpha_d V[:, d]; the depth axis is removed and the in-plane layout kept."""
    torch.manual_seed(0)
    bridge = MRIProjectionBridge(MRIProjectionBridgeConfig(in_channels=16, num_projections=3, attention_temperature=0.5))
    features = torch.randn(2, 16, 8, 5, 5)
    projected, attention = bridge(features)

    tokens = features.mean(dim=(3, 4)).permute(0, 2, 1)  # [B, D, C]
    expected_attention = torch.softmax(torch.einsum("bdc,kc->bkd", tokens, bridge.depth_queries) / 0.5, dim=-1)
    expected_projection = torch.einsum("bkd,bcdhw->bkchw", expected_attention, features)
    assert torch.allclose(attention, expected_attention, atol=1e-6)
    assert torch.allclose(projected, expected_projection, atol=1e-5)
    assert projected.shape == (2, 3, 16, 5, 5)


def test_mri_projection_bridge_attention_sums_to_one_over_depth() -> None:
    bridge = MRIProjectionBridge(MRIProjectionBridgeConfig(in_channels=64, num_projections=4))
    features = torch.randn(2, 64, 8, 16, 16)
    _, attention = bridge(features)

    sums = attention.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_mri_projection_bridge_backward_pass_propagates_gradients() -> None:
    bridge = MRIProjectionBridge(MRIProjectionBridgeConfig(in_channels=64, num_projections=4))
    features = torch.randn(2, 64, 8, 16, 16, requires_grad=True)
    projected, attention = bridge(features)

    loss = projected.square().mean() + attention.square().mean()
    loss.backward()

    assert features.grad is not None
    assert bridge.depth_queries.grad is not None


def test_mri_projection_bridge_rejects_more_projections_than_depth() -> None:
    bridge = MRIProjectionBridge(MRIProjectionBridgeConfig(in_channels=64, num_projections=8))
    features = torch.randn(2, 64, 6, 16, 16)

    with pytest.raises(ValueError) as excinfo:
        bridge(features)

    message = str(excinfo.value)
    assert "K=8" in message
    assert "D=6" in message
    assert "0.0476" in message  # (8/6 - 1) / (8 - 1)


def test_mri_projection_2d_head_output_shape() -> None:
    head = MRIProjection2DHead(MRIProjection2DHeadConfig(in_channels=64, embedding_dim=256))
    projected_features = torch.randn(2, 4, 64, 16, 16)
    embeddings = head(projected_features)

    assert embeddings.shape == (2, 4, 256)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_mri_projection_bridge_global_average_pooling_collapses_to_one_uniform_projection() -> None:
    bridge = MRIProjectionBridge(
        MRIProjectionBridgeConfig(in_channels=64, num_projections=4, pooling_mode="global_average")
    )
    features = torch.randn(2, 64, 8, 16, 16)
    projected, attention = bridge(features)

    assert bridge.depth_queries is None
    assert bridge.num_projections == 1
    assert projected.shape == (2, 1, 64, 16, 16)
    assert attention.shape == (2, 1, 8)
    assert torch.allclose(attention, torch.full_like(attention, 1.0 / 8.0))
    assert torch.allclose(projected.squeeze(1), features.mean(dim=2), atol=1e-6)


def test_mri_projection_bridge_global_average_pooling_backward_pass_skips_depth_queries() -> None:
    bridge = MRIProjectionBridge(
        MRIProjectionBridgeConfig(in_channels=64, num_projections=4, pooling_mode="global_average")
    )
    features = torch.randn(2, 64, 8, 16, 16, requires_grad=True)
    projected, _ = bridge(features)

    projected.square().mean().backward()

    assert features.grad is not None
    assert not list(bridge.parameters())


def test_mri_projection_2d_head_backward_pass_propagates_gradients() -> None:
    head = MRIProjection2DHead(MRIProjection2DHeadConfig(in_channels=64, embedding_dim=256))
    projected_features = torch.randn(2, 4, 64, 16, 16, requires_grad=True)
    embeddings = head(projected_features)

    loss = embeddings.square().mean()
    loss.backward()

    assert projected_features.grad is not None
    assert head.conv1x1.weight.grad is not None
    assert head.bn.weight.grad is not None
    assert head.projection.weight.grad is not None
