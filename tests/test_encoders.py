from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.models import (
    MRI3DEncoder,
    MRIEncoderConfig,
    MultiModalEncoderSystem,
    MultiModalEncoderSystemConfig,
    XRay2DEncoder,
    XRayEncoderConfig,
)
from radiant.models.xray_encoder import _build_resnet2d_backbone, get_resnet_stage_modules


def test_pretrained_single_channel_resnet_uses_mean_rgb_conv1_weights(monkeypatch) -> None:
    original_conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
    pretrained_weight = torch.arange(original_conv1.weight.numel(), dtype=torch.float32).reshape_as(original_conv1.weight)
    with torch.no_grad():
        original_conv1.weight.copy_(pretrained_weight)

    dummy_net = nn.Module()
    dummy_net.conv1 = original_conv1
    dummy_net.relu = nn.ReLU(inplace=True)

    mocked_builder = Mock(return_value=dummy_net)
    monkeypatch.setattr("radiant.models.xray_encoder.torchvision.models.resnet18", mocked_builder)

    backbone, _ = _build_resnet2d_backbone(backbone="resnet18", in_channels=1, pretrained=True)

    expected_weight = pretrained_weight.mean(dim=1, keepdim=True)
    assert backbone.conv1.in_channels == 1
    assert torch.allclose(backbone.conv1.weight, expected_weight)
    mocked_builder.assert_called_once()


def test_get_resnet_stage_modules_returns_stem_then_layers_in_order() -> None:
    encoder = XRay2DEncoder(XRayEncoderConfig(backbone="resnet18", embedding_dim=16))

    stages = get_resnet_stage_modules(encoder)

    assert stages == [encoder.stem, encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4]


def test_xray_encoder_returns_single_normalized_embedding() -> None:
    encoder = XRay2DEncoder(
        XRayEncoderConfig(
            backbone="resnet18",
            embedding_dim=128,
            projector_hidden_dim=256,
        )
    )
    batch = torch.randn(2, 1, 224, 224)
    output = encoder(batch)

    assert output.feature_map.ndim == 4
    assert output.embedding.shape == (2, 128)
    assert torch.allclose(output.embedding.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.embedding, torch.nn.functional.normalize(output.projection, dim=-1), atol=1e-6)
    assert "layer4" in output.intermediate_feature_maps


def test_mri_encoder_returns_single_normalized_embedding() -> None:
    encoder = MRI3DEncoder(
        MRIEncoderConfig(
            backbone="resnet34",
            base_channels=16,
            embedding_dim=96,
            projector_hidden_dim=128,
        )
    )
    batch = torch.randn(2, 1, 16, 96, 96)
    output = encoder(batch)

    assert output.feature_map.ndim == 5
    assert output.embedding.shape == (2, 96)
    assert torch.allclose(output.embedding.norm(dim=1), torch.ones(2), atol=1e-5)
    assert "layer4" in output.intermediate_feature_maps
    assert output.mri_proj_feats is None
    assert output.mri_proj_embs is None
    assert output.mri_depth_attn is None


def test_mri_encoder_projection_bridge_global_average_pooling_mode_threads_through_config() -> None:
    encoder = MRI3DEncoder(
        MRIEncoderConfig(
            backbone="resnet34",
            base_channels=8,
            embedding_dim=32,
            use_mri_projection_bridge=True,
            mri_projection_num_projections=4,
            mri_projection_embedding_dim=32,
            mri_projection_bridge_pooling_mode="global_average",
            stage_stride_schedule="half_depth",
        )
    )
    batch = torch.randn(2, 1, 16, 64, 64)
    output = encoder(batch)

    assert encoder.mri_projection_bridge.pooling_mode == "global_average"
    assert output.mri_proj_embs is not None
    assert output.mri_proj_embs.shape == (2, 1, 32)
    assert output.mri_depth_attn is not None
    # The half_depth stage-stride schedule halves depth once: 16 -> 8 (one uniform weight per depth position).
    assert output.mri_depth_attn.shape == (2, 1, 8)
    assert torch.allclose(output.mri_depth_attn, torch.full_like(output.mri_depth_attn, 1.0 / 8.0))


def test_multimodal_encoder_system_runs_both_modalities_on_single_device() -> None:
    system = MultiModalEncoderSystem(
        MultiModalEncoderSystemConfig(
            xray=XRayEncoderConfig(backbone="resnet18", embedding_dim=32),
            mri=MRIEncoderConfig(backbone="resnet34", base_channels=8, embedding_dim=32),
            xray_device="cpu",
            mri_device="cpu",
            output_device="cpu",
        )
    )

    outputs = system(
        xray=torch.randn(1, 1, 128, 128),
        mri=torch.randn(1, 1, 16, 64, 64),
    )

    assert set(outputs) == {"xray", "mri"}
    assert outputs["xray"].embedding.device.type == "cpu"
    assert outputs["mri"].embedding.device.type == "cpu"


def test_multimodal_encoder_smoke_with_optional_mri_projection_bridge() -> None:
    batch_size = 2
    embed_dim = 32
    num_projections = 4

    system = MultiModalEncoderSystem(
        MultiModalEncoderSystemConfig(
            xray=XRayEncoderConfig(
                backbone="resnet18",
                embedding_dim=embed_dim,
            ),
            mri=MRIEncoderConfig(
                backbone="resnet34",
                base_channels=8,
                embedding_dim=embed_dim,
                use_mri_projection_bridge=True,
                mri_projection_num_projections=num_projections,
                mri_projection_embedding_dim=embed_dim,
            ),
            xray_device="cpu",
            mri_device="cpu",
            output_device="cpu",
        )
    )

    outputs = system(
        xray=torch.randn(batch_size, 1, 128, 128),
        mri=torch.randn(batch_size, 1, 32, 64, 64),
    )

    x_output = outputs["xray"]
    m_output = outputs["mri"]
    assert x_output.embedding.shape == (batch_size, embed_dim)
    assert m_output.mri_proj_embs is not None
    assert m_output.mri_proj_embs.shape == (batch_size, num_projections, embed_dim)
    assert torch.allclose(
        m_output.mri_proj_embs.norm(dim=-1),
        torch.ones(batch_size, num_projections),
        atol=1e-5,
    )
    assert m_output.mri_proj_feats is not None
    assert m_output.mri_depth_attn is not None


def test_mri_projection_bridge_can_be_skipped_in_forward() -> None:
    system = MultiModalEncoderSystem(
        MultiModalEncoderSystemConfig(
            xray=XRayEncoderConfig(backbone="resnet18", embedding_dim=32),
            mri=MRIEncoderConfig(
                backbone="resnet34",
                base_channels=8,
                embedding_dim=32,
                use_mri_projection_bridge=True,
                mri_projection_num_projections=4,
                mri_projection_embedding_dim=32,
            ),
            xray_device="cpu",
            mri_device="cpu",
            output_device="cpu",
        )
    )

    mri_batch = torch.randn(2, 1, 32, 64, 64)
    skipped = system(mri=mri_batch, compute_projection_bridge=False)["mri"]
    used = system(mri=mri_batch, compute_projection_bridge=True)["mri"]

    assert skipped.mri_proj_feats is None
    assert skipped.mri_proj_embs is None
    assert skipped.mri_depth_attn is None
    assert used.mri_proj_feats is not None
    assert used.mri_proj_embs is not None
    assert used.mri_depth_attn is not None


def test_mri_encoder_half_depth_schedule_matches_radiant_configs() -> None:
    encoder = MRI3DEncoder(
        MRIEncoderConfig(
            backbone="resnet34",
            base_channels=32,
            stage_stride_schedule="half_depth",
        )
    )

    global_crop = torch.randn(1, 1, 32, 192, 192)
    local_crop = torch.randn(1, 1, 16, 128, 128)

    global_layer4 = encoder(global_crop).intermediate_feature_maps["layer4"]
    local_layer4 = encoder(local_crop).intermediate_feature_maps["layer4"]

    assert global_layer4.shape[-3:] == (16, 6, 6)
    assert local_layer4.shape[-3:] == (8, 4, 4)


def _module_grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


def _named_grad_norm(module: torch.nn.Module, substring: str) -> float:
    total = 0.0
    for name, parameter in module.named_parameters():
        if substring in name and parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


def test_mri_projection_bridge_gradient_reaches_bridge_head_and_backbone() -> None:
    batch = torch.randn(2, 1, 32, 64, 64)
    encoder = MRI3DEncoder(
        MRIEncoderConfig(
            backbone="resnet34",
            base_channels=8,
            embedding_dim=32,
            projector_hidden_dim=64,
            use_mri_projection_bridge=True,
            mri_projection_num_projections=4,
            mri_projection_embedding_dim=32,
        )
    )
    encoder.zero_grad(set_to_none=True)
    output = encoder(batch)
    assert output.mri_proj_feats is not None
    assert output.mri_proj_embs is not None
    assert output.mri_depth_attn is not None

    output.mri_proj_embs.sum().backward()
    assert _module_grad_norm(encoder.mri_projection_bridge) > 0.0
    assert _module_grad_norm(encoder.mri_projection_head) > 0.0
    assert _named_grad_norm(encoder, "layer4") > 0.0


def test_slice_controls_mri_encoders() -> None:
    xray = XRayEncoderConfig(backbone="resnet18", embedding_dim=16, projector_hidden_dim=32)
    # C4: a separate 2D encoder over the slices of a volume; the slice embeddings form the MIL bag.
    mri = MRIEncoderConfig(
        backbone="resnet34", embedding_dim=16, projector_hidden_dim=32, use_mri_projection_bridge=True,
        mri_projection_embedding_dim=16, encoder_type="slice_2d",
    )
    system = MultiModalEncoderSystem(MultiModalEncoderSystemConfig(xray=xray, mri=mri))
    out = system.forward_mri(torch.randn(2, 3, 1, 64, 64))
    assert out.embedding.shape == (2, 16) and out.mri_proj_embs.shape == (2, 3, 16)
    # C3: MRI slices go through the X-ray encoder; the MRI module has no parameters.
    mri = MRIEncoderConfig(encoder_type="shared_xray")
    system = MultiModalEncoderSystem(MultiModalEncoderSystemConfig(xray=xray, mri=mri))
    assert not list(system.mri_encoder.parameters())
    system.eval()
    slices = torch.randn(2, 1, 64, 64)
    assert torch.equal(system.forward_mri(slices).embedding, system.forward_xray(slices).embedding)


def test_slice_transform_shares_one_draw_across_slices() -> None:
    from radiant.data.slices import SliceMultiCropTransform, select_slice_indices
    from radiant.data.transforms import XRayAugmentationConfig

    assert select_slice_indices(32, 8, 0.5, "evenly_spaced") == [8, 10, 12, 14, 17, 19, 21, 23]
    assert all(8 <= index < 24 for index in select_slice_indices(32, 50, 0.5, "random"))
    transform = SliceMultiCropTransform(XRayAugmentationConfig(global_size=32, local_size=16, num_local_crops=1))
    slice_ = torch.randn(1, 64, 64)
    global_views, local_views = transform(slice_.repeat(3, 1, 1))
    assert global_views[0].shape == (3, 1, 32, 32) and local_views[0].shape == (3, 1, 16, 16)
    assert torch.equal(global_views[0][0], global_views[0][2])
