from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.models import XRay2DEncoder, XRayEncoderConfig
from radiant.models.dinov3_backbone import (
    Dinov3Backbone,
    get_embedding_and_norm_modules,
    get_transformer_blocks,
)
from radiant.models.xray_encoder import _build_xray2d_backbone, get_resnet_stage_modules
from radiant.training.trainer import StageTrainer, TrainerConfig
from radiant.data.datasets import MRIDataConfig, XRayDataConfig
from radiant.losses import SwAVAssignmentConfig, SwAVLossConfig
from radiant.models import MRIEncoderConfig, PrototypeBankConfig
from radiant.training import (
    CheckpointConfig,
    OptimizerConfig,
    StageScheduleConfig,
)


class _FakeVitBackbone(nn.Module):
    """Mimics the real DINOv3ViTModel attribute nesting: model.model.layer for
    transformer blocks, model.embeddings for patch embeddings, model.norm for the
    final layernorm (verified against transformers>=5.x's DINOv3ViTModel)."""

    def __init__(self, feature_dim: int = 48) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, feature_dim, kernel_size=1, bias=False)
        self.model = nn.Module()
        self.model.embeddings = nn.Conv2d(1, feature_dim, kernel_size=1, bias=False)
        self.model.model = nn.Module()
        self.model.model.layer = nn.ModuleList(
            [nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False) for _ in range(12)]
        )
        self.model.norm = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.hidden_size = feature_dim

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        for layer in self.model.model.layer:
            outputs = layer(outputs)
        return outputs

    def forward_features_and_pool(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        return features, features.mean(dim=(2, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features_and_pool(inputs)[1]


def _write_nifti(path: Path, array: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), str(path))
    return str(path)


def _build_manifest_files(tmp_path: Path) -> tuple[str, str]:
    xray_paths = [
        _write_nifti(tmp_path / f"xray_{index}.nii.gz", np.random.randn(64, 64))
        for index in range(4)
    ]
    mri_paths = [
        _write_nifti(tmp_path / f"mri_{index}.nii.gz", np.random.randn(12, 32, 32))
        for index in range(4)
    ]
    xray_manifest = pd.DataFrame(
        [
            {
                "split": "train" if index < 3 else "val",
                "filepath": path,
                "patient_id": f"xp{index}",
                "anatomy_family": "Knee",
                "view_or_plane": "frontal",
            }
            for index, path in enumerate(xray_paths)
        ]
    )
    mri_manifest = pd.DataFrame(
        [
            {
                "split": "train" if index < 3 else "val",
                "filepath": path,
                "patient_id": f"mp{index}",
                "anatomy_family": "Knee",
            }
            for index, path in enumerate(mri_paths)
        ]
    )
    xray_path = tmp_path / "xray_manifest.csv"
    mri_path = tmp_path / "mri_manifest.csv"
    xray_manifest.to_csv(xray_path, index=False)
    mri_manifest.to_csv(mri_path, index=False)
    return str(xray_path), str(mri_path)


def _make_trainer_config(tmp_path: Path) -> TrainerConfig:
    xray_manifest, mri_manifest = _build_manifest_files(tmp_path)
    return TrainerConfig(
        output_dir=str(tmp_path / "outputs"),
        use_amp=False,
        log_every_steps=0,
        val_every_epochs=1,
        xray_data=XRayDataConfig(
            manifest_path=xray_manifest,
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            split="train",
            batch_size=2,
            num_workers=1,
            shuffle=False,
            global_crop_size=64,
            local_crop_size=32,
            num_local_crops=1,
            augmentations={
                "affine_prob": 0.0,
                "intensity_jitter_prob": 0.0,
                "gaussian_blur_prob": 0.0,
                "gaussian_noise_prob": 0.0,
            },
        ),
        mri_data=MRIDataConfig(
            manifest_path=mri_manifest,
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            split="train",
            batch_size=1,
            num_workers=1,
            shuffle=False,
            global_crop_size=(8, 24, 24),
            local_crop_size=(6, 16, 16),
            num_local_crops=1,
            augmentations={
                "affine_prob": 0.0,
                "intensity_scale_prob": 0.0,
                "intensity_shift_prob": 0.0,
                "contrast_prob": 0.0,
                "gaussian_noise_prob": 0.0,
                "gaussian_smooth_prob": 0.0,
            },
        ),
        xray_model=XRayEncoderConfig(
            backbone="dinov3_vitb16",
            pretrained=False,
            embedding_dim=32,
            projector_hidden_dim=64,
        ),
        mri_model=MRIEncoderConfig(
            backbone="resnet34",
            base_channels=8,
            embedding_dim=32,
            projector_hidden_dim=64,
        ),
        prototype_banks=PrototypeBankConfig(embedding_dim=32, prototypes_per_family=8),
        xray_assignment=SwAVAssignmentConfig(sinkhorn_iterations=10),
        mri_assignment=SwAVAssignmentConfig(sinkhorn_iterations=10, use_feature_queue=True, queue_size=8),
        xray_swav_loss=SwAVLossConfig(num_assignment_crops=2),
        mri_swav_loss=SwAVLossConfig(num_assignment_crops=2),
        lambda_centroid_alignment=0.5,
        schedule=StageScheduleConfig(warm_start_epochs=1, joint_epochs=1),
        optimizer=OptimizerConfig(lr=1e-3, grad_clip_norm=1.0, warmup_epochs=0),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "checkpoints"), save_every_epochs=1),
        devices={"xray": "cpu", "mri": "cpu", "output": "cpu"},
    )


def test_build_xray2d_backbone_uses_dinov3_pretrained_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[bool, str | None, str | None]] = []

    def _fake_loader(*, pretrained: bool, model_id: str | None, cache_dir: str | None) -> tuple[nn.Module, int]:
        captured.append((pretrained, model_id, cache_dir))
        return _FakeVitBackbone(), 48

    monkeypatch.setattr("radiant.models.xray_encoder.load_dinov3_vitb16_backbone", _fake_loader)

    _build_xray2d_backbone(
        XRayEncoderConfig(
            backbone="dinov3_vitb16",
            pretrained=True,
            model_id="facebook/example",
            cache_dir="/tmp/cache-a",
        )
    )
    _build_xray2d_backbone(
        XRayEncoderConfig(
            backbone="dinov3_vitb16",
            pretrained=False,
            model_id="facebook/example",
            cache_dir="/tmp/cache-b",
        )
    )

    assert captured == [
        (True, "facebook/example", "/tmp/cache-a"),
        (False, "facebook/example", "/tmp/cache-b"),
    ]


def test_xray_encoder_supports_dinov3_vit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "radiant.models.xray_encoder.load_dinov3_vitb16_backbone",
        lambda **_: (_FakeVitBackbone(), 48),
    )

    encoder = XRay2DEncoder(
        XRayEncoderConfig(
            backbone="dinov3_vitb16",
            pretrained=False,
            embedding_dim=24,
            projector_hidden_dim=32,
        )
    )
    batch = torch.randn(2, 1, 32, 32)
    output = encoder(batch)

    assert output.feature_map.ndim == 4
    assert output.pooled_feature.shape == (2, 48)
    assert output.embedding.shape == (2, 24)
    assert torch.allclose(output.embedding.norm(dim=1), torch.ones(2), atol=1e-5)
    assert "patch_tokens" in output.intermediate_feature_maps


def test_stage_trainer_builds_with_dinov3_xray(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "radiant.models.xray_encoder.load_dinov3_vitb16_backbone",
        lambda **_: (_FakeVitBackbone(feature_dim=32), 32),
    )

    trainer = StageTrainer(_make_trainer_config(tmp_path))

    assert trainer.xray_encoder_module.backbone_name == "dinov3_vitb16"
    assert trainer.xray_encoder_module.feature_dim == 32


def test_get_transformer_blocks_and_norm_modules_from_real_dinov3_architecture() -> None:
    cache_dir = str(Path(__file__).resolve().parents[1] / "dinov3_model_cache")
    backbone = Dinov3Backbone(pretrained=False, cache_dir=cache_dir)

    blocks = get_transformer_blocks(backbone)
    embeddings, final_norm = get_embedding_and_norm_modules(backbone)

    assert len(blocks) == backbone.model.config.num_hidden_layers
    assert embeddings is not None
    assert final_norm is not None


def test_apply_xray_vit_gradual_unfreeze_narrows_block_trainability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "radiant.models.xray_encoder.load_dinov3_vitb16_backbone",
        lambda **_: (_FakeVitBackbone(feature_dim=32), 32),
    )
    config = _make_trainer_config(tmp_path)
    config.schedule = StageScheduleConfig(
        warm_start_epochs=1,
        joint_epochs=1,
        enable_xray_vit_gradual_unfreeze=True,
        xray_vit_unfreeze_start_epoch=4,
        xray_vit_unfreeze_epochs_per_stage=2,
        xray_vit_unfreeze_blocks_per_stage=2,
        xray_vit_freeze_patch_embed_until_full=True,
    )
    trainer = StageTrainer(config)
    blocks = get_transformer_blocks(trainer.xray_encoder_module.vit_backbone)
    embeddings, final_norm = get_embedding_and_norm_modules(trainer.xray_encoder_module.vit_backbone)

    # Sec. 2.4: frozen for the 4 warm-up epochs, then two blocks every two epochs, patch embedding last.
    trainer._set_trainability_for_stage("joint")
    trainer._apply_xray_vit_gradual_unfreeze(epoch=3, stage="joint")
    assert [block.weight.requires_grad for block in blocks] == [False] * 12
    assert embeddings.weight.requires_grad is False
    assert final_norm.weight.requires_grad is False

    trainer._apply_xray_vit_gradual_unfreeze(epoch=4, stage="joint")
    assert [block.weight.requires_grad for block in blocks] == [False] * 10 + [True] * 2
    assert final_norm.weight.requires_grad is True
    assert embeddings.weight.requires_grad is False

    trainer._apply_xray_vit_gradual_unfreeze(epoch=12, stage="joint")
    assert [block.weight.requires_grad for block in blocks] == [False] * 2 + [True] * 10
    assert embeddings.weight.requires_grad is False

    trainer._apply_xray_vit_gradual_unfreeze(epoch=14, stage="joint")
    assert all(block.weight.requires_grad for block in blocks)
    assert embeddings.weight.requires_grad is True


def test_apply_xray_cnn_gradual_unfreeze_narrows_stage_trainability(tmp_path: Path) -> None:
    config = _make_trainer_config(tmp_path)
    config.xray_model = replace(config.xray_model, backbone="resnet18", pretrained=False)
    config.schedule = StageScheduleConfig(
        warm_start_epochs=1,
        joint_epochs=1,
        enable_xray_cnn_gradual_unfreeze=True,
        xray_cnn_unfreeze_start_epoch=4,
        xray_cnn_unfreeze_epochs_per_stage=3,
        xray_cnn_unfreeze_stages_per_step=1,
    )
    trainer = StageTrainer(config)
    stages = get_resnet_stage_modules(trainer.xray_encoder_module)

    def _trainable_flags() -> list[bool]:
        return [any(parameter.requires_grad for parameter in stage.parameters()) for stage in stages]

    # Sec. 2.4: frozen for the 4 warm-up epochs, then one ResNet stage every three epochs (stem last);
    # the backbone is fully trainable from epoch 16, i.e. joint epoch 12.
    trainer._set_trainability_for_stage("joint")
    trainer._apply_xray_cnn_gradual_unfreeze(epoch=3, stage="joint")
    assert _trainable_flags() == [False, False, False, False, False]

    trainer._apply_xray_cnn_gradual_unfreeze(epoch=4, stage="joint")
    assert _trainable_flags() == [False, False, False, False, True]

    trainer._apply_xray_cnn_gradual_unfreeze(epoch=15, stage="joint")
    assert _trainable_flags() == [False, True, True, True, True]

    trainer._apply_xray_cnn_gradual_unfreeze(epoch=16, stage="joint")
    assert _trainable_flags() == [True, True, True, True, True]


def test_scratch_vitb16_initializes_offline_without_pretrained_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    def fail_remote_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Scratch initialization must not load pretrained configuration or weights")

    monkeypatch.setattr(DINOv3ViTConfig, "from_pretrained", fail_remote_load)
    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", fail_remote_load)
    backbone = Dinov3Backbone(pretrained=False)
    assert backbone.hidden_size == 768
    assert backbone.model.config.num_hidden_layers == 12
    assert backbone.model.config.num_attention_heads == 12
    assert backbone.model.config.num_register_tokens == 4
