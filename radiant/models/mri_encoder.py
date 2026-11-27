"""3D MRI ResNet encoder and optional depth projection bridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .mri_projection_bridge import (
    MRIProjection2DHead,
    MRIProjection2DHeadConfig,
    MRIProjectionBridge,
    MRIProjectionBridgeConfig,
)
from .outputs import EncoderOutput
from .projection_head import ProjectionHead, normalize_embedding


@dataclass(slots=True)
class MRIEncoderConfig:
    """Configuration for the MRI 3D CNN encoder."""

    backbone: str = "resnet34"
    in_channels: int = 1
    base_channels: int = 32
    embedding_dim: int = 256
    projector_hidden_dim: int = 512
    projector_layers: int = 2
    use_mri_projection_bridge: bool = False
    mri_projection_num_projections: int = 4
    mri_projection_embedding_dim: int = 256
    mri_projection_attention_temperature: float = 1.0
    # "learned_attention" (default) or "global_average" -- see
    # MRIProjectionBridgeConfig.pooling_mode. Only meaningful when
    # use_mri_projection_bridge is true.
    mri_projection_bridge_pooling_mode: str = "learned_attention"
    stage_stride_schedule: str = "half_depth"
    pretrained_checkpoint: str | None = None
    # "volume_3d" (default): this 3D encoder. Controls: "shared_xray" (C3, MRI slices go through the X-ray
    # encoder) and "slice_2d" (C4, models/slice_encoder.py). See MultiModalEncoderSystem.
    encoder_type: str = "volume_3d"


class _BasicBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
    ) -> None:
        super().__init__()
        stride3d = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride3d,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut: nn.Module
        if stride3d != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride3d, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a residual 3D basic block.

        Both ReLUs run in place: the standard torchvision BasicBlock pattern
        (same shape of computation -- one ReLU on a value nothing else reads
        before the next op consumes it, one ReLU on the fresh output of the
        residual add) is safe to run in place, and doing so avoids allocating
        a full extra activation-sized buffer at every block, on every forward,
        for the whole 3D backbone.
        """
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs))
        outputs = outputs + self.shortcut(inputs)
        return F.relu(outputs, inplace=True)


# Strides (depth, height, width) of layer1-layer4. "half_depth" is the reduced depth downsampling of the
# paper: depth is halved once (layer2), so a 32-slice crop gives D = 16 depth positions for the bridge.
STAGE_STRIDE_SCHEDULES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "half_depth": ((1, 1, 1), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
}


class MRI3DEncoder(nn.Module):
    """3D ResNet encoder for MRI volumes with one projection head and the optional projection bridge."""

    _DEPTHS: dict[str, tuple[int, int, int, int]] = {
        "resnet34": (3, 4, 6, 3),
    }

    def __init__(self, config: MRIEncoderConfig) -> None:
        super().__init__()
        if config.backbone not in self._DEPTHS:
            raise ValueError("Unsupported MRI backbone. Use resnet34.")
        if config.stage_stride_schedule not in STAGE_STRIDE_SCHEDULES:
            raise ValueError(
                "Unsupported stage_stride_schedule "
                f"{config.stage_stride_schedule!r}. Use one of "
                f"{sorted(STAGE_STRIDE_SCHEDULES)}."
            )
        self.config = config
        self.in_channels = config.base_channels

        self.conv1 = nn.Conv3d(
            config.in_channels,
            config.base_channels,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(config.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        depths = self._DEPTHS[config.backbone]
        stage_strides = STAGE_STRIDE_SCHEDULES[config.stage_stride_schedule]
        self.layer1 = self._make_layer(config.base_channels, depths[0], stride=stage_strides[0])
        self.layer2 = self._make_layer(config.base_channels * 2, depths[1], stride=stage_strides[1])
        self.layer3 = self._make_layer(config.base_channels * 4, depths[2], stride=stage_strides[2])
        self.layer4 = self._make_layer(config.base_channels * 8, depths[3], stride=stage_strides[3])
        self.feature_dim = config.base_channels * 8

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projector = ProjectionHead(
            in_dim=self.feature_dim,
            out_dim=config.embedding_dim,
            hidden_dim=config.projector_hidden_dim,
            num_layers=config.projector_layers,
        )
        self.mri_projection_bridge: MRIProjectionBridge | None = None
        self.mri_projection_head: MRIProjection2DHead | None = None
        if config.use_mri_projection_bridge:
            self.mri_projection_bridge = MRIProjectionBridge(
                MRIProjectionBridgeConfig(
                    in_channels=self.feature_dim,
                    num_projections=config.mri_projection_num_projections,
                    attention_temperature=config.mri_projection_attention_temperature,
                    pooling_mode=config.mri_projection_bridge_pooling_mode,
                )
            )
            self.mri_projection_head = MRIProjection2DHead(
                MRIProjection2DHeadConfig(
                    in_channels=self.feature_dim,
                    embedding_dim=config.mri_projection_embedding_dim,
                )
            )

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int | tuple[int, int, int],
    ) -> nn.Sequential:
        strides: list[int | tuple[int, int, int]] = [stride] + [1] * (blocks - 1)
        layers: list[nn.Module] = []
        for layer_stride in strides:
            layers.append(_BasicBlock3D(self.in_channels, out_channels, stride=layer_stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(
        self,
        inputs: torch.Tensor,
        compute_projection_bridge: bool = True,
    ) -> EncoderOutput:
        """Encode MRI volumes; return features, the normalized embedding and (optionally) the K projections."""
        intermediate: dict[str, torch.Tensor] = {}
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        intermediate["stem"] = outputs
        outputs = self.maxpool(outputs)
        intermediate["maxpool"] = outputs
        outputs = self.layer1(outputs)
        intermediate["layer1"] = outputs
        outputs = self.layer2(outputs)
        intermediate["layer2"] = outputs
        outputs = self.layer3(outputs)
        intermediate["layer3"] = outputs
        feature_map = self.layer4(outputs)
        intermediate["layer4"] = feature_map

        mri_proj_feats = None
        mri_proj_embs = None
        mri_depth_attn = None
        if (
            compute_projection_bridge
            and self.mri_projection_bridge is not None
            and self.mri_projection_head is not None
        ):
            mri_proj_feats, mri_depth_attn = self.mri_projection_bridge(feature_map)
            mri_proj_embs = self.mri_projection_head(mri_proj_feats)

        pooled = self.pool(feature_map).flatten(1)
        projection = self.projector(pooled)
        return EncoderOutput(
            feature_map=feature_map,
            pooled_feature=pooled,
            embedding=normalize_embedding(projection),
            intermediate_feature_maps=intermediate,
            projection=projection,
            mri_proj_feats=mri_proj_feats,
            mri_proj_embs=mri_proj_embs,
            mri_depth_attn=mri_depth_attn,
        )
