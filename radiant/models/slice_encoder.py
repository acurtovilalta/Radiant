"""2D MRI slice encoder for control C4 (SliceRADIANT), selected with `mri_model.encoder_type: slice_2d`.

A separate 2D ResNet encodes the K slices of a volume, so the input is [B, K, 1, H, W]. The K slice feature
maps replace the projection bridge's K depth projections: the same 2D head maps them to the [B, K, d] MIL
bag. Their mean-pooled feature gives the volume embedding used by MRI SwAV.
"""

from __future__ import annotations

import torch
from torch import nn

from .mri_encoder import MRIEncoderConfig
from .mri_projection_bridge import MRIProjection2DHead, MRIProjection2DHeadConfig
from .outputs import EncoderOutput
from .projection_head import ProjectionHead, normalize_embedding
from .xray_encoder import _build_resnet2d_backbone


class MRISlice2DEncoder(nn.Module):
    """2D ResNet (random init) over a bag of K MRI slices."""

    def __init__(self, config: MRIEncoderConfig) -> None:
        super().__init__()
        self.config = config
        net, self.feature_dim = _build_resnet2d_backbone(config.backbone, config.in_channels, pretrained=False)
        # Same attribute names as MRI3DEncoder, so the gradient audit finds `layer4`.
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = ProjectionHead(
            in_dim=self.feature_dim,
            out_dim=config.embedding_dim,
            hidden_dim=config.projector_hidden_dim,
            num_layers=config.projector_layers,
        )
        self.mri_projection_head = (
            MRIProjection2DHead(
                MRIProjection2DHeadConfig(in_channels=self.feature_dim, embedding_dim=config.mri_projection_embedding_dim)
            )
            if config.use_mri_projection_bridge
            else None
        )

    def forward(self, inputs: torch.Tensor, compute_projection_bridge: bool = True) -> EncoderOutput:
        if inputs.ndim != 5:
            raise ValueError(f"slice_2d MRI encoder expects [B, K, C, H, W], got {tuple(inputs.shape)}.")
        batch_size, num_slices = inputs.shape[:2]
        features = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs.flatten(0, 1))))))
        feature_map = features.reshape(batch_size, num_slices, *features.shape[1:])  # [B, K, C, h, w]
        pooled = self.pool(features).flatten(1).reshape(batch_size, num_slices, -1).mean(dim=1)
        mri_proj_embs = None
        if compute_projection_bridge and self.mri_projection_head is not None:
            mri_proj_embs = self.mri_projection_head(feature_map)
        projection = self.projector(pooled)
        return EncoderOutput(
            feature_map=feature_map,
            pooled_feature=pooled,
            embedding=normalize_embedding(projection),
            intermediate_feature_maps={},
            projection=projection,
            mri_proj_feats=feature_map if mri_proj_embs is not None else None,
            mri_proj_embs=mri_proj_embs,
            mri_depth_attn=None,
        )
