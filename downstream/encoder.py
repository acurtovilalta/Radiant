"""Frozen RADIANT X-ray encoder loaded from a pretraining checkpoint.

Only the X-ray backbone is used downstream; the projector, the MRI branch, the projection bridge,
the prototype banks and the auxiliary heads are discarded.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

from radiant.models.xray_encoder import XRay2DEncoder, XRayEncoderConfig


class FrozenXRayEncoder(nn.Module):
    """X-ray backbone of a RADIANT checkpoint, frozen and in eval mode.

    `pooled(x)` is the globally pooled backbone representation (classification and regression).
    `multiscale(x)` returns the final spatial feature map plus intermediate maps for the
    segmentation decoder: ResNet stages (`layer1`..`layer3`) as resolution skips, or earlier ViT
    blocks (`block_<i>`) as same-resolution auxiliary features.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        xray_config = dict(checkpoint["config"]["xray_model"])
        config_fields = {field.name for field in fields(XRayEncoderConfig)}
        config = XRayEncoderConfig(**{key: value for key, value in xray_config.items() if key in config_fields})
        # Every weight is overwritten by the checkpoint, so never download ImageNet / DINOv3 weights.
        config.pretrained = False
        self.encoder = XRay2DEncoder(config)
        self.encoder.load_state_dict(checkpoint["xray_encoder"])
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.backbone_name = config.backbone
        self.feature_dim = int(self.encoder.feature_dim)

    def train(self, mode: bool = True) -> "FrozenXRayEncoder":
        # Keep BatchNorm statistics fixed whatever the caller does.
        return super().train(False)

    @property
    def is_vit(self) -> bool:
        return self.encoder.vit_backbone is not None

    @torch.no_grad()
    def pooled(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images).pooled_feature

    @torch.no_grad()
    def multiscale(
        self, images: torch.Tensor, resnet_layers: tuple[str, ...] = (), vit_blocks: tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.is_vit:
            return self.encoder.vit_backbone.forward_multiscale(images, block_indices=vit_blocks)
        output = self.encoder(images)
        return output.feature_map, {name: output.intermediate_feature_maps[name] for name in resnet_layers}
