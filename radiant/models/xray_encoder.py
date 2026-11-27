"""2D X-ray encoders and backbone construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torchvision

from .dinov3_backbone import load_dinov3_vitb16_backbone
from .outputs import EncoderOutput
from .projection_head import ProjectionHead, normalize_embedding


@dataclass(slots=True)
class XRayEncoderConfig:
    """Configuration for the X-ray CNN or DINOv3 encoder."""

    backbone: str = "resnet50"
    in_channels: int = 1
    pretrained: bool = False
    model_id: str | None = None
    cache_dir: str | None = None
    embedding_dim: int = 256
    projector_hidden_dim: int = 1024
    projector_layers: int = 2


def _disable_inplace_activations(module: nn.Module) -> None:
    """Disable in-place activations to keep multi-view autograd graphs safe."""
    for child in module.modules():
        if isinstance(child, nn.ReLU):
            child.inplace = False


def _build_resnet2d_backbone(
    backbone: str,
    in_channels: int,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    if backbone == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        feature_dim = 512
    elif backbone == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet34(weights=weights)
        feature_dim = 512
    elif backbone == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        feature_dim = 2048
    else:
        raise ValueError("Unsupported X-ray ResNet backbone. Use resnet18, resnet34, or resnet50.")

    original_conv1 = net.conv1
    if in_channels != original_conv1.in_channels:
        net.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        if pretrained and in_channels == 1:
            with torch.no_grad():
                net.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))
    _disable_inplace_activations(net)
    return net, feature_dim


def _build_xray2d_backbone(config: XRayEncoderConfig) -> tuple[nn.Module, int]:
    if config.backbone == "dinov3_vitb16":
        if config.in_channels != 1:
            raise ValueError("dinov3_vitb16 currently supports only in_channels=1.")
        return load_dinov3_vitb16_backbone(
            pretrained=config.pretrained,
            model_id=config.model_id,
            cache_dir=config.cache_dir,
        )
    return _build_resnet2d_backbone(
        backbone=config.backbone,
        in_channels=config.in_channels,
        pretrained=config.pretrained,
    )


class XRay2DEncoder(nn.Module):
    """2D encoder (ResNet or DINOv3 ViT-B) for X-ray images with one projection head."""

    def __init__(self, config: XRayEncoderConfig) -> None:
        super().__init__()
        self.config = config
        backbone, feature_dim = _build_xray2d_backbone(config)
        self.backbone_name = config.backbone
        self.feature_dim = feature_dim

        self.stem = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.vit_backbone = None
        if config.backbone.startswith("resnet"):
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
        elif config.backbone == "dinov3_vitb16":
            self.vit_backbone = backbone

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = ProjectionHead(
            in_dim=feature_dim,
            out_dim=config.embedding_dim,
            hidden_dim=config.projector_hidden_dim,
            num_layers=config.projector_layers,
        )

    def _forward_resnet(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        intermediate: dict[str, torch.Tensor] = {}
        outputs = self.stem(inputs)
        intermediate["stem"] = outputs
        outputs = self.layer1(outputs)
        intermediate["layer1"] = outputs
        outputs = self.layer2(outputs)
        intermediate["layer2"] = outputs
        outputs = self.layer3(outputs)
        intermediate["layer3"] = outputs
        outputs = self.layer4(outputs)
        intermediate["layer4"] = outputs
        return outputs, intermediate

    def forward(self, inputs: torch.Tensor) -> EncoderOutput:
        """Encode X-rays and return features plus the normalized embedding."""
        pooled_override: torch.Tensor | None = None
        if self.backbone_name.startswith("resnet"):
            feature_map, intermediate = self._forward_resnet(inputs)
        else:
            feature_map, pooled_override = self.vit_backbone.forward_features_and_pool(inputs)
            intermediate = {"patch_tokens": feature_map}

        pooled = pooled_override if pooled_override is not None else self.pool(feature_map).flatten(1)
        projection = self.projector(pooled)
        return EncoderOutput(
            feature_map=feature_map,
            pooled_feature=pooled,
            embedding=normalize_embedding(projection),
            intermediate_feature_maps=intermediate,
            projection=projection,
        )


def get_resnet_stage_modules(xray_encoder: XRay2DEncoder) -> list[nn.Module]:
    """Return the X-ray ResNet backbone stages, ordered input -> output, for gradual unfreezing.

    Mirrors `dinov3_backbone.get_transformer_blocks`'s ordering convention: index 0 is
    closest to the input (the stem), the last index is closest to the output (layer4).
    """
    if not xray_encoder.backbone_name.startswith("resnet"):
        raise RuntimeError(
            f"get_resnet_stage_modules requires a resnet X-ray backbone, got "
            f"'{xray_encoder.backbone_name}'."
        )
    return [xray_encoder.stem, xray_encoder.layer1, xray_encoder.layer2, xray_encoder.layer3, xray_encoder.layer4]
