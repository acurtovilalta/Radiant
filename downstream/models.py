"""Segmentation decoder trained on the frozen X-ray encoder's feature maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(32, out_channels), out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class MultiScaleSegDecoder(nn.Module):
    """Progressive-upsampling decoder with optional skip and auxiliary feature fusion.

    - `skip_channels` (ordered): U-Net-style skips. Stage i upsamples 2x and concatenates a 1x1
      projection of the i-th skip map, which must be 2^(i+1) times the deepest map's resolution
      (true for ResNet `layer3`, `layer2` after `layer4`).
    - `aux_channels`: same-resolution maps concatenated with the deepest map before the first conv
      block (earlier ViT blocks, which have the same patch-grid resolution as the final block).

    At least `min_stages` upsampling stages are always used; the logits are finally resized
    bilinearly to `output_size`. No dropout on the last hidden stage, to keep boundaries sharp.
    """

    def __init__(
        self,
        deepest_channels: int,
        skip_channels: dict[str, int],
        aux_channels: dict[str, int],
        output_size: int,
        base_width: int = 256,
        min_stages: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.output_size = int(output_size)
        self.skip_names = list(skip_channels)
        self.aux_names = list(aux_channels)
        self.in_block = _conv_block(deepest_channels + sum(aux_channels.values()), base_width, dropout)

        self.skip_projections = nn.ModuleList()
        self.stages = nn.ModuleList()
        width = base_width
        num_stages = max(min_stages, len(self.skip_names))
        for index in range(num_stages):
            out_width = max(32, width // 2)
            is_last = index == num_stages - 1
            stage_dropout = 0.0 if is_last else dropout
            if index < len(self.skip_names):
                self.skip_projections.append(nn.Conv2d(skip_channels[self.skip_names[index]], out_width, kernel_size=1))
                self.stages.append(_conv_block(width + out_width, out_width, stage_dropout))
            else:
                self.stages.append(_conv_block(width, out_width, 0.0))
            width = out_width
        self.out_conv = nn.Conv2d(width, 1, kernel_size=1)

    def forward(
        self, deepest: torch.Tensor, skips: dict[str, torch.Tensor], aux: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = torch.cat([deepest] + [aux[name] for name in self.aux_names], dim=1)
        x = self.in_block(x)
        for index, stage in enumerate(self.stages):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if index < len(self.skip_names):
                skip = self.skip_projections[index](skips[self.skip_names[index]])
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = stage(x)
        logits = self.out_conv(x)
        return F.interpolate(logits, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)


class FrozenEncoderSegmenter(nn.Module):
    """Frozen encoder -> MultiScaleSegDecoder. Only the decoder has trainable parameters."""

    def __init__(self, encoder: nn.Module, decoder: MultiScaleSegDecoder, resnet_layers: tuple[str, ...], vit_blocks: tuple[int, ...]) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.resnet_layers = resnet_layers
        self.vit_blocks = vit_blocks

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        deepest, intermediate = self.encoder.multiscale(images, self.resnet_layers, self.vit_blocks)
        skips = {name: intermediate[name] for name in self.decoder.skip_names}
        aux = {name: intermediate[name] for name in self.decoder.aux_names}
        return self.decoder(deepest.float(), {k: v.float() for k, v in skips.items()}, {k: v.float() for k, v in aux.items()})


@torch.no_grad()
def build_segmenter(encoder: nn.Module, image_size: int, decoder_config: dict, device: torch.device) -> FrozenEncoderSegmenter:
    """Decoder sized from one dummy forward pass of the encoder (channels of every fused map)."""
    resnet_layers = tuple(decoder_config.get("resnet_skip_layers", ())) if not encoder.is_vit else ()
    vit_blocks = tuple(int(i) for i in decoder_config.get("vit_aux_blocks", ())) if encoder.is_vit else ()
    dummy = torch.zeros(1, 1, image_size, image_size, device=device)
    deepest, intermediate = encoder.multiscale(dummy, resnet_layers, vit_blocks)
    skip_channels = {name: int(intermediate[name].shape[1]) for name in resnet_layers}
    aux_channels = {f"block_{i}": int(intermediate[f"block_{i}"].shape[1]) for i in vit_blocks}
    decoder = MultiScaleSegDecoder(
        deepest_channels=int(deepest.shape[1]),
        skip_channels=skip_channels,
        aux_channels=aux_channels,
        output_size=image_size,
        base_width=int(decoder_config.get("base_width", 256)),
        min_stages=int(decoder_config.get("min_stages", 2)),
        dropout=float(decoder_config.get("dropout", 0.2)),
    )
    return FrozenEncoderSegmenter(encoder, decoder, resnet_layers, vit_blocks).to(device)
