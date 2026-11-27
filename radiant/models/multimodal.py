"""Device routing for paired X-ray and MRI encoders."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .mri_encoder import MRI3DEncoder, MRIEncoderConfig
from .outputs import EncoderOutput
from .slice_encoder import MRISlice2DEncoder
from .xray_encoder import XRay2DEncoder, XRayEncoderConfig


@dataclass(slots=True)
class MultiModalEncoderSystemConfig:
    """Configuration for a paired X-ray/MRI encoder system."""

    xray: XRayEncoderConfig = field(default_factory=XRayEncoderConfig)
    mri: MRIEncoderConfig = field(default_factory=MRIEncoderConfig)
    xray_device: str | None = None
    mri_device: str | None = None
    output_device: str | None = None


class MultiModalEncoderSystem(nn.Module):
    """Wrapper that can keep X-ray and MRI branches on one or two devices."""

    def __init__(self, config: MultiModalEncoderSystemConfig) -> None:
        super().__init__()
        self.config = config
        self.xray_encoder = XRay2DEncoder(config.xray)
        if config.mri.encoder_type == "volume_3d":
            self.mri_encoder: nn.Module = MRI3DEncoder(config.mri)
        elif config.mri.encoder_type == "slice_2d":
            self.mri_encoder = MRISlice2DEncoder(config.mri)
        elif config.mri.encoder_type == "shared_xray":
            self.mri_encoder = nn.Module()  # parameter-free: forward_mri uses the X-ray encoder
        else:
            raise ValueError(f"Unsupported mri_model.encoder_type {config.mri.encoder_type!r}.")

        xray_device = self._resolve_device(config.xray_device)
        mri_device = self._resolve_device(config.mri_device, fallback=xray_device)
        output_device = self._resolve_device(config.output_device, fallback=xray_device)

        self.xray_device = xray_device
        self.mri_device = mri_device
        self.output_device = output_device

        self.xray_encoder.to(self.xray_device)
        self.mri_encoder.to(self.mri_device)

    @staticmethod
    def _resolve_device(device_name: str | None, fallback: torch.device | None = None) -> torch.device:
        if device_name is None:
            return fallback or torch.device("cpu")
        return torch.device(device_name)

    def _move_output(self, output: EncoderOutput) -> EncoderOutput:
        if self.output_device is None:
            return output

        def move_optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return tensor.to(self.output_device) if tensor is not None else None

        return EncoderOutput(
            feature_map=output.feature_map.to(self.output_device),
            pooled_feature=output.pooled_feature.to(self.output_device),
            embedding=output.embedding.to(self.output_device),
            intermediate_feature_maps={
                name: feature.to(self.output_device)
                for name, feature in output.intermediate_feature_maps.items()
            },
            projection=output.projection.to(self.output_device),
            mri_proj_feats=move_optional(output.mri_proj_feats),
            mri_proj_embs=move_optional(output.mri_proj_embs),
            mri_depth_attn=move_optional(output.mri_depth_attn),
        )

    def forward_xray(self, inputs: torch.Tensor, move_to_output_device: bool = True) -> EncoderOutput:
        """Run the X-ray encoder on its assigned device."""
        output = self.xray_encoder(inputs.to(self.xray_device, non_blocking=True))
        return self._move_output(output) if move_to_output_device else output

    def forward_mri(
        self,
        inputs: torch.Tensor,
        move_to_output_device: bool = True,
        compute_projection_bridge: bool = True,
    ) -> EncoderOutput:
        """Run the MRI encoder on its assigned device."""
        if self.config.mri.encoder_type == "shared_xray":
            return self.forward_xray(inputs, move_to_output_device=move_to_output_device)
        output = self.mri_encoder(
            inputs.to(self.mri_device, non_blocking=True),
            compute_projection_bridge=compute_projection_bridge,
        )
        return self._move_output(output) if move_to_output_device else output

    def forward(
        self,
        xray: torch.Tensor | None = None,
        mri: torch.Tensor | None = None,
        move_to_output_device: bool = True,
        compute_projection_bridge: bool = True,
    ) -> dict[str, EncoderOutput]:
        """Run one or both encoders and return their outputs."""
        outputs: dict[str, EncoderOutput] = {}
        if xray is not None:
            outputs["xray"] = self.forward_xray(xray, move_to_output_device=move_to_output_device)
        if mri is not None:
            outputs["mri"] = self.forward_mri(
                mri,
                move_to_output_device=move_to_output_device,
                compute_projection_bridge=compute_projection_bridge,
            )
        if not outputs:
            raise ValueError("At least one of xray or mri must be provided.")
        return outputs
