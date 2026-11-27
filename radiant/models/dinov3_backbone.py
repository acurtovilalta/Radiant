"""DINOv3 ViT-B backbone wrapper for grayscale X-ray inputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class Dinov3Backbone(nn.Module):
    """DINOv3 ViT-B wrapper exposing pooled and spatial features."""

    def __init__(
        self,
        *,
        pretrained: bool,
        model_id: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id or os.getenv("DINOV3_HF_MODEL", DEFAULT_DINOV3_MODEL_ID)
        self.cache_dir = cache_dir or os.getenv(
            "DINOV3_CACHE_DIR",
            str(Path(__file__).resolve().parents[2] / "dinov3_model_cache"),
        )
        self.model = self._build_model(pretrained=pretrained)
        self.hidden_size = int(self.model.config.hidden_size)

    def _build_model(self, *, pretrained: bool) -> nn.Module:
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel

            if pretrained:
                return DINOv3ViTModel.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                )

            # ViT-B/16 with register tokens, matching the pretrained architecture.
            # Random initialization requires neither downloaded weights nor a remote config.
            config = DINOv3ViTConfig(
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_register_tokens=4,
                patch_size=16,
                pos_embed_rescale=2.0,
            )
            return DINOv3ViTModel(config)
        except Exception as exc:  # pragma: no cover - exercised via message assertions
            raise RuntimeError(
                "Failed to initialize the DINOv3 ViT-B backbone. "
                "Install transformers and ensure the DINOv3 model/config is accessible. "
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _to_rgb_unit(inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        mins = inputs.amin(dim=(2, 3), keepdim=True)
        maxs = inputs.amax(dim=(2, 3), keepdim=True)
        denom = (maxs - mins).clamp_min(1e-8)
        return ((inputs - mins) / denom).clamp(0.0, 1.0)

    def _forward_outputs(self, inputs: torch.Tensor) -> Any:
        return self.model(pixel_values=self._to_rgb_unit(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self._forward_outputs(inputs)
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def _patch_tokens(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # DINOv3 embeddings concatenate [CLS, register_tokens..., patch_tokens...]; drop
        # both the CLS token and any register tokens so only spatial patch tokens remain.
        num_registers = getattr(self.model.config, "num_register_tokens", 0) or 0
        return hidden_state[:, 1 + num_registers :, :]

    @staticmethod
    def _tokens_to_spatial(tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = tokens.shape
        side = int(num_tokens**0.5)
        if side * side != num_tokens:
            return tokens.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        return tokens.reshape(batch_size, side, side, channels).permute(0, 3, 1, 2).contiguous()

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the spatial patch-token grid, excluding CLS and register tokens."""
        spatial, _ = self.forward_features_and_pool(inputs)
        return spatial

    def forward_features_and_pool(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return spatial patch features and the mean of patch and register tokens.

        Pooling all non-CLS tokens preserves the embeddings used by existing
        RADIANT projector checkpoints while spatial maps contain only patches.
        """
        hidden = self._forward_outputs(inputs).last_hidden_state
        pooled = hidden[:, 1:, :].mean(dim=1)
        return self._tokens_to_spatial(self._patch_tokens(hidden)), pooled

    def forward_multiscale(
        self, inputs: torch.Tensor, block_indices: tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Final patch-token grid plus the grids of earlier transformer blocks (`block_<i>`).

        `block_indices` index HF's `hidden_states` (0 = embeddings, i = output of block i). A ViT
        has no resolution pyramid, so these maps have the same resolution as the final one.
        """
        if not block_indices:
            return self.forward_features(inputs), {}
        outputs = self.model(pixel_values=self._to_rgb_unit(inputs), output_hidden_states=True)
        deepest = self._tokens_to_spatial(self._patch_tokens(outputs.last_hidden_state))
        blocks = {
            f"block_{i}": self._tokens_to_spatial(self._patch_tokens(outputs.hidden_states[i]))
            for i in block_indices
        }
        return deepest, blocks


def load_dinov3_vitb16_backbone(
    *,
    pretrained: bool,
    model_id: str | None = None,
    cache_dir: str | None = None,
) -> tuple[nn.Module, int]:
    """Build a DINOv3 ViT-B encoder and return it with feature width."""
    model = Dinov3Backbone(pretrained=pretrained, model_id=model_id, cache_dir=cache_dir)
    return model, int(model.hidden_size)


def get_transformer_blocks(vit_backbone: nn.Module) -> nn.ModuleList:
    """Return the ViT transformer block list for gradual unfreezing.

    Checks multiple attribute paths because the HF class actually loaded varies by
    installed ``transformers`` version: older/aliased ``Dinov3Model`` nests blocks at
    ``model.encoder.layer``, while the ``DINOv3ViTModel`` class used by transformers>=5.x
    nests them at ``model.model.layer`` (confirmed against the locally cached checkpoint).
    """
    model = getattr(vit_backbone, "model", None)
    candidates = (
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(model, "model", None), "layer", None),
        getattr(model, "layer", None),
        getattr(model, "layers", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError(
        "Could not locate transformer blocks on the DINOv3 backbone. Checked "
        "model.encoder.layer, model.model.layer, model.layer, and model.layers."
    )


def get_embedding_and_norm_modules(vit_backbone: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    """Return (patch/pos embeddings, final layernorm) modules if present.

    Handles both the ``layernorm`` attribute name (older/aliased ``Dinov3Model``) and
    ``norm`` (``DINOv3ViTModel`` on transformers>=5.x).
    """
    model = getattr(vit_backbone, "model", None)
    embeddings = getattr(model, "embeddings", None)
    final_norm = getattr(model, "layernorm", None) or getattr(model, "norm", None)
    return embeddings, final_norm
