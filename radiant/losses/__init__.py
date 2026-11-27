"""SwAV, centroid alignment, projection MIL, projection-attention, and X-ray view losses."""

from .centroid_alignment import CentroidAlignmentConfig, centroid_alignment_loss
from .projection_attention import projection_attention_regularization
from .projection_mil import projection_mil_contrastive_loss, xray_family_mil_loss
from .swav import (
    FamilyFeatureQueue,
    SwAVAssigner,
    SwAVAssignmentConfig,
    SwAVAssignmentOutput,
    SwAVLossConfig,
    SwAVLossOutput,
    sinkhorn_knopp,
    swav_loss,
)
from .xray_view import XRayViewHead

__all__ = [
    "CentroidAlignmentConfig",
    "FamilyFeatureQueue",
    "SwAVAssigner",
    "SwAVAssignmentConfig",
    "SwAVAssignmentOutput",
    "SwAVLossConfig",
    "SwAVLossOutput",
    "XRayViewHead",
    "centroid_alignment_loss",
    "projection_attention_regularization",
    "projection_mil_contrastive_loss",
    "sinkhorn_knopp",
    "swav_loss",
    "xray_family_mil_loss",
]
