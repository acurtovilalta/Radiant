from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.data.transforms import (
    MRIAugmentationConfig,
    XRayAugmentationConfig,
    build_mri_transform,
    build_xray_transform,
)


def test_xray_safe_defaults_use_small_rotation_and_no_flips() -> None:
    transform = build_xray_transform(XRayAugmentationConfig())

    assert transform.config.rotation_degrees <= 5.0
    for pipeline in (transform.global_transform, transform.local_transform):
        assert not any("Flip" in type(step).__name__ for step in pipeline.transforms)


def test_xray_transform_respects_configurable_crop_counts() -> None:
    transform = build_xray_transform(
        XRayAugmentationConfig(
            global_size=128,
            local_size=64,
            num_global_crops=2,
            num_local_crops=3,
            affine_prob=0.0,
            intensity_jitter_prob=0.0,
            gaussian_blur_prob=0.0,
            gaussian_noise_prob=0.0,
        )
    )
    image = torch.randn(1, 256, 256)
    global_views, local_views = transform(image)

    assert len(global_views) == 2
    assert len(local_views) == 3
    assert global_views[0].shape == (1, 128, 128)
    assert local_views[0].shape == (1, 64, 64)


def test_mri_safe_defaults_use_small_in_plane_rotation_only() -> None:
    transform = build_mri_transform(MRIAugmentationConfig())

    assert transform.config.rotate_range_degrees == (0.0, 0.0, 3.0)
    assert transform.config.affine_prob <= 0.15


def test_mri_transform_respects_configurable_crop_counts() -> None:
    transform = build_mri_transform(
        MRIAugmentationConfig(
            global_size=(16, 64, 64),
            local_size=(8, 48, 48),
            num_global_crops=2,
            num_local_crops=1,
            affine_prob=0.0,
            intensity_scale_prob=0.0,
            intensity_shift_prob=0.0,
            contrast_prob=0.0,
            gaussian_noise_prob=0.0,
            gaussian_smooth_prob=0.0,
        )
    )
    volume = torch.randn(1, 24, 96, 96)
    global_views, local_views = transform(volume)

    assert len(global_views) == 2
    assert len(local_views) == 1
    assert global_views[0].shape == (1, 16, 64, 64)
    # Local views keep the full global-crop depth (16), not local_size[0] (8):
    # depth is cropped once per sample and shared by every view.
    assert local_views[0].shape == (1, 16, 48, 48)


def test_mri_transform_shares_one_depth_slab_across_all_views() -> None:
    # With intensity augmentations disabled, every global (and local) view must
    # come from the same depth window and only vary in-plane (H, W).
    config = MRIAugmentationConfig(
        global_size=(16, 64, 64),
        local_size=(16, 48, 48),
        num_global_crops=2,
        num_local_crops=2,
        intensity_scale_prob=0.0,
        intensity_shift_prob=0.0,
        contrast_prob=0.0,
        gaussian_noise_prob=0.0,
        gaussian_smooth_prob=0.0,
        affine_prob=0.0,
    )
    transform = build_mri_transform(config)

    # Encode each voxel's exact (d, h, w) coordinates as d * 10_000 + h * 100 + w
    # (h, w < 96 so the low-order digits never bleed into the depth digits).
    # This lets us recover the exact depth index sampled at any in-plane
    # position, independent of whichever (h, w) offset a view's crop used.
    depth = torch.arange(96, dtype=torch.float32).view(96, 1, 1)
    row = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    col = torch.arange(96, dtype=torch.float32).view(1, 1, 96)
    volume = (depth * 10_000 + row * 100 + col).unsqueeze(0)

    global_views, local_views = transform(volume)
    all_views = global_views + local_views

    def recovered_depth_sequence(view: torch.Tensor) -> torch.Tensor:
        # In-plane resampling mixes (h, w) values of one slice only (convex weights), so the
        # depth digits survive; +0.5 guards against float32 rounding just below a multiple.
        return torch.div(view[0, :, 0, 0] + 0.5, 10_000, rounding_mode="floor")

    reference_depths = recovered_depth_sequence(all_views[0])
    for view in all_views[1:]:
        assert view.shape[1] == reference_depths.shape[0]
        assert torch.equal(recovered_depth_sequence(view), reference_depths)

    # The two global views' in-plane content differs (different H/W crop offsets).
    assert not torch.equal(global_views[0][0, 0], global_views[1][0, 0])


def test_mri_views_are_random_resized_crops_in_plane() -> None:
    from radiant.data.transforms import RandResizedCropInPlane

    torch.manual_seed(0)
    crop = RandResizedCropInPlane(output_size=(64, 64), scale=(0.3, 0.6), ratio=(0.9, 1.1))
    volume = torch.randn(1, 8, 100, 100)
    boxes = {crop._crop_box(100, 100) for _ in range(20)}
    # Crop size varies (it is not a fixed-size crop) and covers the requested area fraction.
    assert len({(h, w) for _, _, h, w in boxes}) > 1
    for _, _, h, w in boxes:
        assert 0.25 * 100 * 100 <= h * w <= 0.65 * 100 * 100
    out = crop(volume)
    assert out.shape == (1, 8, 64, 64)

    # Full-area crop reduces to a plain in-plane resize; every depth slice is resampled on its own.
    full = RandResizedCropInPlane(output_size=(50, 50), scale=(1.0, 1.0), ratio=(1.0, 1.0))
    volume = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 100, 100).contiguous()
    resized = full(volume)
    assert resized.shape == (1, 4, 50, 50)
    assert torch.allclose(resized[0, :, 25, 25], torch.arange(4, dtype=torch.float32))


def test_mri_transform_resizes_crops_to_the_view_sizes() -> None:
    config = MRIAugmentationConfig(
        global_size=(16, 64, 64),
        local_size=(16, 32, 32),
        affine_prob=0.0,
        intensity_scale_prob=0.0,
        intensity_shift_prob=0.0,
        contrast_prob=0.0,
        gaussian_noise_prob=0.0,
        gaussian_smooth_prob=0.0,
    )
    # In-plane extent smaller than the global view: the crop is upsampled, not padded.
    global_views, local_views = build_mri_transform(config)(torch.randn(1, 20, 48, 56))
    assert all(view.shape == (1, 16, 64, 64) for view in global_views)
    assert all(view.shape == (1, 16, 32, 32) for view in local_views)
