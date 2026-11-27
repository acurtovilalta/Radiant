"""Models Genesis pretraining of the C4 slice encoder and loading it into the RADIANT trainer's MRI branch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pretrain_mri  # noqa: E402
from radiant.data.datasets import MRIDataConfig  # noqa: E402
from radiant.models.mri_encoder import MRIEncoderConfig  # noqa: E402
from radiant.models.slice_encoder import MRISlice2DEncoder  # noqa: E402
from radiant.training.trainer import MRI_BACKBONE_STAGE_PREFIXES  # noqa: E402


def _slice_config() -> MRIEncoderConfig:
    payload = yaml.safe_load((ROOT / "configs/mri_pretraining/slice_encoder_genesis.yaml").read_text())["mri_model"]
    return MRIEncoderConfig(**payload)


def test_slice_genesis_model_reconstructs_every_slice() -> None:
    model = pretrain_mri.MRISliceGenesisModel(_slice_config())
    slices = torch.randn(2, 8, 1, 64, 64)
    reconstruction = model(slices)
    assert reconstruction.shape == (16, 1, 64, 64)
    target = pretrain_mri._reconstruction_target(slices, "slice_2d")
    assert target.shape == reconstruction.shape


def test_slice_pair_transform_corrupts_each_slice_of_one_shared_crop() -> None:
    config = MRIDataConfig(
        manifest_path="unused.csv",
        global_crop_size=64,
        local_crop_size=32,
        num_local_crops=0,
        slice_mode="evenly_spaced",
        num_slices=8,
        augmentations={"num_global_crops": 1, "intensity_jitter_prob": 0.0, "gaussian_blur_prob": 0.0,
                       "gaussian_noise_prob": 0.0, "affine_prob": 0.0},
    )
    transform = pretrain_mri._GenesisSlicePairTransform(config, pretrain_mri.GenesisAugmentationConfig(nonlinear_prob=1.0))
    (clean, corrupted), local_views = transform(torch.randn(8, 96, 96))
    assert local_views == []
    assert clean.shape == corrupted.shape == (8, 1, 64, 64)
    assert not torch.equal(clean, corrupted)


def test_slice_genesis_checkpoint_loads_into_the_c4_encoder(tmp_path: Path) -> None:
    config = _slice_config()
    model = pretrain_mri.MRISliceGenesisModel(config)
    path = tmp_path / "checkpoint_latest.pt"
    pretrain_mri._save_checkpoint(model, config, epoch=0, path=path)
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["encoder_type"] == "slice_2d"

    encoder = MRISlice2DEncoder(config)
    prefixes = tuple(MRI_BACKBONE_STAGE_PREFIXES.values())
    state = {key: value for key, value in checkpoint["mri_encoder_state_dict"].items() if key.startswith(prefixes)}
    result = encoder.load_state_dict(state, strict=False)
    assert not result.unexpected_keys
    # Every backbone weight is loaded; only the projector and the projection head stay random.
    assert all(key.startswith(("projector.", "mri_projection_head.")) for key in result.missing_keys)
    loaded = encoder.state_dict()
    assert all(np.allclose(loaded[key].numpy(), value.numpy()) for key, value in state.items())
