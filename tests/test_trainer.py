from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.training import (
    AblationConfig,
    CheckpointConfig,
    OptimizerConfig,
    StageScheduleConfig,
    TrainerConfig,
)
from radiant.training import trainer as trainer_module
from radiant.training.trainer import (
    StageTrainer,
    projection_attention_regularization,
)
from radiant.data.datasets import MRIDataConfig, XRayDataConfig
from radiant.models import MRIEncoderConfig, PrototypeBankConfig, XRayEncoderConfig
from radiant.losses import (
    CentroidAlignmentConfig,
    SwAVAssignmentConfig,
    SwAVLossConfig,
)

def _write_nifti(path: Path, array: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), str(path))
    return str(path)


def _build_manifest_files(tmp_path: Path) -> tuple[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    xray_paths = [
        _write_nifti(tmp_path / f"xray_{index}.nii.gz", np.random.randn(128, 128))
        for index in range(4)
    ]
    mri_paths = [
        _write_nifti(tmp_path / f"mri_{index}.nii.gz", np.random.randn(16, 64, 64))
        for index in range(4)
    ]
    xray_manifest = pd.DataFrame(
        [
            {
                "split": "train" if index < 3 else "val",
                "filepath": path,
                "patient_id": f"xp{index}",
                "anatomy_family": "Knee" if index % 2 == 0 else "Distal Arm",
                "anatomy_family_id": 4 if index % 2 == 0 else 0,
                "view_or_plane": "frontal",
            }
            for index, path in enumerate(xray_paths)
        ]
    )
    mri_manifest = pd.DataFrame(
        [
            {
                "split": "train" if index < 3 else "val",
                "filepath": path,
                "patient_id": f"mp{index}",
                "anatomy_family": "Knee" if index % 2 == 0 else "Distal Arm",
                "anatomy_family_id": 4 if index % 2 == 0 else 0,
            }
            for index, path in enumerate(mri_paths)
        ]
    )
    xray_path = tmp_path / "xray_manifest.csv"
    mri_path = tmp_path / "mri_manifest.csv"
    xray_manifest.to_csv(xray_path, index=False)
    mri_manifest.to_csv(mri_path, index=False)
    return str(xray_path), str(mri_path)


def _make_trainer_config(tmp_path: Path) -> TrainerConfig:
    xray_manifest, mri_manifest = _build_manifest_files(tmp_path)
    return TrainerConfig(
        output_dir=str(tmp_path / "outputs"),
        use_amp=False,
        log_every_steps=0,
        val_every_epochs=1,
        xray_data=XRayDataConfig(
            manifest_path=xray_manifest,
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            split="train",
            batch_size=2,
            shuffle=False,
            global_crop_size=64,
            local_crop_size=32,
            num_local_crops=1,
            augmentations={
                "affine_prob": 0.0,
                "intensity_jitter_prob": 0.0,
                "gaussian_blur_prob": 0.0,
                "gaussian_noise_prob": 0.0,
            },
        ),
        mri_data=MRIDataConfig(
            manifest_path=mri_manifest,
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            split="train",
            batch_size=1,
            shuffle=False,
            global_crop_size=(12, 48, 48),
            local_crop_size=(8, 32, 32),
            num_local_crops=1,
            augmentations={
                "affine_prob": 0.0,
                "intensity_scale_prob": 0.0,
                "intensity_shift_prob": 0.0,
                "contrast_prob": 0.0,
                "gaussian_noise_prob": 0.0,
                "gaussian_smooth_prob": 0.0,
            },
        ),
        xray_model=XRayEncoderConfig(
            backbone="resnet18",
            embedding_dim=32,
            projector_hidden_dim=64,
        ),
        mri_model=MRIEncoderConfig(
            backbone="resnet34",
            base_channels=8,
            embedding_dim=32,
            projector_hidden_dim=64,
        ),
        prototype_banks=PrototypeBankConfig(embedding_dim=32, prototypes_per_family=8),
        xray_assignment=SwAVAssignmentConfig(sinkhorn_iterations=10),
        mri_assignment=SwAVAssignmentConfig(
            sinkhorn_iterations=10,
            use_feature_queue=True,
            queue_size=8,
            queue_start_steps=0,
        ),
        xray_swav_loss=SwAVLossConfig(num_assignment_crops=2),
        mri_swav_loss=SwAVLossConfig(num_assignment_crops=2),
        centroid_alignment=CentroidAlignmentConfig(),
        lambda_centroid_alignment=0.5,
        schedule=StageScheduleConfig(
            warm_start_epochs=1,
            joint_epochs=1,
            cross_modal_ramp_epochs=1,
        ),
        optimizer=OptimizerConfig(lr=1e-3, grad_clip_norm=1.0, warmup_epochs=0),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "checkpoints"), save_every_epochs=1),
        devices={"xray": "cpu", "mri": "cpu", "output": "cpu"},
    )


def test_stage_schedule_progression(tmp_path: Path) -> None:
    trainer = StageTrainer(_make_trainer_config(tmp_path))

    assert trainer._iter_stage(0) == ("warm_start", 0.0)
    assert trainer._iter_stage(1) == ("joint", 1.0)
    assert trainer._iter_stage(2) == ("done", 0.0)


def test_checkpoint_resume_roundtrip(tmp_path: Path) -> None:
    trainer = StageTrainer(_make_trainer_config(tmp_path))
    trainer.state["epoch"] = 2
    trainer.state["global_step"] = 7
    checkpoint_path = trainer._save_checkpoint(epoch=2, tag="roundtrip")

    resumed = StageTrainer(
        replace(
            trainer.config,
            checkpoint=CheckpointConfig(
                output_dir=str(tmp_path / "checkpoints"),
                save_every_epochs=1,
                resume_path=str(checkpoint_path),
            ),
        )
    )

    assert resumed.state["epoch"] == 2
    assert resumed.state["global_step"] == 7


def test_global_prototype_scope_uses_one_shared_bank(tmp_path: Path) -> None:
    """Ablation A5: one prototype bank shared by every anatomy family."""
    config = _make_trainer_config(tmp_path)
    config.ablations = AblationConfig(prototype_scope="global")
    trainer = StageTrainer(config)
    assert trainer.config.prototype_banks.families == ("UNKNOWN",)
    assert list(trainer.prototype_banks_module.prototype_banks.keys()) == ["UNKNOWN"]


def test_enable_mri_false_forces_xray_only_schedule(tmp_path: Path) -> None:
    """RADIANT-X: enable_mri=False removes the joint stage and every cross-modal loss."""
    config = _make_trainer_config(tmp_path)
    config.ablations = AblationConfig(enable_mri=False)
    trainer = StageTrainer(config)

    assert trainer.config.schedule.joint_epochs == 0
    assert trainer.config.lambda_centroid_alignment == 0.0
    assert trainer._iter_stage(0) == ("warm_start", 0.0)
    assert trainer._iter_stage(1) == ("done", 0.0)


def test_enable_mri_false_never_feeds_mri_through_the_model(tmp_path: Path) -> None:
    """RADIANT-X: with enable_mri=False no MRI batch is ever encoded, so the MRI encoder gets no
    gradient and contributes nothing to the shared prototype banks (not merely a zero loss weight)."""
    config = _make_trainer_config(tmp_path)
    config.ablations = AblationConfig(enable_mri=False)
    # 3 xray train records at batch_size=3 -> one full batch, avoiding a
    # batch-of-1 tail (BatchNorm rejects batch size 1 in train mode).
    config.xray_data.batch_size = 3
    trainer = StageTrainer(config)

    mri_params = list(trainer.model_system.mri_encoder.parameters())
    assert mri_params, "fixture MRI encoder must expose trainable parameters"
    assert all(parameter.grad is None for parameter in mri_params)

    lr = trainer._set_lr(0)
    train_metrics = trainer._train_epoch(0, "warm_start", cross_modal_weight=0.0, lr=lr)
    assert train_metrics["train/mri_loss"] == 0.0
    assert train_metrics["train/xray_loss"] != 0.0
    assert all(parameter.grad is None for parameter in mri_params)
    assert any(
        parameter.grad is not None
        for parameter in trainer.model_system.xray_encoder.parameters()
        if parameter.requires_grad
    )

    val_metrics = trainer._validate(0, "warm_start", cross_modal_weight=0.0)
    assert val_metrics["val/mri_loss"] == 0.0


def test_xray_family_mil_loss_uses_same_family_xrays(tmp_path: Path) -> None:
    """Positive bag = other same-family X-rays; gradient reaches the embeddings. Weight 0 -> exact zero."""
    config = _make_trainer_config(tmp_path)
    config.lambda_xray_family_mil = 0.5
    trainer = StageTrainer(config)
    families = list(trainer.prototype_banks_module.prototype_banks.keys())
    batch_families = [families[i % 2] for i in range(4)]
    embedding = torch.randn(4, config.prototype_banks.embedding_dim, device=trainer.loss_device, requires_grad=True)
    loss = trainer._xray_family_mil_loss(embedding, batch_families)
    assert torch.isfinite(loss) and float(loss) > 0.0
    loss.backward()
    assert embedding.grad is not None and float(embedding.grad.abs().sum()) > 0.0

    # No anchor has a same-family partner -> no valid anchors -> zero loss.
    assert float(trainer._xray_family_mil_loss(embedding.detach(), ["f0", "f1", "f2", "f3"])) == 0.0

    config_off = _make_trainer_config(tmp_path / "off")
    trainer_off = StageTrainer(config_off)
    assert float(trainer_off._xray_family_mil_loss(embedding, batch_families)) == 0.0


def test_xray_family_mil_runs_in_xray_only_warm_start(tmp_path: Path) -> None:
    config = _make_trainer_config(tmp_path)
    config.ablations = AblationConfig(enable_mri=False)
    config.lambda_xray_family_mil = 0.5
    config.xray_data.batch_size = 3
    trainer = StageTrainer(config)
    lr = trainer._set_lr(0)
    train_metrics = trainer._train_epoch(0, "warm_start", cross_modal_weight=0.0, lr=lr)
    assert "train/xray_family_mil_loss" in train_metrics
    assert np.isfinite(train_metrics["train/xray_family_mil_loss"])
    assert train_metrics["train/mri_loss"] == 0.0

def test_xray_view_loss_is_only_added_when_active(tmp_path: Path) -> None:
    """The view-prediction loss is used in warm-up only (view_loss_active=True)."""
    config = _make_trainer_config(tmp_path)
    config.lambda_xray_view = 0.5
    trainer = StageTrainer(config)
    trainer._xray_view_loss = lambda embedding, view_labels: embedding.sum() * 0.0 + 1.0
    batch = next(iter(trainer.xray_train_loader))

    active = trainer._run_xray_step(batch, update_queue=False, train=False, view_loss_active=True)
    inactive = trainer._run_xray_step(batch, update_queue=False, train=False, view_loss_active=False)

    assert float(active["xray_view_loss"]) == pytest.approx(1.0)
    assert float(inactive["xray_view_loss"]) == 0.0
    assert float(active["xray_loss"]) == pytest.approx(float(active["xray_swav_loss"]) + 0.5)
    assert float(inactive["xray_loss"]) == pytest.approx(float(inactive["xray_swav_loss"]))


def test_final_checkpoint_is_the_last_epoch(tmp_path: Path) -> None:
    """Models are evaluated at their final checkpoint: final.pt holds the last epoch, no best.pt."""
    config = _make_trainer_config(tmp_path)
    # X-ray-only keeps the real training loop small; 3 records at batch_size=3 avoid a batch-of-1
    # tail (BatchNorm rejects batch size 1 in train mode).
    config.ablations = AblationConfig(enable_mri=False)
    config.xray_data.batch_size = 3
    config.schedule.warm_start_epochs = 2
    trainer = StageTrainer(config)
    summary = trainer.train()
    total_epochs = config.schedule.warm_start_epochs + config.schedule.joint_epochs
    checkpoint_dir = Path(config.checkpoint.output_dir)
    final = torch.load(checkpoint_dir / "final.pt", map_location="cpu", weights_only=False)
    assert final["epoch"] == total_epochs == summary["epochs_completed"]
    assert not (checkpoint_dir / "best.pt").exists()
    assert summary["final_checkpoint"] == str(checkpoint_dir / "final.pt")

def test_projection_mil_contrastive_loss_smoke() -> None:
    z_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    z_m_proj = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [1.0, 0.1, 0.0], [0.9, -0.1, 0.0]],
            [[0.0, 1.0, 0.0], [0.1, 0.8, 0.0], [-0.1, 1.0, 0.0], [0.0, 0.9, 0.1]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, stats = trainer_module.projection_mil_contrastive_loss(
        z_x=z_x,
        z_m_proj=z_m_proj,
        xray_families=["Knee", "Distal Arm"],
        mri_families=["Knee", "Distal Arm"],
        temperature=0.07,
    )

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0
    assert float(stats["valid_xray_anchors"].item()) == 2.0
    assert float(stats["positive_fraction"].item()) > 0.0

    loss.backward()
    assert z_m_proj.grad is not None
    assert float(z_m_proj.grad.abs().sum().item()) > 0.0
    assert z_x.grad is not None

    z_x_zero = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    z_m_proj_zero = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [1.0, 0.1, 0.0], [0.9, -0.1, 0.0]],
            [[0.0, 1.0, 0.0], [0.1, 0.8, 0.0], [-0.1, 1.0, 0.0], [0.0, 0.9, 0.1]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    zero_loss, zero_stats = trainer_module.projection_mil_contrastive_loss(
        z_x=z_x_zero,
        z_m_proj=z_m_proj_zero,
        xray_families=["Knee", "Distal Arm"],
        mri_families=["Hip", "Foot"],
        temperature=0.07,
    )

    assert float(zero_loss.item()) == 0.0
    assert float(zero_stats["valid_xray_anchors"].item()) == 0.0
    assert float(zero_stats["positive_fraction"].item()) == 0.0

def test_projection_attention_diversity_is_mean_squared_cosine() -> None:
    identical = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    non_overlapping = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    # Eq. 3: mean squared cosine between the depth attentions of distinct queries.
    assert float(projection_attention_regularization(identical)["diversity_loss"]) == pytest.approx(1.0, abs=1e-6)
    assert float(projection_attention_regularization(non_overlapping)["diversity_loss"]) == pytest.approx(0.0, abs=1e-6)
    partial = torch.tensor([[[0.6, 0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]])
    cos01 = (0.6 * 0.8 + 0.8 * 0.6) / 1.0
    expected = (cos01**2 + 0.0 + 0.0) / 3.0  # pairs (0,1), (0,2), (1,2)
    assert float(projection_attention_regularization(partial)["diversity_loss"]) == pytest.approx(expected, rel=1e-4)


def test_projection_attention_losses_backpropagate_when_weighted() -> None:
    mri_depth_attn = torch.tensor(
        [[[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    metrics = projection_attention_regularization(mri_depth_attn)
    (0.5 * metrics["diversity_loss"]).backward()

    assert mri_depth_attn.grad is not None
    assert torch.isfinite(mri_depth_attn.grad).all()
    assert float(mri_depth_attn.grad.abs().sum().item()) > 0.0

def test_centroid_ramp_starts_at_zero_and_finishes_on_tenth_joint_epoch(tmp_path: Path) -> None:
    config = _make_trainer_config(tmp_path)
    config.schedule = replace(config.schedule, warm_start_epochs=20, joint_epochs=46, cross_modal_ramp_epochs=10)
    trainer = StageTrainer(config)
    assert trainer._iter_stage(19) == ("warm_start", 0.0)
    assert trainer._iter_stage(20) == ("joint", 0.0)
    assert trainer._iter_stage(21)[1] == pytest.approx(1 / 9)
    assert trainer._iter_stage(29) == ("joint", 1.0)
    assert trainer._iter_stage(30) == ("joint", 1.0)


def _make_projection_trainer(tmp_path: Path) -> StageTrainer:
    config = _make_trainer_config(tmp_path)
    config.mri_model = replace(
        config.mri_model, use_mri_projection_bridge=True,
        mri_projection_num_projections=2, mri_projection_embedding_dim=32,
    )
    config.lambda_projection_mil_contrastive = 1.0
    config.lambda_projection_attention_diversity = 0.1
    return StageTrainer(config)


def _projection_batch(modality: str) -> dict[str, Any]:
    global_shape = (4, 1, 64, 64) if modality == "xray" else (4, 1, 12, 48, 48)
    local_shape = (4, 1, 32, 32) if modality == "xray" else (4, 1, 8, 32, 32)
    return {
        "global_views": [torch.randn(global_shape) for _ in range(2)],
        "local_views": [torch.randn(local_shape) for _ in range(2)],
        "anatomy_family": ["Knee", "Knee", "Distal Arm", "Distal Arm"],
    }


def test_mri_warmup_uses_only_swav_and_does_not_execute_bridge(tmp_path: Path) -> None:
    trainer = _make_projection_trainer(tmp_path)
    bridge = trainer.mri_encoder_module.mri_projection_bridge
    calls = []
    handle = bridge.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[0].shape[0]))
    outputs = trainer._run_mri_step(_projection_batch("mri"), projection_joint_path=False)
    handle.remove()
    assert calls == []
    assert outputs["mri_proj_embs"] is None
    assert outputs["projection_attention_diversity_loss"].item() == 0
    outputs["mri_loss"].backward()
    assert bridge.depth_queries.grad is None
    assert trainer.mri_encoder_module.conv1.weight.grad is not None


def test_bridge_executes_once_on_only_first_global_crop(tmp_path: Path) -> None:
    trainer = _make_projection_trainer(tmp_path)
    calls = []
    bridge = trainer.mri_encoder_module.mri_projection_bridge
    handle = bridge.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[0].shape[0]))
    outputs = trainer._encode_mri_views(_projection_batch("mri"), compute_projection_bridge=True)
    handle.remove()
    assert len(outputs) == 4
    assert calls == [4]
    assert outputs[0].mri_proj_embs is not None
    assert all(output.mri_proj_embs is None for output in outputs[1:])


def test_joint_mil_backpropagates_into_both_encoders(tmp_path: Path) -> None:
    trainer = _make_projection_trainer(tmp_path)
    x_batch, m_batch = _projection_batch("xray"), _projection_batch("mri")
    x_outputs = trainer._run_xray_step(x_batch, view_loss_active=False)
    m_outputs = trainer._run_mri_step(m_batch, projection_joint_path=True)
    loss, stats = trainer._projection_mil_loss(x_batch, m_batch, x_outputs, m_outputs)
    assert stats["valid_xray_anchors"].item() == 4
    loss.backward()
    for parameter in (trainer.xray_encoder_module.stem[0].weight,
                      trainer.mri_encoder_module.conv1.weight,
                      trainer.mri_encoder_module.mri_projection_bridge.depth_queries,
                      trainer.mri_encoder_module.mri_projection_head.projection.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_projection_mil_averages_over_all_xray_anchors() -> None:
    """Eq. 4: the loss is a mean over all B_X X-rays; anchors with an empty positive bag add 0."""
    torch.manual_seed(0)
    z_x = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    z_m_proj = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    mri_families = ["Knee", "Foot"]
    full, stats = trainer_module.projection_mil_contrastive_loss(
        z_x, z_m_proj, ["Knee", "Foot", "Hip"], mri_families, temperature=0.07
    )
    assert float(stats["valid_xray_anchors"]) == 2.0
    per_anchor = []
    for index, family in enumerate(["Knee", "Foot"]):
        single, _ = trainer_module.projection_mil_contrastive_loss(
            z_x[index : index + 1], z_m_proj, [family], mri_families, temperature=0.07
        )
        per_anchor.append(float(single))
    assert float(full) == pytest.approx(sum(per_anchor) / 3.0, rel=1e-5)


def test_xray_family_mil_averages_over_all_xray_anchors() -> None:
    from radiant.losses.projection_mil import xray_family_mil_loss

    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    with_singleton = xray_family_mil_loss(z, ["Knee", "Knee", "Hip"], temperature=0.07)
    pair_only = xray_family_mil_loss(z[:2], ["Knee", "Knee"], temperature=0.07)
    # The "Hip" anchor has no partner: it adds nothing but counts in B_X, and it is a negative
    # for the two Knee anchors, so compare against an explicit computation.
    sim = (z @ z.t()) / 0.07
    expected = 0.0
    for i, j in ((0, 1), (1, 0)):
        others = [k for k in range(3) if k != i]
        expected += -(sim[i, j] - torch.logsumexp(sim[i, others], dim=0)).item()
    assert float(with_singleton) == pytest.approx(expected / 3.0, rel=1e-5)
    # Two same-family X-rays and no negatives: the positive is the whole denominator -> 0.
    assert float(pair_only) == pytest.approx(0.0, abs=1e-6)
