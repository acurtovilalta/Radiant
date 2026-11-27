"""Run short, real-model RADIANT training for the four full-method configurations.

Uses synthetic images, small crops, one warm-up epoch and one joint epoch.
Pretrained variants load actual weights; cached weights allow an offline run.
Production model widths, depths, projection counts and loss weights are retained.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radiant.training.trainer import StageTrainer, TrainerConfig, _build_config, load_yaml

VARIANTS = ("resnet50_scratch", "resnet50_imagenet", "vitb16_scratch", "vitb16_dinov3")


def write_manifests(directory: Path) -> tuple[Path, Path]:
    """Create two anatomy families and two view labels in both dataset splits."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    manifests = []
    for modality, shape in (("xray", (64, 64)), ("mri", (32, 32, 32))):
        rows = []
        for split in ("train", "val"):
            for index in range(4):
                path = directory / f"{modality}_{split}_{index}.nii.gz"
                nib.save(nib.Nifti1Image(rng.standard_normal(shape).astype(np.float32), np.eye(4)), path)
                rows.append({
                    "filepath": str(path),
                    "split": split,
                    "modality": modality,
                    "patient_id": f"{modality}_{split}_{index}",
                    "study_id": f"{modality}_{split}_{index}",
                    "anatomy_family": "spine" if index < 2 else "knee",
                    "view_or_plane": "AP" if index % 2 == 0 else "lateral",
                    "sequence": "T1" if modality == "mri" else "",
                })
        manifest = directory / f"{modality}_manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        manifests.append(manifest)
    return manifests[0], manifests[1]


def smoke_config(
    variant: str,
    output: Path,
    manifests: tuple[Path, Path],
    device: str,
    mri_checkpoint: str | None,
) -> TrainerConfig:
    """Use the production architecture and losses with a short synthetic schedule."""
    config = _build_config(load_yaml(ROOT / "configs/radiant" / f"{variant}.yaml"))
    config.output_dir = str(output)
    config.devices = {key: device for key in ("xray", "mri", "output")}
    config.distributed.enabled = int(os.environ.get("WORLD_SIZE", "1")) > 1
    config.wandb.enabled = False
    # FP32 isolates model/gradient correctness from GradScaler warm-up overflows
    # on the tiny synthetic batches used by this short run.
    config.use_amp = False
    config.val_every_epochs = 1
    config.log_every_steps = 0
    config.logging.log_every_steps = 0
    config.log_bridge_gradient_audit = True
    config.bridge_gradient_audit_every_steps = 1
    config.runtime.cudnn_benchmark = False
    config.schedule = replace(
        config.schedule,
        warm_start_epochs=1,
        joint_epochs=1,
        cross_modal_ramp_epochs=1,
        xray_cnn_unfreeze_start_epoch=0,
        xray_cnn_unfreeze_stages_per_step=5,
        xray_vit_unfreeze_start_epoch=0,
        xray_vit_unfreeze_blocks_per_stage=12,
    )
    config.checkpoint = replace(config.checkpoint, output_dir=str(output / "checkpoints"), save_every_epochs=0, resume_path=None)
    config.mri_model = replace(config.mri_model, pretrained_checkpoint=mri_checkpoint)
    config.mri_assignment = replace(config.mri_assignment, queue_start_steps=0)
    # Keep every learned branch trainable during smoke validation, including the
    # gradually unfrozen pretrained backbones.
    for name, manifest in zip(("xray_data", "mri_data"), manifests):
        common = dict(
            manifest_path=str(manifest), path_column="filepath", anatomy_column="anatomy_family",
            split_group_column="patient_id", batch_size=4, num_workers=0, shuffle=False,
            pin_memory=False, shared_family_batching=True, families_per_batch=2,
        )
        if name == "xray_data":
            common.update(global_crop_size=64, local_crop_size=32, num_local_crops=2,
                          augmentations={"affine_prob": 0.0, "intensity_jitter_prob": 0.0,
                                         "gaussian_blur_prob": 0.0, "gaussian_noise_prob": 0.0})
        else:
            common.update(global_crop_size=(32, 32, 32), local_crop_size=(32, 32, 32), num_local_crops=2,
                          augmentations={"affine_prob": 0.0, "intensity_scale_prob": 0.0,
                                         "intensity_shift_prob": 0.0, "contrast_prob": 0.0,
                                         "gaussian_noise_prob": 0.0, "gaussian_smooth_prob": 0.0})
        setattr(config, name, replace(getattr(config, name), **common))
    return config


def run_variant(variant: str, directory: Path, manifests: tuple[Path, Path], device: str, mri_checkpoint: str | None) -> dict:
    trainer = StageTrainer(smoke_config(variant, directory, manifests, device, mri_checkpoint))
    before = next(trainer.xray_encoder_module.parameters()).detach().cpu().clone()
    summary = trainer.train()
    history = summary["history"]
    assert summary["epochs_completed"] == 2
    assert [record["train/stage"] for record in history] == ["warm_start", "joint"]
    assert all(math.isfinite(record["train/total_loss"]) for record in history)
    assert trainer.state.get("skipped_nonfinite_steps", 0) == 0
    assert history[0]["train/xray_view_loss"] > 0
    assert history[1]["train/xray_view_loss"] == 0
    assert history[1]["train/centroid_loss"] > 0
    assert history[1]["train/projection_mil_loss"] > 0
    assert history[1]["train/projection_attention_diversity_loss"] > 0
    assert not torch.equal(before, next(trainer.xray_encoder_module.parameters()).detach().cpu())
    assert (directory / "checkpoints/final.pt").is_file()
    # Reload the actual saved checkpoint to validate serialized model keys.
    checkpoint = torch.load(directory / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    trainer.xray_encoder_module.load_state_dict(checkpoint["xray_encoder"], strict=True)
    trainer.mri_encoder_module.load_state_dict(checkpoint["mri_encoder"], strict=True)
    del checkpoint
    result = {
        "variant": variant,
        "epochs": 2,
        "steps": summary["global_step"],
        "loss": history[1]["train/total_loss"],
        "mri_checkpoint": mri_checkpoint,
        "xray_weights_updated": True,
    }
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/smoke/full_method")
    parser.add_argument("--mri-checkpoint", help="Optional real pretrained MRI-encoder checkpoint; omitted for synthetic validation.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl")
        if rank == 0:
            write_manifests(directory / "data")
        torch.distributed.barrier()
        manifests = (directory / "data/xray_manifest.csv", directory / "data/mri_manifest.csv")
    else:
        manifests = write_manifests(directory / "data")
    results = []
    for variant in args.variants:
        print(f"Smoke training {variant} on {args.device}", flush=True)
        results.append(run_variant(variant, directory / variant, manifests, args.device, args.mri_checkpoint))
        if rank == 0:
            (directory / "smoke_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
