"""The configs match the paper: RADIANT (Table 3, Sec. 2.4, App. B), RADIANT-X, ablations A1-A5 and controls C1-C4."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("resnet50_scratch", "resnet50_imagenet", "vitb16_scratch", "vitb16_dinov3")


@pytest.mark.parametrize("variant", VARIANTS)
def test_full_method_config_matches_paper_curriculum(variant: str) -> None:
    """Sec. 2.4 and App. B: warm-up/joint epochs, centroid ramp, crops and the patient-level split."""
    config = yaml.safe_load((ROOT / "configs/radiant" / f"{variant}.yaml").read_text())
    assert config["schedule"]["warm_start_epochs"] == (20 if variant.endswith("scratch") else 4)
    assert config["schedule"]["joint_epochs"] == 46
    assert config["schedule"]["cross_modal_ramp_epochs"] == 10
    for modality in ("xray_data", "mri_data"):
        assert config[modality]["num_local_crops"] == 2
        assert config[modality]["split_group_column"] == "patient_id"
    assert (config["xray_data"]["global_crop_size"], config["xray_data"]["local_crop_size"]) == (224, 96)
    assert config["mri_data"]["global_crop_size"] == [32, 192, 192]
    assert config["mri_data"]["local_crop_size"] == [32, 128, 128]
    assert config["mri_model"]["backbone"] == "resnet34"
    assert config["mri_model"]["stage_stride_schedule"] == "half_depth"
    expected_lr, expected_wd = (4.48e-4, 4.07e-6) if variant.startswith("resnet") else (1.35e-4, 4.26e-6)
    assert config["optimizer"]["lr"] == pytest.approx(expected_lr)
    assert config["optimizer"]["weight_decay"] == pytest.approx(expected_wd)


@pytest.mark.parametrize("name", ["mri_encoder_genesis", "slice_encoder_genesis"])
def test_models_genesis_pretraining_runs_50_epochs(name: str) -> None:
    """Sec. 2.1: the MRI encoder (and the C4 slice encoder) is pretrained for 50 epochs with Models Genesis."""
    config = yaml.safe_load((ROOT / "configs/mri_pretraining" / f"{name}.yaml").read_text())
    assert config["epochs"] == 50
    assert config["mri_model"]["backbone"] == "resnet34"


@pytest.mark.parametrize("variant", VARIANTS)
def test_radiant_x_has_the_same_unfreezing_policy(variant: str) -> None:
    full = yaml.safe_load((ROOT / "configs/radiant" / f"{variant}.yaml").read_text())
    control = yaml.safe_load((ROOT / "configs/radiant_x" / f"{variant}.yaml").read_text())
    prefixes = ("enable_xray_cnn_", "enable_xray_vit_", "xray_cnn_", "xray_vit_")
    full_policy = {key: value for key, value in full["schedule"].items() if key.startswith(prefixes)}
    control_policy = {key: value for key, value in control["schedule"].items() if key.startswith(prefixes)}
    assert control_policy == full_policy
    assert control["schedule"]["joint_epochs"] == 0
    # Matched control: same total training duration as full RADIANT (optimizer and loss weights are tuned separately).
    full_epochs = full["schedule"]["warm_start_epochs"] + full["schedule"]["joint_epochs"]
    assert control["schedule"]["warm_start_epochs"] == full_epochs


def test_c1_matches_radiant_x_resnet50_scratch_plus_xray_family_mil() -> None:
    base = _flatten(yaml.safe_load((ROOT / "configs/radiant_x/resnet50_scratch.yaml").read_text()))
    c1 = _flatten(yaml.safe_load((ROOT / "configs/controls/c1_xray_only_family_mil.yaml").read_text()))
    ignored = {("output_dir",), ("checkpoint", "output_dir"), ("wandb", "run_name"), ("wandb", "group"), ("wandb", "tags")}
    changed = {key for key in set(base) | set(c1) if base.get(key) != c1.get(key)} - ignored
    assert changed == {("lambda_xray_family_mil",), ("projection_mil_temperature",)}
    full = yaml.safe_load((ROOT / "configs/radiant/resnet50_scratch.yaml").read_text())
    assert c1[("lambda_xray_family_mil",)] == pytest.approx(full["lambda_projection_mil_contrastive"])


ABLATIONS = {
    "a1_global_pooling": {("mri_model", "mri_projection_num_projections"): 1, ("mri_model", "mri_projection_bridge_pooling_mode"): "global_average"},
    "a2_no_diversity": {("lambda_projection_attention_diversity",): 0.0},
    "a3_no_mil": {("lambda_projection_mil_contrastive",): 0.0},
    "a4_no_centroid_alignment": {("lambda_centroid_alignment",): 0.0},
    "a5_no_anatomy_conditioning": {("ablations", "prototype_scope"): "global"},
}


def _flatten(config: dict, prefix: tuple = ()) -> dict:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix + (key,)))
        else:
            flat[prefix + (key,)] = value
    return flat


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_ablation_differs_from_radiant_only_in_its_component(name: str) -> None:
    base = _flatten(yaml.safe_load((ROOT / "configs/radiant/resnet50_scratch.yaml").read_text()))
    ablation = _flatten(yaml.safe_load((ROOT / "configs/ablations" / f"{name}.yaml").read_text()))
    ignored = {("output_dir",), ("checkpoint", "output_dir"), ("wandb", "run_name"), ("wandb", "group"), ("wandb", "tags")}
    changed = {key for key in set(base) | set(ablation) if base.get(key) != ablation.get(key)} - ignored
    assert changed == set(ABLATIONS[name])
    for key, value in ABLATIONS[name].items():
        assert ablation[key] == value


# Paper Table 3 (rounded to the printed precision) and Sec. 3.1.
PAPER_TABLE3 = {
    "resnet50_scratch": {"lambda_projection_mil_contrastive": 0.037, "lambda_projection_attention_diversity": 0.042, "lambda_centroid_alignment": 3.87},
    "vitb16_scratch": {"lambda_projection_mil_contrastive": 0.034, "lambda_projection_attention_diversity": 0.032, "lambda_centroid_alignment": 1.65},
    "resnet50_imagenet": {"lambda_projection_mil_contrastive": 0.037, "lambda_projection_attention_diversity": 0.040, "lambda_centroid_alignment": 2.64},
    "vitb16_dinov3": {"lambda_projection_mil_contrastive": 0.036, "lambda_projection_attention_diversity": 0.039, "lambda_centroid_alignment": 1.90},
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_full_method_matches_paper_table3(variant: str) -> None:
    config = yaml.safe_load((ROOT / "configs/radiant" / f"{variant}.yaml").read_text())
    for key, value in PAPER_TABLE3[variant].items():
        decimals = len(str(value).split(".")[1])
        assert round(float(config[key]), decimals) == pytest.approx(value), key
    assert config["projection_mil_temperature"] == pytest.approx(0.07)
    assert config["mri_model"]["mri_projection_attention_temperature"] == pytest.approx(0.5)
    assert config["mri_model"]["mri_projection_num_projections"] == 8
    assert config["prototype_banks"]["prototypes_per_family"] == 96
    assert config["prototype_banks"]["embedding_dim"] == 256
    assert config["xray_data"]["batch_size"] == 32 and config["mri_data"]["batch_size"] == 16
    for modality in ("xray_data", "mri_data"):
        assert config[modality]["families_per_batch"] == 4
        assert config[modality]["val_fraction"] == pytest.approx(0.05)


FROM_RADIANT_X = "from radiant_x/resnet50_scratch.yaml"
SLICE_CONTROLS = {
    "c3_slice_swav": {
        ("log_bridge_gradient_audit",): False,
        ("mri_data", "global_crop_size"): 224,
        ("mri_data", "local_crop_size"): 96,
        ("mri_data", "slice_mode"): "random",
        ("mri_data", "num_slices"): 1,
        ("mri_data", "slice_central_fraction"): 0.5,
        ("mri_model", "encoder_type"): "shared_xray",
        ("mri_model", "use_mri_projection_bridge"): False,
        ("mri_model", "pretrained_checkpoint"): None,
        ("lambda_centroid_alignment",): 0.0,
        ("lambda_projection_mil_contrastive",): 0.0,
        ("lambda_projection_attention_diversity",): 0.0,
        ("lambda_mri_loss",): 0.25,
        # No cross-modal objective: optimizer and view weight come from RADIANT-X (checked below).
        ("optimizer", "lr"): FROM_RADIANT_X,
        ("optimizer", "weight_decay"): FROM_RADIANT_X,
        ("optimizer", "min_lr_ratio"): FROM_RADIANT_X,
        ("lambda_xray_view",): FROM_RADIANT_X,
    },
    "c4_slice_radiant": {
        ("mri_data", "global_crop_size"): 224,
        ("mri_data", "local_crop_size"): 96,
        ("mri_data", "slice_mode"): "evenly_spaced",
        ("mri_data", "num_slices"): 8,
        ("mri_data", "slice_central_fraction"): 0.5,
        ("mri_model", "encoder_type"): "slice_2d",
        ("mri_model", "pretrained_checkpoint"): "outputs/mri_pretraining_slice2d/checkpoint_latest.pt",
        ("lambda_projection_attention_diversity",): 0.0,
    },
}


@pytest.mark.parametrize("name", sorted(SLICE_CONTROLS))
def test_slice_controls_differ_from_radiant_only_in_the_mri_branch(name: str) -> None:
    base = _flatten(yaml.safe_load((ROOT / "configs/radiant/resnet50_scratch.yaml").read_text()))
    control = _flatten(yaml.safe_load((ROOT / "configs/controls" / f"{name}.yaml").read_text()))
    ignored = {("output_dir",), ("checkpoint", "output_dir"), ("wandb", "run_name"), ("wandb", "group"), ("wandb", "tags")}
    changed = {key for key in set(base) | set(control) if base.get(key) != control.get(key)} - ignored
    assert changed == set(SLICE_CONTROLS[name])
    radiant_x = _flatten(yaml.safe_load((ROOT / "configs/radiant_x/resnet50_scratch.yaml").read_text()))
    for key, value in SLICE_CONTROLS[name].items():
        assert control[key] == (radiant_x[key] if value == FROM_RADIANT_X else value), key
