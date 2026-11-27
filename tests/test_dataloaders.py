from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.data.datasets import (
    MRIDataConfig,
    SharedFamilyBatchSampler,
    XRayDataConfig,
    build_mri_dataloader,
    build_xray_dataloader,
)


def _write_nifti(path: Path, array: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), str(path))
    return str(path)


def test_xray_dataloader_returns_expected_multicrop_batch(tmp_path: Path) -> None:
    image_a = _write_nifti(tmp_path / "xray_a.nii.gz", np.random.randn(256, 256))
    image_b = _write_nifti(tmp_path / "xray_b.nii.gz", np.random.randn(256, 256))
    manifest = pd.DataFrame(
        [
            {
                "split": "train",
                "filepath": image_a,
                "patient_id": "p1",
                "anatomy_family": "Distal Arm",
                "view_or_plane": "frontal",
            },
            {
                "split": "train",
                "filepath": image_b,
                "patient_id": "p2",
                "anatomy_family": "Distal Arm",
                "view_or_plane": "lateral",
            },
        ]
    )
    manifest_path = tmp_path / "train_xray.csv"
    manifest.to_csv(manifest_path, index=False)

    loader = build_xray_dataloader(
        XRayDataConfig(
            manifest_path=str(manifest_path),
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            batch_size=2,
            shuffle=False,
            global_crop_size=224,
            local_crop_size=96,
            num_local_crops=3,
        )
    )
    batch = next(iter(loader))

    assert len(batch["global_views"]) == 2
    assert len(batch["local_views"]) == 3
    assert batch["global_views"][0].shape == (2, 1, 224, 224)
    assert batch["local_views"][0].shape == (2, 1, 96, 96)
    assert torch.equal(batch["anatomy_family_id"], torch.zeros(2, dtype=torch.long))
    assert batch["view_label"] == ["frontal", "lateral"]


def test_mri_dataloader_returns_expected_multicrop_batch(tmp_path: Path) -> None:
    volume_a = _write_nifti(tmp_path / "mri_a.nii.gz", np.random.randn(32, 192, 192))
    volume_b = _write_nifti(tmp_path / "mri_b.nii.gz", np.random.randn(32, 192, 192))
    manifest = pd.DataFrame(
        [
            {
                "split": "train",
                "filepath": volume_a,
                "patient_id": "p3",
                "anatomy_family": "Foot",
            },
            {
                "split": "train",
                "filepath": volume_b,
                "patient_id": "p4",
                "anatomy_family": "Foot",
            },
        ]
    )
    manifest_path = tmp_path / "train_mri.csv"
    manifest.to_csv(manifest_path, index=False)

    loader = build_mri_dataloader(
        MRIDataConfig(
            manifest_path=str(manifest_path),
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            batch_size=2,
            shuffle=False,
            global_crop_size=(24, 128, 128),
            local_crop_size=(24, 96, 96),
            num_local_crops=1,
        )
    )
    batch = next(iter(loader))

    assert len(batch["global_views"]) == 2
    assert len(batch["local_views"]) == 1
    assert batch["global_views"][0].shape == (2, 1, 24, 128, 128)
    # Local views share the global crop's depth (24) — only H, W shrink.
    assert batch["local_views"][0].shape == (2, 1, 24, 96, 96)
    assert torch.equal(batch["anatomy_family_id"], torch.zeros(2, dtype=torch.long))
    assert batch["view_label"] == ["", ""]  # MRI has no view label


def test_xray_dataloader_auto_splits_without_split_column(tmp_path: Path) -> None:
    records = []
    for index in range(10):
        image_path = _write_nifti(tmp_path / f"xray_auto_{index}.nii.gz", np.random.randn(128, 128))
        records.append(
            {
                "filepath": image_path,
                "patient_id": f"patient_{index}",
                "anatomy_family": "Knee",
                "view_or_plane": "frontal",
            }
        )
    manifest_path = tmp_path / "xray_auto.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    train_loader = build_xray_dataloader(
        XRayDataConfig(
            manifest_path=str(manifest_path),
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            split="train",
            batch_size=2,
            shuffle=False,
            val_fraction=0.2,
            split_seed=7,
        )
    )
    val_loader = build_xray_dataloader(
        XRayDataConfig(
            manifest_path=str(manifest_path),
            anatomy_column="anatomy_family",
            path_column="filepath",
            split_group_column="patient_id",
            split="val",
            batch_size=2,
            shuffle=False,
            val_fraction=0.2,
            split_seed=7,
        )
    )

    train_patients = {record["patient_id"] for record in train_loader.dataset.records}
    val_patients = {record["patient_id"] for record in val_loader.dataset.records}

    assert len(train_loader.dataset.records) == 8
    assert len(val_loader.dataset.records) == 2
    assert train_patients.isdisjoint(val_patients)


def test_mri_dataloader_auto_split_is_deterministic(tmp_path: Path) -> None:
    records = []
    for index in range(8):
        volume_path = _write_nifti(tmp_path / f"mri_auto_{index}.nii.gz", np.random.randn(16, 64, 64))
        records.append(
            {
                "filepath": volume_path,
                "patient_id": f"patient_{index}",
                "anatomy_family": "Knee",
            }
        )
    manifest_path = tmp_path / "mri_auto.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    config = MRIDataConfig(
        manifest_path=str(manifest_path),
        anatomy_column="anatomy_family",
        path_column="filepath",
        split_group_column="patient_id",
        split="val",
        batch_size=2,
        shuffle=False,
        val_fraction=0.25,
        split_seed=11,
    )
    loader_a = build_mri_dataloader(config)
    loader_b = build_mri_dataloader(config)

    patients_a = [record["patient_id"] for record in loader_a.dataset.records]
    patients_b = [record["patient_id"] for record in loader_b.dataset.records]
    assert patients_a == patients_b


def test_shared_family_batch_sampler_draws_the_same_four_families_in_both_modalities() -> None:
    families = [
        "Distal Arm",
        "Proximal Arm",
        "Hip",
        "Foot",
        "Knee",
        "Chest",
        "Thigh",
        "Distal Leg",
    ]
    xray_families = [family for family in families for _ in range(5)]
    mri_families = list(xray_families)

    x_sampler = SharedFamilyBatchSampler(
        anatomy_families=xray_families,
        batch_size=8,
        num_batches=4,
        families_per_batch=4,
        shared_families=families,
        shuffle=True,
        seed=23,
    )
    m_sampler = SharedFamilyBatchSampler(
        anatomy_families=mri_families,
        batch_size=4,
        num_batches=4,
        families_per_batch=4,
        shared_families=families,
        shuffle=True,
        seed=23,
    )

    x_batches = list(iter(x_sampler))
    m_batches = list(iter(m_sampler))

    assert len(x_batches) == len(m_batches) == 4
    for x_batch, m_batch in zip(x_batches, m_batches):
        x_families = {xray_families[index] for index in x_batch}
        m_families = {mri_families[index] for index in m_batch}
        # Each batch pair covers the same four randomly drawn anatomy families in both modalities.
        assert x_families == m_families
        assert len(x_families) == 4


def test_patient_split_keeps_numeric_patient_ids_disjoint() -> None:
    from radiant.data.datasets import _apply_auto_split
    frame = pd.DataFrame({"patient_id": [patient for patient in range(20) for _ in range(2)]})
    train = _apply_auto_split(frame, "train", 0.05, 42, "patient_id")
    val = _apply_auto_split(frame, "val", 0.05, 42, "patient_id")
    assert set(train.patient_id).isdisjoint(val.patient_id)
    assert val.patient_id.nunique() == 1
    assert len(train) + len(val) == len(frame)


def test_explicit_splits_reject_patient_overlap() -> None:
    from radiant.data.datasets import _apply_auto_split
    frame = pd.DataFrame({"patient_id": [7, 7], "split": ["train", "val"]})
    with pytest.raises(ValueError, match="overlap"):
        _apply_auto_split(frame, "train", 0.05, 42, "patient_id")


@pytest.mark.parametrize("patient_id", [None, "", "unknown"])
def test_patient_split_rejects_missing_patient_ids(patient_id) -> None:
    from radiant.data.datasets import _apply_auto_split
    frame = pd.DataFrame({"patient_id": ["p1", patient_id]})
    with pytest.raises(ValueError, match="known patient/group ID"):
        _apply_auto_split(frame, "train", 0.05, 42, "patient_id")
