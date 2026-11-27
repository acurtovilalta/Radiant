"""Build the task manifests (CSV: image_path, patient_id, label | mask_path) from the raw datasets.

Subcommands, one per dataset layout:

  image-folder   one sub-folder per class (DT1 internal tumors: benign/malign; DT2 FracAtlas
                 Fractured/Non_fractured; DT3 FracAtlas three classes; DT4 OAI KL0..KL4).
                 Labels are the sorted folder names mapped to 0..C-1.
  bone-age       RSNA bone age CSV (id, boneage, ...) + image folder (DT5).
  mask-folders   images/ and masks/ folders with matching file names (DT6 internal tumors).
  coco-masks     COCO JSON annotations + image folder; writes one binary mask PNG per image (DT7
                 FracAtlas fracture segmentation; requires pycocotools).

Patient ids: `--patient-map` (CSV with columns file_name, patient_id) takes precedence; otherwise
`--patient-regex` (first capture group of the file stem, e.g. `^(.+)_(?:left|right)$` for the OAI
knee crops); otherwise the file stem with a trailing `_<n>` crop index removed. Without a patient
map, datasets with several images per patient are NOT split at the patient level.

Examples:
    python -m downstream.prepare_manifests image-folder --root /data/FracAtlas/images_processed \
        --out data/downstream/dt2_fracture_binary.csv
    python -m downstream.prepare_manifests image-folder --root /data/osteoarthritis_classification \
        --patient-regex '^(.+)_(?:left|right)$' --out data/downstream/dt4_kl_grading.csv
    python -m downstream.prepare_manifests bone-age --csv /data/BoneAge/boneage-training-dataset.csv \
        --images /data/BoneAge/boneage-training-dataset --out data/downstream/dt5_bone_age.csv
    python -m downstream.prepare_manifests coco-masks --coco-json /data/FracAtlas/COCO_fracture_masks.json \
        --images /data/FracAtlas/images_processed/Fractured --masks-out data/downstream/dt7_masks \
        --out data/downstream/dt7_fracture_segmentation.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _patient_resolver(patient_map: Path | None, patient_regex: str | None):
    mapping = None
    if patient_map is not None:
        frame = pd.read_csv(patient_map)
        mapping = dict(zip(frame["file_name"].astype(str), frame["patient_id"].astype(str)))
    pattern = re.compile(patient_regex) if patient_regex else None

    def resolve(path: Path) -> str:
        if mapping is not None:
            if path.name not in mapping:
                raise KeyError(f"{path.name} is not in the patient map.")
            return mapping[path.name]
        if pattern is not None:
            match = pattern.match(path.stem)
            if match is None:
                raise ValueError(f"--patient-regex does not match {path.stem!r}.")
            return match.group(1)
        return re.sub(r"_\d+$", "", path.stem)

    return resolve


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def image_folder(args) -> pd.DataFrame:
    resolve = _patient_resolver(args.patient_map, args.patient_regex)
    classes = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    rows = [
        {"image_path": str(path.resolve()), "patient_id": resolve(path), "label": label, "class_name": name}
        for label, name in enumerate(classes)
        for path in _images(args.root / name)
    ]
    print("classes:", {name: label for label, name in enumerate(classes)})
    return pd.DataFrame(rows)


def bone_age(args) -> pd.DataFrame:
    table = pd.read_csv(args.csv)
    return pd.DataFrame(
        {
            "image_path": [str((args.images / f"{int(i)}.png").resolve()) for i in table["id"]],
            "patient_id": table["id"].astype(int).astype(str),
            "label": table["boneage"].astype(float),
        }
    )


def mask_folders(args) -> pd.DataFrame:
    resolve = _patient_resolver(args.patient_map, args.patient_regex)
    rows = [
        {"image_path": str(path.resolve()), "patient_id": resolve(path), "mask_path": str((args.masks / path.name).resolve())}
        for path in _images(args.images)
        if (args.masks / path.name).exists()
    ]
    return pd.DataFrame(rows)


def coco_masks(args) -> pd.DataFrame:
    from pycocotools.coco import COCO

    resolve = _patient_resolver(args.patient_map, args.patient_regex)
    coco = COCO(str(args.coco_json))
    args.masks_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in sorted(coco.getImgIds()):
        info = coco.loadImgs(image_id)[0]
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for annotation in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
            mask = np.maximum(mask, coco.annToMask(annotation).astype(np.uint8))
        image_path = args.images / info["file_name"]
        mask_path = args.masks_out / f"{Path(info['file_name']).stem}.png"
        Image.fromarray(mask * 255).save(mask_path)
        rows.append({"image_path": str(image_path.resolve()), "patient_id": resolve(image_path), "mask_path": str(mask_path.resolve())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func):
        command = sub.add_parser(name)
        command.set_defaults(func=func)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--patient-map", type=Path, default=None)
        command.add_argument("--patient-regex", default=None)
        return command

    add("image-folder", image_folder).add_argument("--root", required=True, type=Path)
    command = add("bone-age", bone_age)
    command.add_argument("--csv", required=True, type=Path)
    command.add_argument("--images", required=True, type=Path)
    command = add("mask-folders", mask_folders)
    command.add_argument("--images", required=True, type=Path)
    command.add_argument("--masks", required=True, type=Path)
    command = add("coco-masks", coco_masks)
    command.add_argument("--coco-json", required=True, type=Path)
    command.add_argument("--images", required=True, type=Path)
    command.add_argument("--masks-out", required=True, type=Path)

    args = parser.parse_args()
    frame = args.func(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(frame)} images, {frame['patient_id'].nunique()} patients")


if __name__ == "__main__":
    main()
