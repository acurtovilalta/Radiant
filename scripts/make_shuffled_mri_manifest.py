"""Build the anatomy-shuffled MRI manifest used by control experiment C2.

C2 (configs/controls/c2_shuffled_mri_anatomy.yaml) is the full RADIANT method, unchanged, except
that it reads a copy of the MRI manifest whose anatomy labels are randomly permuted across MRI
volumes:

  - the permutation is done per volume (`volume_folder`), so every sub-volume of a volume keeps one
    shared (wrong) label;
  - label frequencies are preserved (it is a permutation, not a resampling);
  - it is seeded (default 13), so the manifest is reproducible.

Every anatomy-conditioned part of the MRI branch (family prototype banks, family batching,
cross-modal pairing, MIL positives, centroid alignment) then sees wrong anatomy.

Relative file paths are rewritten as absolute paths, because the dataset resolves relative paths
against the manifest's own folder and the copy may live elsewhere.

    python scripts/make_shuffled_mri_manifest.py \
        --input data/manifests/mri_manifest.csv \
        --output data/manifests/mri_manifest_shuffled_anatomy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANATOMY_COLUMN = "anatomy"
VOLUME_COLUMN = "volume_folder"
PATH_COLUMN = "file_path_processed"  # must match mri_data.path_column in the config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Original MRI manifest CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the shuffled manifest CSV.")
    parser.add_argument("--seed", type=int, default=13, help="Permutation seed (paper: 13).")
    parser.add_argument("--anatomy-column", default=ANATOMY_COLUMN)
    parser.add_argument("--volume-column", default=VOLUME_COLUMN)
    parser.add_argument("--path-column", default=PATH_COLUMN, help="Image-path column (mri_data.path_column).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    manifest = pd.read_csv(source, low_memory=False)
    if args.anatomy_column not in manifest or args.volume_column not in manifest:
        raise KeyError(f"{source} lacks '{args.anatomy_column}' or '{args.volume_column}'.")

    if args.path_column not in manifest:
        raise KeyError(f"{source} lacks the path column '{args.path_column}'.")
    manifest[args.path_column] = [
        value if not isinstance(value, str) or not value or Path(value).is_absolute()
        else str((source.parent / value).resolve())
        for value in manifest[args.path_column]
    ]

    per_volume = manifest.groupby(args.volume_column, sort=True)[args.anatomy_column].first()
    shuffled = pd.Series(np.random.default_rng(args.seed).permutation(per_volume.to_numpy()), index=per_volume.index)
    manifest[f"{args.anatomy_column}_original"] = manifest[args.anatomy_column]
    manifest[args.anatomy_column] = manifest[args.volume_column].map(shuffled)
    changed = float((manifest[args.anatomy_column] != manifest[f"{args.anatomy_column}_original"]).mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(per_volume)} volumes permuted (seed {args.seed}), {changed:.1%} of rows changed label.")


if __name__ == "__main__":
    main()
