# RADIANT

Self-supervised pretraining of a musculoskeletal X-ray encoder from **unpaired** X-rays and 3D MRI.

RADIANT trains a 2D X-ray encoder (ResNet50 or ViT-B/16) jointly with a 3D MRI encoder (ResNet34-3D).
Both use SwAV-style prototype assignment. A learned **MRI projection bridge** connects the two
modalities: it turns each MRI volume into `n` depth-attended 2D projections, which can be compared with
X-rays in one common embedding space. The X-ray and MRI images come from different patients.

This repository contains the full pretraining code, the configurations of every pretrained model
reported in the paper (RADIANT, RADIANT-X, ablations A1-A5, controls C1-C4), and the frozen-encoder
downstream evaluation on tasks DT1-DT7 (`downstream/`).

---

## Repository layout

```
Radiant/
├── train.py                     # RADIANT / RADIANT-X / control pretraining (single or multi-GPU)
├── pretrain_mri.py              # Models Genesis pretraining of the MRI encoder (3D; 2D slice encoder for C4), run first
├── configs/
│   ├── mri_pretraining/         # MRI-encoder pretraining config
│   ├── radiant/                 # RADIANT (X-ray + MRI): 4 backbone / init variants
│   ├── radiant_x/               # RADIANT-X (X-ray only) counterparts
│   ├── ablations/               # ablations A1-A5 (ResNet50, from scratch)
│   └── controls/                # control experiments C1-C4
├── radiant/                     # the library
│   ├── data/                    # anatomy labels, augmentations, X-ray / MRI datasets, family-synchronised batching,
│   │                            # 2D MRI slices (slices.py, controls C3/C4)
│   ├── models/                  # X-ray 2D + MRI 3D encoders, DINOv3 wrapper, projection bridge, prototype banks
│   ├── losses/                  # SwAV, centroid alignment, projection MIL, attention diversity, X-ray view loss
│   └── training/                # staged trainer
├── downstream/                  # frozen-encoder evaluation on DT1-DT7 (see downstream/README.md)
│   ├── configs/                 # one config per task, named as in the paper
│   ├── prepare_manifests.py, evaluate.py, significance.py
│   └── data.py, encoder.py, models.py, splits.py, engine.py
├── scripts/
│   ├── make_shuffled_mri_manifest.py   # builds the MRI manifest for control C2
│   └── evaluate_embedding_space.py     # post-hoc embedding-space diagnostic (modality mixing / anatomy)
└── tests/                       # unit tests (pytest)
```

---

## Model layout

Encoder configurations live beside their implementations. Public classes remain
available through `from radiant.models import ...`.

| Module under `radiant/models/` | Responsibility |
|---|---|
| `xray_encoder.py` | X-ray CNN/ViT encoder, backbone construction, ResNet stage access |
| `mri_encoder.py` | 3D MRI ResNet, stride schedules, projection bridge integration |
| `multimodal.py` | Paired encoder setup and device routing |
| `outputs.py` | Shared `EncoderOutput` container |
| `projection_head.py` | Shared MLP projector and embedding normalization |
| `dinov3_backbone.py` | DINOv3 initialization, spatial features, pooled embeddings, ViT stage access |
| `mri_projection_bridge.py` | MRI depth attention and shared 2D projection head |
| `slice_encoder.py` | 2D MRI slice encoder of control C4 (`mri_model.encoder_type: slice_2d`) |
| `prototypes.py` | Anatomy-family prototype banks |

## Method overview

| Component | Where | Config fields |
|---|---|---|
| X-ray encoder (ResNet50 or DINOv3 ViT-B/16) + one projection head → L2-normalized embedding | `radiant/models/xray_encoder.py`, `radiant/models/dinov3_backbone.py` | `xray_model.*` |
| 3D MRI encoder (ResNet34-3D, half-depth strides), initialised from the Models Genesis MRI-encoder pretraining, + one projection head into the same embedding space | `radiant/models/mri_encoder.py` | `mri_model.*` |
| MRI projection bridge: `n` learned depth queries → `n` 2D MRI projections | `radiant/models/mri_projection_bridge.py` | `mri_model.use_mri_projection_bridge`, `mri_projection_num_projections`, `mri_projection_attention_temperature` |
| Anatomy-family prototype banks, shared by both modalities | `radiant/models/prototypes.py` | `prototype_banks.*` |
| Intra-modal multi-crop SwAV with per-family Sinkhorn codes (+ per-family feature queue for MRI) | `radiant/losses/swav.py` | `xray_assignment`, `mri_assignment`, `xray_swav_loss`, `mri_swav_loss` |
| Projection MIL contrastive loss (same-anatomy MRI projections as positive bag) | `radiant/losses/projection_mil.py` | `lambda_projection_mil_contrastive`, `projection_mil_temperature` |
| Prototype-centroid alignment: the X-ray and MRI centroids of the same prototype are pulled together | `radiant/losses/centroid_alignment.py` | `lambda_centroid_alignment`, `centroid_alignment.*` |
| Projection-attention diversity | `radiant/losses/projection_attention.py` | `lambda_projection_attention_diversity` |
| X-ray view/plane prediction from the X-ray embedding, **warm-up stage only** | `radiant/losses/xray_view.py` | `lambda_xray_view`, `xray_data.view_column` |
| Staged schedule: independent modality warm-up → joint X-ray + MRI with a cross-modal ramp | trainer | `schedule.*` |
| Gradual top-down unfreezing for pretrained initialisations | trainer | `schedule.enable_xray_{cnn,vit}_gradual_unfreeze`, `xray_{cnn,vit}_unfreeze_*` |
| Evaluated checkpoint: `final.pt` = the final epoch (no checkpoint selection) | trainer | `checkpoint.*` |

The MRI encoder is pretrained first; RADIANT training then has two stages (warm-up and joint):

0. **MRI encoder pretraining** (`pretrain_mri.py`). The 3D MRI encoder learns to restore Models
   Genesis–style corrupted crops (non-linear intensity remap, local pixel shuffle, in-/out-painting).
   Only the encoder backbone is kept. You run this stage once, and every RADIANT config loads its output.
1. **RADIANT pretraining** (`train.py`). This is the multimodal SwAV + projection-bridge training described above:
   - *warm-up*: X-ray SwAV + X-ray view prediction, and MRI SwAV, trained separately.
     The bridge and diversity regularizer are inactive. Full scratch variants use 20 epochs;
     pretrained X-ray variants use 4. MRI reconstruction remains the separate MRI-encoder pretraining job (step 0);
   - *joint*: X-ray and MRI SwAV on anatomy-synchronised batches plus the cross-modal losses
     (projection MIL contrastive loss and prototype-centroid alignment, the latter
     ramped from zero on the first joint epoch to full weight on the tenth).
     The bridge runs on only the first global MRI crop; MIL and diversity start at full
     weights, and gradients flow through both encoders. The X-ray view prediction loss is
     switched off. All full variants use 46 joint epochs and two global plus two local
     crops per modality. Crops are random resized crops in both modalities: X-ray views are
     resized to 224 / 96 px; MRI views share one 32-slice slab and take an in-plane random
     resized crop resized to 32×192×192 / 32×128×128 (depth is never cropped or resampled).
     The MIL loss is averaged over all X-rays of the per-GPU batch (Eq. 4); an X-ray with no
     same-family MRI in the batch contributes zero. Pretrained X-ray backbones stay frozen for the four warm-up
     epochs, then gradually unfreeze from the output stages toward the input.

---

## Installation

```bash
git clone https://github.com/acurtovilalta/Radiant.git && cd Radiant
uv sync --all-extras         # exact environment from uv.lock
# or, with pip:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 && pip install -e . --no-deps
```

All dependencies are pinned (`pyproject.toml`, `uv.lock`, `requirements.txt`): Python 3.11, PyTorch 2.11.0 and
torchvision 0.26.0 (CUDA 12.6 wheels). Scratch ViT-B/16
initializes locally without downloading a configuration or weights. Pretrained DINOv3 weights are
downloaded from Hugging Face (`facebook/dinov3-vitb16-pretrain-lvd1689m`, a gated model, so run
`huggingface-cli login` first). They are cached in `./dinov3_model_cache`. You can override the cache
location with `DINOV3_CACHE_DIR` and the model ID with `DINOV3_HF_MODEL`.

---

## Full-method smoke validation

```bash
python scripts/smoke_full_method.py --device cuda
```

This runs all four `configs/radiant/` variants with real ResNet50/ViT-B models,
small synthetic crops, one warm-up epoch and one joint epoch in FP32.
ImageNet and DINOv3 variants load actual pretrained weights; cached weights permit
an offline run with `HF_HUB_OFFLINE=1`. Scratch ViT-B requires no model download.
The runner checks finite losses, parameter updates, view-loss staging, cross-modal
losses, active feature queues and saved checkpoint loading. Results are written to
`outputs/smoke/full_method/smoke_results.json` by default.

Use `--variants resnet50_scratch vitb16_scratch` for scratch-only validation,
`--device cpu` for CPU execution, or `--mri-checkpoint PATH` to additionally load a
real pretrained MRI-encoder checkpoint. Without that option, MRI weights are randomly
initialized for this synthetic check. Production configs retain their MRI-encoder
checkpoint paths. The smoke run shortens gradual unfreezing so pretrained
backbones can receive updates within two epochs; it does not measure training
quality or validate full-resolution distributed/AMP training.

## Data

**Data and model availability.** The internal TUM University Hospital data (all pretraining MRI
volumes, the internal pretraining X-rays, and the downstream DT1 and DT6 cohorts) cannot be shared,
because of patient privacy and the ethics approval. The pretrained RADIANT weights are not released
either. The repository therefore contains the code and the exact configurations, but the reported
models cannot be retrained or re-evaluated from it without access to the data. Public datasets used
in the paper: MURA (pretraining), FracAtlas (DT2, DT3, DT7), OAI (DT4) and RSNA Bone Age (DT5).

All config paths are relative to the repository root. Create a `data/` folder (it is git-ignored) with the following files, or edit the paths in the configs:

```
data/
└── manifests/
    ├── xray_manifest.csv                     # one row per 2D X-ray
    └── mri_manifest.csv                      # one row per 3D MRI volume
```

**Image files.** X-rays are single-channel NIfTI (`.nii.gz`) images preprocessed to 512×512. MRI
sub-volumes are NIfTI files preprocessed to 32×256×256 (depth × H × W). Relative image paths are
resolved against the manifest's own folder.

**Manifest columns used by the code**

The columns used for training are named in the config, per modality (`xray_data.*`, `mri_data.*`).
The names below are the ones used in the paper.

| | X-ray manifest | MRI manifest |
|---|---|---|
| image path (config: `path_column`) | `file_path_processed` | `file_path_processed` |
| anatomy family (config: `anatomy_column`) | `coarse_anatomy` | `anatomy` |
| X-ray view (config: `view_column`) | `view_or_plane` (frontal / lateral / oblique / axial / special) | – |
| patient ID for the automatic 5 % val split (config: `split_group_column`) | `patient_id` | `patient_id` |
| optional | `split` (`train`/`val`), with patient-disjoint train/val assignments | `split`; `volume_folder` (needed for C2); `kw_ImageOrientationPatient` and `slice_axis` (C3/C4) |

Patient IDs are required, including when explicit `split` labels are supplied.
Missing IDs and patients appearing in both train and validation are rejected.

The eight anatomy families are `Distal Arm`, `Proximal Arm`, `Chest`, `Knee`, `Foot`, `Thigh`, `Hip` and
`Distal Leg` (`radiant/data/anatomy.py`). Each family gets its own prototype bank. Manifests may also use the
legacy labels `Upper Limb – Distal`, `Upper Limb – Proximal`, `Chest / Thorax`, `Hip / Thigh`, `Abdomen / Pelvis`
and `Lower Leg`, which are mapped to these names when the data are loaded.
Batches are built from `families_per_batch` families at a time.

---

## Training

The paper models were trained on 2 GPUs with DDP. The batch sizes in the configs are per GPU
(X-ray 32, MRI 16).

```bash
# Step 0 -- MRI encoder pretraining (once). Writes outputs/mri_pretraining/checkpoint_latest.pt
python pretrain_mri.py --config configs/mri_pretraining/mri_encoder_genesis.yaml

# Step 1 -- any model from the table below
torchrun --nproc_per_node=2 train.py --config configs/radiant/resnet50_scratch.yaml
```

**Seeds.** Every config sets `seed: 13` (the training seed; each DDP rank uses `seed + rank`), and the
X-ray/MRI train/validation split uses `split_seed: 42`. To train with another seed, change `seed` in the
config. All results in the paper come from a single pretraining run per model (seed 13). The downstream
bootstrap captures sampling variability of the evaluation, not variance across pretraining seeds.

Outputs, including `checkpoints/epoch_XXXX.pt`, `checkpoints/final.pt` and `training_summary.json`, are
written to `outputs/<group>/<name>/`. Weights & Biases logging is on by default. To turn it off, set
`wandb.enabled: false` or `WANDB_MODE=disabled`. Set `checkpoint.resume_path` to resume a run.

The X-ray encoder weights are stored under the `xray_encoder` key of each checkpoint. Models are
evaluated at their final checkpoint: `final.pt` is written once, after the last epoch, and is the
file to pass to `downstream/`. Validation losses (every `val_every_epochs`) are logged for monitoring
only and do not select a checkpoint. `latest.pt` is the resume checkpoint.

### Configurations

Hyperparameters (learning rates, loss weights, schedules) are the tuned values of the paper runs.
All configs share the same options. [`configs/radiant/resnet50_imagenet.yaml`](configs/radiant/resnet50_imagenet.yaml)
is the annotated reference: a comment beside every option explains it and lists its accepted values.

**RADIANT (X-ray + MRI)**

| Config | X-ray backbone | Init | Epochs (warm-up + joint) |
|---|---|---|---|
| `configs/radiant/resnet50_scratch.yaml` | ResNet50 | random | 20 + 46 |
| `configs/radiant/vitb16_scratch.yaml` | ViT-B/16 | random | 20 + 46 |
| `configs/radiant/resnet50_imagenet.yaml` | ResNet50 | ImageNet, gradual unfreeze | 4 + 46 |
| `configs/radiant/vitb16_dinov3.yaml` | ViT-B/16 | DINOv3, gradual unfreeze | 4 + 46 |

**RADIANT-X (X-ray only).** The MRI branch and every cross-modal loss are switched off
with `ablations.enable_mri: false`. The X-ray view loss stays on for the whole run.
Each variant uses the same backbone-unfreezing policy as its full RADIANT counterpart,
including the first four frozen epochs for pretrained variants. Each variant trains for the
same number of epochs as its full RADIANT counterpart (66 from scratch, 50 when initialized).
Its optimizer settings and loss weights were tuned separately, with the same random search as full RADIANT.
C1 likewise trains for 66 epochs.

| Config | X-ray backbone | Init | Epochs |
|---|---|---|---|
| `configs/radiant_x/resnet50_scratch.yaml` | ResNet50 | random | 66 |
| `configs/radiant_x/vitb16_scratch.yaml` | ViT-B/16 | random | 66 |
| `configs/radiant_x/resnet50_imagenet.yaml` | ResNet50 | ImageNet, gradual unfreeze | 50 |
| `configs/radiant_x/vitb16_dinov3.yaml` | ViT-B/16 | DINOv3, gradual unfreeze (same recipe as `radiant/vitb16_dinov3.yaml`) | 50 |

**Ablations (A1-A5).** Each is `radiant/resnet50_scratch.yaml` with exactly one change.

| Config | Ablation | Change |
|---|---|---|
| `configs/ablations/a1_global_pooling.yaml` | **A1**: projection bridge → global pooling | `mri_projection_bridge_pooling_mode: global_average` (one MRI embedding, n=1) |
| `configs/ablations/a2_no_diversity.yaml` | **A2**: no diversity regularizer | `lambda_projection_attention_diversity: 0` |
| `configs/ablations/a3_no_mil.yaml` | **A3**: no MIL alignment | `lambda_projection_mil_contrastive: 0` |
| `configs/ablations/a4_no_centroid_alignment.yaml` | **A4**: no centroid alignment | `lambda_centroid_alignment: 0` |
| `configs/ablations/a5_no_anatomy_conditioning.yaml` | **A5**: no anatomy conditioning | `ablations.prototype_scope: global` (one bank of 96 prototypes shared by all families) |

**Controls.** Each control reuses the hyperparameters of the method it modifies. C1 is `radiant_x/resnet50_scratch.yaml`
plus the MIL term (weight = RADIANT's λ_MIL). C2 and C4 are `radiant/resnet50_scratch.yaml` with one component changed.
C3 has no cross-modal objective, so it takes RADIANT-X's optimizer and view weight with RADIANT's 20 + 46 schedule;
its MRI loss weight (0.25) is the only additional tuned hyperparameter.

| Config | What it tests |
|---|---|
| `configs/controls/c1_xray_only_family_mil.yaml` | **C1**: RADIANT-X plus the MIL objective, with the MRI projections replaced by same-family X-ray embeddings from the batch. Does extra family-level supervision explain the gains? |
| `configs/controls/c2_shuffled_mri_anatomy.yaml` | **C2**: full RADIANT, but MRI anatomy labels are permuted across MRI volumes. Does the anatomy correspondence between modalities matter? |
| `configs/controls/c3_slice_swav.yaml` | **C3** (SliceSwAV): no 3D MRI branch. One shared 2D encoder (the X-ray backbone, projector and prototype banks) is trained on X-rays plus 1 random MRI slice per volume per step (central 50%), with X-ray SwAV + view loss + slice SwAV (`lambda_mri_loss: 0.25`). No bridge, MIL or centroid alignment. Is a dedicated 3D MRI encoder needed? |
| `configs/controls/c4_slice_radiant.yaml` | **C4** (SliceRADIANT): full RADIANT with the 3D encoder + projection bridge replaced by a separate 2D ResNet34 over 8 evenly spaced slices (central 50%); the 8 slice embeddings form the MIL bag, L_div is dropped. The 2D encoder is initialised by Models Genesis pretraining on the same slices, like RADIANT's 3D encoder. Is the learned 3D projection bridge needed? |

C4 needs its own Models Genesis checkpoint for the 2D slice encoder. Build it once before training
(writes `outputs/mri_pretraining_slice2d/checkpoint_latest.pt`):

```bash
python pretrain_mri.py --config configs/mri_pretraining/slice_encoder_genesis.yaml
```

C2 needs the shuffled manifest. Build it once before training:

```bash
python scripts/make_shuffled_mri_manifest.py \
    --input data/manifests/mri_manifest.csv \
    --output data/manifests/mri_manifest_shuffled_anatomy.csv
```

### Hyperparameter search space

Pretraining hyperparameters were chosen by random search. Each trial draws every hyperparameter
independently; ranges marked log-uniform are sampled uniformly in log space. Everything not listed
(temperatures τ_c = 0.07 and τ_p = 0.5, batch sizes, crops, prototypes, schedule) is fixed to the
values in the configs.

| Hyperparameter | Config field | Search space |
|---|---|---|
| Learning rate, X-ray encoder from scratch | `optimizer.lr` | log-uniform [1e-4, 9e-4] |
| Learning rate, pretrained X-ray encoder | `optimizer.lr` | log-uniform [3e-5, 6e-4] |
| Weight decay | `optimizer.weight_decay` | log-uniform [3.33e-6, 3e-5] |
| Learning-rate warm-up epochs | `optimizer.warmup_epochs` | {1, 2, 3} |
| Gradient-clipping norm | `optimizer.grad_clip_norm` | {0.5, 1.0, 2.0, 3.0} |
| Minimum learning-rate ratio | `optimizer.min_lr_ratio` | log-uniform [0.03, 0.3] |
| λ_view | `lambda_xray_view` | log-uniform [0.05, 0.25] |
| λ_cent (final value) | `lambda_centroid_alignment` | log-uniform [1.5, 6.0] |
| λ_MIL | `lambda_projection_mil_contrastive` | log-uniform [0.025, 0.1] |
| λ_div | `lambda_projection_attention_diversity` | log-uniform [0.025, 0.1] |

Only the hyperparameters a method uses are searched: RADIANT-X has no MRI or cross-modal weights.

---

## Downstream evaluation

`downstream/` evaluates a frozen pretraining checkpoint on DT1–DT7: linear heads on the pooled
representation (classification, regression) and a multi-scale decoder (segmentation), with
patient-level 5-fold CV, a nested hyperparameter search on validation partitions, and a paired
patient bootstrap between encoders. See [downstream/README.md](downstream/README.md).

```bash
python -m downstream.evaluate --task downstream/configs/dt1_bone_tumor_malignancy.yaml \
    --checkpoint outputs/radiant/resnet50_scratch/checkpoints/final.pt \
    --out-dir outputs/downstream/radiant_resnet50_scratch/dt1
```

---

## Utilities

```bash
# Embedding-space diagnostic: modality mixing + anatomy preservation of the embeddings
python scripts/evaluate_embedding_space.py \
    --config configs/radiant/resnet50_scratch.yaml \
    --checkpoint outputs/radiant/resnet50_scratch/checkpoints/final.pt \
    --output-json outputs/radiant/resnet50_scratch/embedding_space_metrics.json

# Diagnose a pretrained MRI-encoder checkpoint (depth-token diversity)
python pretrain_mri.py --config configs/mri_pretraining/mri_encoder_genesis.yaml \
    --diagnose outputs/mri_pretraining/checkpoint_latest.pt

# Unit tests
pytest tests
```

---

## Citation

The paper is under review. Until it has a venue or an arXiv identifier, please cite it as:

```bibtex
@misc{curtovilalta2027radiant,
  title  = {{RADIANT}: Unpaired {3D} {MRI} Supervision for X-Ray Representation Learning},
  author = {Curto-Vilalta, Anna and Gillani, Amna and von Eisenhart-Rothe, R{\"u}diger and
            Rueckert, Daniel and Hinterwimmer, Florian},
  year   = {2027},
  note   = {Under review}
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The datasets and the pretrained
third-party weights (ImageNet, DINOv3) keep their own licenses.
