# Downstream evaluation (DT1–DT7)

Frozen-encoder evaluation of a pretrained X-ray encoder on the seven downstream tasks of the paper.
Only the X-ray backbone of a pretraining checkpoint is used; the MRI branch, projectors, projection
bridge, prototype banks and auxiliary heads are discarded.

| Task | Config | Dataset | Type | Reported metric |
|---|---|---|---|---|
| **DT1** bone-tumor malignancy | `configs/dt1_bone_tumor_malignancy.yaml` | internal | 2-class | balanced accuracy |
| **DT2** binary fracture classification | `configs/dt2_fracture_binary.yaml` | FracAtlas | 2-class | balanced accuracy |
| **DT3** three-class fracture classification | `configs/dt3_fracture_three_class.yaml` | FracAtlas | 3-class | balanced accuracy |
| **DT4** Kellgren–Lawrence grading | `configs/dt4_kl_grading.yaml` | OAI | 5-class, class-weighted | balanced accuracy |
| **DT5** bone age | `configs/dt5_bone_age.yaml` | RSNA | regression | MAE (months) |
| **DT6** bone-tumor segmentation | `configs/dt6_bone_tumor_segmentation.yaml` | internal | binary mask | Dice |
| **DT7** fracture segmentation | `configs/dt7_fracture_segmentation.yaml` | FracAtlas | binary mask | Dice |

## Protocol

- **Preprocessing:** grayscale, resized to 224×224, normalized per image.
- **Encoder:** frozen throughout (eval mode, no gradients).
- **Classification / regression:** a linear head on the globally pooled backbone representation, with
  cross-entropy (class-weighted for DT4) or mean squared error (on the standardized target). The
  pooled features are extracted once per dataset.
- **Segmentation:** a multi-scale decoder on the frozen feature maps.
  - **Fused features:** ResNet `layer3`/`layer2` skip connections, or ViT block-4 auxiliary features.
  - **Training:** soft Dice loss and mask-synchronized augmentation of the training images
    (horizontal flip, ±10° rotation, brightness/contrast).
  - **Metric:** Dice score of the prediction thresholded at 0.5, computed per image and averaged
    over the images of a fold (an empty prediction on an empty mask scores 1).
- **Cross-validation:** patient-level 5-fold. Folds are stratified by class for classification.
- **Validation partition:** inside each training fold, 15% of the patients are held out.
  - It is used for early stopping.
  - It is used for a random search over learning rate, batch size and weight decay (the search
    space is in each config).
  - The trial with the best validation score is evaluated once on the test fold.
  - Trial candidates are seeded, so every encoder gets the same candidates.
- **Reporting:** the mean and SD over folds (the values in the paper), plus the metric on the pooled
  out-of-fold predictions.
- **Significance (`significance.py`):** a paired bootstrap over patients (10,000 resamples) between
  two encoders evaluated on identical folds. Δ is the difference in fold-mean performance; each
  resample draws patients with replacement within every test fold, the same draw for both models.
  It reports Δ, the 95% percentile interval and the two-sided p = 2(b + 1)/(B + 1).
  Holm-Bonferroni correction is applied afterwards, across the tasks of each comparison family.

## Usage

All commands are run from the repository root.

```bash
# 1. Build the task manifests (CSV: image_path, patient_id, label | mask_path). See each config's
#    header for the exact command, and `python -m downstream.prepare_manifests -h`.
python -m downstream.prepare_manifests image-folder --root <FracAtlas/images_processed> \
    --out data/downstream/dt2_fracture_binary.csv

# 2. Evaluate a checkpoint on a task.
python -m downstream.evaluate --task downstream/configs/dt2_fracture_binary.yaml \
    --checkpoint outputs/radiant/resnet50_scratch/checkpoints/final.pt \
    --out-dir outputs/downstream/radiant_resnet50_scratch/dt2

# 3. Compare two encoders on the same task.
python -m downstream.significance --task downstream/configs/dt2_fracture_binary.yaml \
    --a outputs/downstream/radiant_resnet50_scratch/dt2 \
    --b outputs/downstream/radiant_x_resnet50_scratch/dt2 \
    --out outputs/downstream/significance/radiant_vs_radiant_x/dt2.json

# 4. Holm-Bonferroni across the tasks of one comparison family (adds p_holm to each file).
python -m downstream.significance --holm outputs/downstream/significance/radiant_vs_radiant_x/dt*.json
```

**Patient ids.**
- DT4 (OAI): patients come from the knee-crop file names (`<patientID>_<left|right>`).
- DT5 (RSNA): one image per patient.
- FracAtlas (DT2, DT3, DT7) and the internal datasets (DT1, DT6) have no patient ids: each image is
  treated as its own patient. (`--patient-map`, a `file_name, patient_id` CSV, can supply ids when
  they exist.)

## Files

| File | Content |
|---|---|
| `prepare_manifests.py` | builds the CSV manifests from each dataset's on-disk layout |
| `data.py` | manifest dataset, preprocessing, segmentation augmentation |
| `encoder.py` | loads the frozen X-ray backbone from a RADIANT checkpoint |
| `models.py` | multi-scale segmentation decoder |
| `splits.py` | patient-level (stratified) K-fold and validation hold-out |
| `engine.py` | training/prediction of the heads and the decoder, metrics |
| `evaluate.py` | nested CV + hyperparameter search for one task and one checkpoint |
| `significance.py` | paired patient bootstrap between two encoders |
