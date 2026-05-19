# SG-CBM

Spatially Grounded Concept Bottleneck Models for CUB and ImageNet.

This repository contains the release code for training concept layers, fitting
sparse GLM heads, and evaluating accuracy, concept prediction, and localization.

## Install

```bash
pip install -r requirements.txt
```

## Scripts

```text
train_cbm.py                         Train SG-CBM/SALF/VLG/LF concept layers.
scripts/precompute_imagenet_targets.py Precompute ImageNet GDINO targets.
scripts/train_sparse_nec.py          Fit sparse GLM heads for NEC sweeps.
scripts/eval_nec.py                  Evaluate classification accuracy at NEC levels.
scripts/eval_concept_accuracy.py     Evaluate concept prediction against GDINO/CUB labels.
scripts/eval_gdino_localization.py   Evaluate GDINO-box localization for CUB/ImageNet.
evaluations/cub_part_localization.py Evaluate CUB part-point localization.
```

## Data

CUB should be in ImageFolder form:

```text
datasets/CUB/train/<class_name>/*.jpg
datasets/CUB/test/<class_name>/*.jpg
```

Create that split from CUB-200-2011:

```bash
mkdir -p datasets && curl -L "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" -o datasets/CUB_200_2011.tgz && tar -xzf datasets/CUB_200_2011.tgz -C datasets && python datasets/split_cub_dataset.py --cub_root datasets/CUB_200_2011 --output_root datasets/CUB
```

ImageNet training expects either an ImageFolder train root or a JSONL manifest
with `path`, `class_id`, and `sample_index`. ImageNet validation can be an
extracted directory or the official validation tar where supported by the eval
script.

GDINO annotations are expected as JSON files:

```text
annotations/cub_train/0.json
annotations/cub_val/0.json
annotations/imagenet_val/0.json
```

For ImageNet validation, use a filename-to-annotation mapping JSON when the
annotation filenames are not in `ILSVRC2012_val_00000001.JPEG -> 0.json` order.
The concept accuracy and localization scripts expose `--annotation_mapping_json`
for this case.

## Train Concept Layers

The main entry point is `train_cbm.py`.

```bash
python train_cbm.py --config configs/cub_gcbm.json
python train_cbm.py --config configs/cub_salf.json
python train_cbm.py --config configs/cub_gcbm.json --model_name vlg_cbm
python train_cbm.py --config configs/cub_gcbm.json --model_name lf_cbm
python train_cbm.py --config configs/imagenet_gcbm.yaml
python train_cbm.py --config configs/imagenet_vlg.yaml
```

For ImageNet SG-CBM, set these paths in `configs/imagenet_gcbm.yaml`:

```text
train_root: /path/to/imagenet/train
annotation_dir: /path/to/imagenet_annotations
precomputed_target_dir: /path/to/precomputed_targets
```

The ImageNet SG-CBM config uses the verified ResNet-50 `IMAGENET1K_V1` setup:
deterministic resize/center-crop, 14x14 masks, `conv4+conv5`, soft-box targets,
and soft-align KL loss. `configs/imagenet_vlg.yaml` uses the same backbone with
`branch_arch=global_only`, so it skips spatial branch compute.

Precompute ImageNet GDINO targets:

```bash
python scripts/precompute_imagenet_targets.py --image_root /path/to/imagenet/train --annotation_dir /path/to/imagenet_annotations --concept_file concept_files/imagenet_filtered.txt --output_dir /path/to/precomputed_targets --split train
```

## Train Sparse NEC Heads

```bash
python scripts/train_sparse_nec.py --dataset cub --load_path /path/to/cub_run
python scripts/train_sparse_nec.py --dataset imagenet --artifact_dir /path/to/imagenet_run --output_dir /path/to/sparse_sweep --lam_max 0.0007 --table_device cuda
```

For ImageNet, `W_g@NEC=<k>.pt` is saved after VLG-CBM-style global threshold
truncation, so `NEC=5` keeps roughly `5 * 1000` nonzero final-layer weights.

## Evaluate NEC Accuracy

```bash
python scripts/eval_nec.py --dataset cub --load_path /path/to/cub_run --output_json results/cub_nec.json
python scripts/eval_nec.py --dataset imagenet --artifact_dir /path/to/imagenet_run_or_sparse_sweep --val_root /path/to/imagenet_val --devkit_dir /path/to/ILSVRC2012_devkit_t12 --output_json results/imagenet_nec.json
```

## Evaluate Concept Accuracy

Concept accuracy compares concept scores to binary concept-presence labels and
reports AUROC, AP, Macro AP, P@5, threshold metrics, and best-F1.

```bash
python scripts/eval_concept_accuracy.py --dataset cub --gt_source gdino --load_paths /path/to/cub_sgcbm_run /path/to/cub_vlg_run --model_names savlg_cbm vlg_cbm --names SG-CBM VLG-CBM --annotation_dir /path/to/cub_gdino_annotations --normalization sigmoid --output results/cub_concept_accuracy.json
python scripts/eval_concept_accuracy.py --dataset imagenet --gt_source gdino --load_paths /path/to/imagenet_sgcbm_run --annotation_dir /path/to/imagenet_gdino_annotations --annotation_mapping_json /path/to/imagenet_val_filename_to_annotation.json --val_root /path/to/imagenet_val --normalization sigmoid --output results/imagenet_concept_accuracy.json
python scripts/eval_concept_accuracy.py --dataset imagenet --gt_source gdino --load_paths /path/to/salf_imagenet_checkpoint --model_names salf_cbm --names SALF-CBM --annotation_dir /path/to/imagenet_gdino_annotations --annotation_mapping_json /path/to/imagenet_val_filename_to_annotation.json --val_root /path/to/imagenet_val --normalization concept_zscore_minmax --output results/imagenet_salf_concept_accuracy.json
```

ImageNet concept accuracy supports SG-CBM/VLG-style release checkpoints and
SALF checkpoints with `W_c.pt`, `proj_mean.pt`, and `proj_std.pt`.

## Evaluate GDINO Localization

Localization uses the concept-layer checkpoint, not sparse GLM weights.

```bash
python scripts/eval_gdino_localization.py --dataset cub --gcbm_path /path/to/cub_sgcbm_run --annotation_dir /path/to/cub_gdino_annotations --output results/cub_gdino_localization.json --map_normalization concept_zscore_minmax
python scripts/eval_gdino_localization.py --dataset imagenet --gcbm_path /path/to/imagenet_sgcbm_run --annotation_dir /path/to/imagenet_gdino_annotations --annotation_mapping_json /path/to/imagenet_val_filename_to_annotation.json --val_root /path/to/imagenet_val --output results/imagenet_gdino_localization.json --map_normalization concept_zscore_minmax
```

The output includes RMA-style `mass_in_gt`, pointing accuracy, mean IoU, and
LocAcc at requested IoU thresholds.

## Evaluate CUB Part Localization

```bash
python evaluations/cub_part_localization.py --load_path /path/to/cub_sgcbm_run --annotation_dir annotations --cub_root /path/to/CUB_200_2011 --mapping_json data/cub_concept_part_mapping_gpt54.json --output results/cub_part_localization.json --map_normalization concept_zscore_minmax
python evaluations/cub_part_localization.py --load_path /path/to/cub_sgcbm_run --annotation_dir annotations --cub_root /path/to/CUB_200_2011 --mapping_json data/cub_concept_part_mapping_gpt54.json --output results/cub_part_localization_oracle.json --map_normalization concept_zscore_minmax --compute_concept_oracle
```

`--compute_concept_oracle` evaluates all concept maps for each part target and
selects the best concept per metric.

## Notes

- Public model name: SG-CBM.
- Some internal names still use `savlg` or `gcbm` for checkpoint compatibility.
- SG-CBM training/localization uses GDINO annotations.
- NEC and classification evaluation use sparse GLM heads; localization does not.
