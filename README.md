# SG-CBM

Code for training and evaluating Spatially Grounded Concept Bottleneck Models
(SG-CBM) on CUB and ImageNet.

## Installation

```bash
pip install -r requirements.txt
```

## Data

### CUB

CUB training expects an ImageFolder-style split:

```text
datasets/CUB/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

Download and split CUB-200-2011:

```bash
mkdir -p datasets
curl -L "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" \
  -o datasets/CUB_200_2011.tgz
tar -xzf datasets/CUB_200_2011.tgz -C datasets
python datasets/split_cub_dataset.py \
  --cub_root datasets/CUB_200_2011 \
  --output_root datasets/CUB
export CUB_DATASET_ROOT="$PWD/datasets/CUB"
```

CUB SG-CBM/SALF training and GDINO localization use annotation JSON files:

```text
annotations/
  cub_train/0.json
  cub_val/0.json
  ...
```

### ImageNet

ImageNet SG-CBM training supports either an ImageFolder train root or a JSONL
manifest with `path`, `class_id`, and `sample_index`. Training also expects
GDINO annotations and precomputed GDINO target tensors for concept-layer
supervision.

For ImageNet validation, `eval_imagenet_nec.py` supports a flat validation
directory when the official devkit metadata is supplied. Localization supports
either an extracted validation directory or the official validation tar.

## Unified CLI

The unified entry point supports CUB training for SG-CBM, SALF-CBM, VLG-CBM,
and LF-CBM, plus ImageNet SG-CBM training.

```bash
python scripts/cbm.py train --dataset cub --model sgcbm --config configs/cub_gcbm.json
python scripts/cbm.py train --dataset cub --model salf --config configs/cub_salf.json
python scripts/cbm.py train --dataset cub --model vlg --config configs/cub_gcbm.json
python scripts/cbm.py train --dataset cub --model lf --config configs/cub_gcbm.json

python scripts/cbm.py train \
  --dataset imagenet \
  --model sgcbm \
  --config configs/imagenet_gcbm.yaml
```

Run sparse GLM / NEC evaluation for a trained CUB checkpoint:

```bash
python scripts/cbm.py test --load_path /path/to/cub_run --lam 0.1
```

## ImageNet

Train the SG-CBM concept layer:

```bash
python scripts/train_imagenet_gcbm.py \
  --train_root /path/to/imagenet/train \
  --train_manifest /path/to/train_manifest.jsonl \
  --annotation_dir /path/to/imagenet_annotations \
  --precomputed_target_dir /path/to/precomputed_targets \
  --concept_file concept_files/imagenet_filtered.txt \
  --save_dir artifacts/imagenet \
  --resnet50_weights v1 \
  --mask_h 14 \
  --mask_w 14
```

Train sparse GLM heads and evaluate NEC accuracy:

```bash
python scripts/run_glm_path.py --artifact_dir /path/to/gcbm_run

python scripts/eval_imagenet_nec.py \
  --artifact_dir /path/to/gcbm_run \
  --val_root /path/to/imagenet_val \
  --devkit_dir /path/to/ILSVRC2012_devkit_t12 \
  --nec_values 1,5,10,20,50,4309
```

Evaluate GDINO-box localization:

```bash
python scripts/eval_gdino_localization.py \
  --dataset imagenet \
  --gcbm_path /path/to/gcbm_run \
  --annotation_dir /path/to/imagenet_annotations \
  --val_root /path/to/imagenet_val \
  --output results/imagenet_gdino_localization.json \
  --map_normalization concept_zscore_minmax \
  --activation_thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

The ImageNet localization evaluator maps validation annotations by filename.
Use the extracted `val_root` when available; otherwise pass `--val_tar`. The
script reports distribution metrics such as mass inside GT and pointing
accuracy, plus thresholded mask/box IoU and LocAcc at the requested box IoU
thresholds.

For ImageNet-v1 checkpoint reproduction, keep the saved checkpoint
configuration, including `resnet50_weights=v1`.

## CUB

Dedicated training and NEC wrappers:

```bash
python scripts/train_cub_gcbm.py --config configs/cub_gcbm.json
python scripts/train_cub_salf.py --config configs/cub_salf.json
python scripts/eval_cub_nec.py --load_path /path/to/cub_run
```

Evaluate GDINO-box localization:

```bash
python scripts/eval_gdino_localization.py \
  --dataset cub \
  --gcbm_path /path/to/gcbm_run \
  --annotation_dir annotations \
  --output results/cub_gdino_localization.json \
  --map_normalization concept_zscore_minmax \
  --activation_thresholds 0.3,0.5,0.7,0.9
```

For CUB GDINO localization, `--annotation_dir` should contain the CUB GDINO JSON
files used during SG-CBM concept-layer training/evaluation. The evaluator uses
the trained concept-layer checkpoint and saved concept statistics from
`--gcbm_path`; sparse GLM weights are not used for native localization metrics.
The `--gcbm_path` flag is kept as the checkpoint-directory argument name.

Evaluate CUB localization across model variants:

```bash
python scripts/eval_cub_localization.py \
  --gcbm_path /path/to/gcbm_run \
  --salf_path /path/to/salf_run \
  --vlg_path /path/to/vlg_run \
  --lf_path /path/to/lf_run \
  --cub70_root /path/to/CUB70-PartSegmentationDataset \
  --cub_root /path/to/CUB_200_2011 \
  --mapping_json /path/to/cub_concept_part_mapping.json
```

Evaluate CUB part-point localization:

```bash
python scripts/precompute_cub_part_annotation_cache.py \
  --load_path /path/to/gcbm_run \
  --annotation_dir annotations \
  --cub_root /path/to/CUB_200_2011 \
  --mapping_json /path/to/cub_concept_part_mapping.json \
  --output artifacts/cub_part_annotation_cache.json

python scripts/eval_cub_part_localization.py \
  --load_path /path/to/gcbm_run \
  --annotation_dir annotations \
  --annotation_cache_json artifacts/cub_part_annotation_cache.json \
  --cub_root /path/to/CUB_200_2011 \
  --mapping_json /path/to/cub_concept_part_mapping.json \
  --output results/cub_part_localization.json
```

Evaluate concept accuracy on a common concept set:

```bash
python scripts/eval_concept_accuracy.py \
  --load_paths /path/to/gcbm_run /path/to/salf_run /path/to/vlg_run /path/to/lf_run
```

## Localization Details

`scripts/eval_gdino_localization.py` is the shared GDINO pseudo-GT localization
entry point for CUB and ImageNet. Dataset loading differs, but the metric code is
shared once images, annotations, and native spatial maps are built.

Required inputs:

- `--dataset cub|imagenet`
- `--gcbm_path`: SG-CBM concept-layer run directory.
- `--annotation_dir`: GDINO annotation JSON directory.
- `--output`: destination JSON file.
- ImageNet only: `--val_root` for extracted validation images or `--val_tar` for
  the official validation tar.

Recommended normalization:

```bash
--map_normalization concept_zscore_minmax
```

This is the recommended evaluation normalization. On CUB it uses saved
`proj_mean.pt` and `proj_std.pt` when present; on ImageNet it applies per-map
z-score followed by min-max scaling.

Common full-split commands:

```bash
python scripts/eval_gdino_localization.py \
  --dataset cub \
  --gcbm_path /path/to/cub_sgcbm_run \
  --annotation_dir /path/to/cub_gdino_annotations \
  --output results/cub_gdino_localization.json \
  --batch_size 128 \
  --map_normalization concept_zscore_minmax \
  --activation_thresholds 0.3,0.5,0.7,0.9 \
  --box_iou_thresholds 0.1,0.3,0.5

python scripts/eval_gdino_localization.py \
  --dataset imagenet \
  --gcbm_path /path/to/imagenet_sgcbm_run \
  --annotation_dir /path/to/imagenet_gdino_annotations \
  --val_root /path/to/imagenet_val \
  --output results/imagenet_gdino_localization.json \
  --batch_size 64 \
  --num_workers 8 \
  --map_normalization concept_zscore_minmax \
  --activation_thresholds 0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,mean \
  --box_iou_thresholds 0.1,0.3,0.5
```

The output JSON includes `distribution_metrics` and `threshold_metrics`.
`mass_in_gt` is the RMA-style score, `point_hit` is pointing accuracy, and
`threshold_metrics[*].box_acc` contains LocAcc at each requested box IoU
threshold.

## Notes

- ImageNet training in this repository is SG-CBM concept-layer training.
- Sparse GLM / NEC evaluation uses the trained concept-layer checkpoint and GLM
  sweep outputs.
- Localization evaluation uses concept-layer checkpoints, not sparse GLM heads.
- Some file names and flags still contain `gcbm` because they refer to existing
  checkpoint directory conventions.
