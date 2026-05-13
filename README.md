# SG-CBM

Code for training and evaluating Spatially Grounded Concept Bottleneck Models
(SG-CBM) on CUB and ImageNet.

## Installation

```bash
pip install -r requirements.txt
```

## Public Scripts

The release has five user-facing entry points:

```text
train_cbm.py                         Train CBM concept layers.
scripts/train_sparse_nec.py          Train sparse GLM heads with an NEC sweep.
scripts/eval_nec.py                  Evaluate/report CBM+sparse accuracy at NEC values.
scripts/eval_gdino_localization.py   Evaluate localization against GDINO pseudo-GT boxes.
scripts/eval_cub_part_localization.py Evaluate localization against CUB part points.
```

Other Python files under `gcbm/`, `methods/`, `model/`, `data/`, and
`glm_saga/` are implementation modules used by these entry points.

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

CUB SG-CBM/SALF training and localization use annotation JSON files:

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

ImageNet NEC evaluation supports an extracted validation directory or the
official validation tar. Supply the devkit metadata when evaluating a flat
validation directory.

## 1. Train CBM

`train_cbm.py` is the unified training entry point. It supports CUB training for
SG-CBM, SALF-CBM, VLG-CBM, and LF-CBM, plus ImageNet SG-CBM training.

```bash
python train_cbm.py --config configs/cub_gcbm.json
python train_cbm.py --config configs/cub_salf.json
python train_cbm.py --config configs/cub_gcbm.json --model_name vlg_cbm
python train_cbm.py --config configs/cub_gcbm.json --model_name lf_cbm

python train_cbm.py --config configs/imagenet_gcbm.yaml
```

For ImageNet-v1 checkpoint reproduction, keep the saved checkpoint
configuration, including `resnet50_weights=v1`.

## 2. Train Sparse NEC Heads

CUB:

```bash
python scripts/train_sparse_nec.py \
  --dataset cub \
  --load_path /path/to/cub_run \
  --lam 0.1
```

ImageNet:

```bash
python scripts/train_sparse_nec.py \
  --dataset imagenet \
  --artifact_dir /path/to/imagenet_run \
  --nec_values 1,5,10,20,50,4309
```

## 3. Evaluate NEC Accuracy

CUB reads the `metrics.csv` written by the sparse sweep:

```bash
python scripts/eval_nec.py \
  --dataset cub \
  --load_path /path/to/cub_run \
  --nec_values 5,10,20,50 \
  --output_json results/cub_nec.json
```

ImageNet evaluates the requested NEC values on validation images:

```bash
python scripts/eval_nec.py \
  --dataset imagenet \
  --artifact_dir /path/to/imagenet_run_or_sparse_sweep \
  --val_root /path/to/imagenet_val \
  --devkit_dir /path/to/ILSVRC2012_devkit_t12 \
  --nec_values 1,5,10,20,50,4309 \
  --output_json results/imagenet_nec.json
```

## 4. Evaluate GDINO Localization

This script is shared by CUB and ImageNet. Dataset loading differs, but the
metric code is shared once images, annotations, and native spatial maps are
built. Localization uses the concept-layer checkpoint, not sparse GLM heads.

CUB:

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
```

ImageNet:

```bash
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

Recommended normalization:

```bash
--map_normalization concept_zscore_minmax
```

On CUB this uses saved `proj_mean.pt` and `proj_std.pt`; on ImageNet it applies
per-map z-score followed by min-max scaling.

## 5. Evaluate CUB Part Localization

This evaluates CUB part-point localization using official CUB part annotations
and a concept-to-part mapping.

```bash
python scripts/eval_cub_part_localization.py \
  --load_path /path/to/cub_sgcbm_run \
  --annotation_dir annotations \
  --cub_root /path/to/CUB_200_2011 \
  --mapping_json /path/to/cub_concept_part_mapping.json \
  --output results/cub_part_localization.json \
  --map_normalization concept_zscore_minmax
```

To report the concept oracle, where all concept maps are evaluated for each
part target and the best concept is selected per metric:

```bash
python scripts/eval_cub_part_localization.py \
  --load_path /path/to/cub_sgcbm_run \
  --annotation_dir annotations \
  --cub_root /path/to/CUB_200_2011 \
  --mapping_json /path/to/cub_concept_part_mapping.json \
  --output results/cub_part_localization_oracle.json \
  --map_normalization concept_zscore_minmax \
  --compute_concept_oracle
```

## Notes

- Public model name: SG-CBM.
- Sparse GLM / NEC evaluation uses trained concept-layer checkpoints and sparse
  head outputs.
- Localization evaluation uses concept-layer checkpoints, not sparse GLM heads.
- Some internal class and function names still contain `savlg` or `gcbm` for
  checkpoint compatibility.
