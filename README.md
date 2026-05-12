# G-CBM Release

Minimal code for training and evaluating the paper CBM experiments. The public
model name is **G-CBM**; some internal code still uses `savlg_cbm` for backward
compatibility with trained checkpoints.

## Install

```bash
pip install -r requirements.txt
```

## What Is Included

- CUB training for `vlg_cbm`, `lf_cbm`, `salf_cbm`, and `savlg_cbm`/G-CBM.
- ImageNet G-CBM concept-layer training with precomputed GDINO targets.
- Sparse GLM / NEC sweeps and final accuracy evaluation.
- GDINO-box localization for CUB and ImageNet G-CBM.
- CUB part-localization utilities.
- Focused unit tests for bbox transforms, target precompute, GLM/NEC helpers,
  CLI dispatch, and localization metrics.

ImageNet baseline training for VLG/LF/SALF and Stanford Cars are not part of
this cleaned release tree.

## Data

CUB expects an ImageFolder-style split:

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

CUB G-CBM/SALF training and GDINO localization need annotation JSONs:

```text
annotations/
  cub_train/0.json
  cub_val/0.json
  ...
```

ImageNet G-CBM training expects:

- ImageFolder train root or a JSONL manifest with `path`, `class_id`,
  `sample_index`.
- GDINO annotation directory.
- Precomputed GDINO target store matching the training set, or a larger target
  store addressed by manifest `sample_index`.

## Unified CLI

Train CUB models:

```bash
python scripts/cbm.py train --dataset cub --model gcbm --config configs/cub_gcbm.json
python scripts/cbm.py train --dataset cub --model salf --config configs/cub_salf.json
python scripts/cbm.py train --dataset cub --model vlg --config configs/cub_gcbm.json
python scripts/cbm.py train --dataset cub --model lf --config configs/cub_gcbm.json
```

Run sparse GLM / NEC accuracy evaluation for CUB checkpoints:

```bash
python scripts/cbm.py test --load_path /path/to/cub_run --lam 0.1
```

Train ImageNet G-CBM:

```bash
python scripts/cbm.py train \
  --dataset imagenet \
  --model gcbm \
  --config configs/imagenet_gcbm.yaml
```

## ImageNet Commands

Concept-layer training:

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

Sparse GLM path and NEC accuracy:

```bash
python scripts/run_glm_path.py --artifact_dir /path/to/gcbm_run

python scripts/eval_imagenet_nec.py \
  --artifact_dir /path/to/gcbm_run \
  --val_root /path/to/imagenet_val \
  --devkit_dir /path/to/ILSVRC2012_devkit_t12 \
  --nec_values 1,5,10,20,50,4309
```

GDINO localization:

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

For official paper-number reproduction, use the checkpoint configuration saved
with the run. The ImageNet-v1 G-CBM checkpoint uses `resnet50_weights=v1`.

## CUB Commands

Dedicated wrappers:

```bash
python scripts/train_cub_gcbm.py --config configs/cub_gcbm.json
python scripts/train_cub_salf.py --config configs/cub_salf.json
python scripts/train_cub_savlg.py --config configs/cub_gcbm.json
python scripts/eval_cub_nec.py --load_path /path/to/cub_run
```

GDINO localization for G-CBM:

```bash
python scripts/eval_gdino_localization.py \
  --dataset cub \
  --gcbm_path /path/to/gcbm_run \
  --annotation_dir annotations \
  --output results/cub_gdino_localization.json \
  --map_normalization concept_zscore_minmax \
  --activation_thresholds 0.3,0.5,0.7,0.9
```

CUB localization across SAVLG/SALF/VLG/LF checkpoints:

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

CUB part-point localization:

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

Concept accuracy on a common concept set:

```bash
python scripts/eval_concept_accuracy.py \
  --load_paths /path/to/savlg_run /path/to/salf_run /path/to/vlg_run /path/to/lf_run
```

## Smoke Tests

Static checks:

```bash
python -m py_compile $(find . -name '*.py' -not -path './.git/*')
python -m unittest discover -s tests -v
```

ImageNet training smoke tested on a GTX 1080 pod:

```bash
python scripts/train_imagenet_gcbm.py \
  --train_root /workspace/imagenet_100k_balanced/train \
  --train_manifest /workspace/imagenet_100k_balanced_index/train_present_timing_manifest.jsonl \
  --annotation_dir /workspace/imagenet_annotations \
  --precomputed_target_dir /workspace/imagenet_100k_balanced_precomputed \
  --save_dir artifacts/imagenet_train_smoke \
  --run_name gcbm_imagenet_smoke \
  --epochs 1 \
  --max_train_images 8 \
  --max_val_images 0 \
  --eval_every 0 \
  --batch_size 1 \
  --workers 0 \
  --device cuda \
  --amp fp16 \
  --resnet50_weights v1 \
  --mask_h 14 \
  --mask_w 14
```
