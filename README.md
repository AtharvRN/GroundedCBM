# G-CBM Release

Minimal training/evaluation release tree for reproducing the final G-CBM paper runs on ImageNet and CUB.

## Install

```bash
pip install -r requirements.txt
```

## Data Setup

CUB training expects an ImageFolder-style split:

```text
datasets/CUB/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

Download the official CUB-200-2011 archive from CaltechDATA:

```bash
mkdir -p datasets
curl -L "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" \
  -o datasets/CUB_200_2011.tgz
tar -xzf datasets/CUB_200_2011.tgz -C datasets
python datasets/split_cub_dataset.py \
  --cub_root datasets/CUB_200_2011 \
  --output_root datasets/CUB
```

Then point the code at the split:

```bash
export CUB_DATASET_ROOT="$PWD/datasets/CUB"
```

CUB G-CBM/SALF training also needs concept annotation JSONs. Place or unpack the
released annotation files with this layout:

```text
annotations/
  cub_train/0.json
  cub_train/1.json
  ...
  cub_val/0.json
  cub_val/1.json
  ...
```

Then update `annotation_dir` in `configs/cub_gcbm.json` and
`configs/cub_salf.json`, or override it on the command line:

```bash
python scripts/cbm.py train \
  --dataset cub \
  --model gcbm \
  --config configs/cub_gcbm.json \
  --annotation_dir annotations
```

For CUB localization, download or place the CUB-70 part segmentation dataset
and pass it via `--cub70_root`.

## Reproduce

Basic unified train/test entrypoint:

```bash
python scripts/cbm.py train --dataset cub --model gcbm --config configs/cub_gcbm.json
python scripts/cbm.py train --dataset cub --model salf --config configs/cub_salf.json
python scripts/cbm.py test --load_path /path/to/cub_run --lam 0.1
```

ImageNet G-CBM can also be launched through the same entrypoint:

```bash
python scripts/cbm.py train --dataset imagenet --model gcbm --config configs/imagenet_gcbm.yaml
```

The commands below are the lower-level task-specific wrappers preserved for
paper reproduction.

ImageNet concept-layer training:

```bash
python scripts/train_imagenet_gcbm.py \
  --train_root /path/to/imagenet/train \
  --annotation_dir /path/to/imagenet_annotations \
  --precomputed_target_dir /path/to/precomputed_targets \
  --concept_file concept_files/imagenet_filtered.txt \
  --save_dir artifacts/imagenet
```

ImageNet GLM path / NEC:

```bash
python scripts/run_glm_path.py --artifact_dir /path/to/run_dir
python scripts/eval_imagenet_nec.py --artifact_dir /path/to/run_dir --val_root /path/to/imagenet_val
```

GDINO-box localization has a shared entry point for CUB and ImageNet. CUB uses
the paper-style native-map evaluator with `gt_present` concepts; ImageNet uses
the validation GDINO annotations with filename-based annotation mapping.

ImageNet localization:

```bash
python scripts/eval_gdino_localization.py \
  --dataset imagenet \
  --gcbm_path /path/to/run_dir \
  --val_tar /path/to/ILSVRC2012_img_val.tar \
  --devkit_dir /path/to/ILSVRC2012_devkit_t12 \
  --annotation_dir /path/to/imagenet_annotations \
  --output results/imagenet_gdino_localization.json
```

CUB SAVLG / G-CBM training:

```bash
python scripts/train_cub_savlg.py --config configs/cub_gcbm.json
```

CUB SALF-CBM training:

```bash
python scripts/train_cub_salf.py --config configs/cub_salf.json
```

CUB NEC / sparse evaluation:

```bash
python scripts/eval_cub_nec.py --load_path /path/to/cub_run
```

CUB G-CBM localization against GDINO pseudo boxes:

```bash
python scripts/eval_gdino_localization.py \
  --dataset cub \
  --gcbm_path /path/to/gcbm_run \
  --annotation_dir annotations \
  --output results/cub_gdino_localization.json \
  --activation_thresholds 0.3,0.5,0.7,0.9
```

CUB70 localization across any subset of SAVLG, SALF, VLG, LF:

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

G-CBM CUB part-point localization using the official CUB part annotations:

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

Concept accuracy evaluation on the common concept set:

```bash
python scripts/eval_concept_accuracy.py \
  --load_paths /path/to/savlg_run /path/to/salf_run /path/to/vlg_run /path/to/lf_run
```

## Smoke tests

```bash
python -m py_compile $(find . -name '*.py' -not -path './.git/*')
python -m unittest discover -s tests -v
```
