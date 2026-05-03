# G-CBM Release

Minimal training/evaluation release tree for reproducing the final G-CBM paper runs on ImageNet and CUB.

## Install

```bash
pip install -r requirements.txt
```

## Reproduce

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

ImageNet localization:

```bash
python scripts/eval_imagenet_localization.py \
  --artifact_dir /path/to/run_dir \
  --val_tar /path/to/ILSVRC2012_img_val.tar \
  --devkit_dir /path/to/ILSVRC2012_devkit_t12 \
  --annotation_dir /path/to/imagenet_annotations
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

CUB localization across any subset of SAVLG, SALF, VLG, LF:

```bash
python scripts/eval_cub_localization.py \
  --savlg_path /path/to/savlg_run \
  --salf_path /path/to/salf_run \
  --vlg_path /path/to/vlg_run \
  --lf_path /path/to/lf_run \
  --cub70_root /path/to/CUB70-PartSegmentationDataset \
  --cub_root /path/to/CUB_200_2011 \
  --mapping_json /path/to/cub_concept_part_mapping.json
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
