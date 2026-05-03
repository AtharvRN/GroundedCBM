# G-CBM Release

Minimal release tree for reproducing the final G-CBM paper runs on ImageNet and CUB.

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

CUB training:

```bash
python scripts/train_cub_gcbm.py --config configs/cub_gcbm.json
```

CUB NEC / sparse evaluation:

```bash
python scripts/eval_cub_nec.py --load_path /path/to/cub_run
```

CUB localization:

```bash
python scripts/eval_cub_localization.py \
  --load_path /path/to/cub_run \
  --annotation_dir /path/to/cub_annotations \
  --cub_root /path/to/CUB_200_2011 \
  --mapping_json /path/to/cub_concept_part_mapping.json \
  --output /path/to/cub_localization.json
```

ImageNet figure rendering:

```bash
python scripts/render_imagenet_spatial_grid.py --output_dir /tmp/gcbm_spatial_grid
```

## Smoke tests

```bash
python -m py_compile $(find . -name '*.py' -not -path './.git/*')
python -m unittest discover -s tests -v
```
