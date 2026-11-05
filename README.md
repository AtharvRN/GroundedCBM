# SpG-CBM

Spatially Grounded Concept Bottleneck Models.

A shared training and evaluation pipeline for **CUB, ImageNet, Places365, and
PartImageNet++**, with LF-CBM, VLG-CBM, and SALF-CBM baselines.

## Setup

Use a CUDA-compatible PyTorch/torchvision installation. Run all commands from
the repository root; the example below uses CUB.

```bash
pip install -r requirements-release.txt
CONFIG=configs/cub.yaml
RUN=runs/cub/seed42
```

[Prepare your data](spatial_cbm/README.md#prepare-data), then edit the config's
data, backbone checkpoint, cache, and output paths. `RUN` must match
`training.output`. Choose another [config](configs/) for a different dataset.

## Train SpG-CBM

```bash
python -m spatial_cbm.training check --config "$CONFIG"
python -m spatial_cbm.training train --config "$CONFIG"
```

Add `--resume` to the training command after an interruption. The checkpoint
with minimum validation loss is saved as `$RUN/best.pt`.

## Concept and Localization Evaluation

Calibrate thresholds on internal validation data, then evaluate held-out test data:

```bash
python -m spatial_cbm.training calibrate --config "$CONFIG"
python -m spatial_cbm.training evaluate --config "$CONFIG"
```

Results: `$RUN/test_metrics.json`. Test data is never used for calibration.

## Classification: ACC@NEC5 and AVGACC

Export features, fit sparse classifiers, and evaluate them:

```bash
python -m spatial_cbm.training export --config "$CONFIG"
python -m spatial_cbm.training.classifier --artifact_dir "$RUN" \
  --output_dir "$RUN/nec" --device cuda --seed 42 --saga_workers 0 \
  --table_device cuda --cache_features_device cuda_preserve_sampler \
  --n_iters 1000 --max_glm_steps 150 --lam_max 0.1 --epsilon 0.00001 --skip_train_eval
python -m spatial_cbm.training classify --config "$CONFIG" --classifier-dir "$RUN/nec"
```

Results: `$RUN/nec/classification_test.json`, including accuracy at NEC
5, 10, 15, 20, 25, and 30 and their average (AVGACC).

## CUB-70 Test

Use a CUB-trained model, its validation calibration, and human part masks:

```bash
python -m evals.cub70 --config "$CONFIG" \
  --mapping evals/cub70_assets/mapping.json --masks-root /data/CUB-70/masks \
  --output results/cub70.json
```

See the [mask format](evals/cub70_assets/README.md#inputs) for dataset layout.

## Deletion Test

After classifier fitting, compare targeted patch deletion with random controls:

```bash
python -m evals.deletion --config "$CONFIG" \
  --head-dir "$RUN/nec" --count 100 --output results/deletion/cub
```

This runs a 100-image check using NEC=5, 32x32 patches, and three random controls.
Remove `--count 100` for the full test split. Results: `results/deletion/cub/results.json`.

## VLM as a Judge

Export validation examples for calibration and 100 test images for a pilot:

```bash
python -m evals.vlm export --config "$CONFIG" --head-dir "$RUN/nec" \
  --split validation --count 200 --output results/vlm/val
python -m evals.vlm export --config "$CONFIG" --head-dir "$RUN/nec" \
  --split test --count 100 --output results/vlm/test
```

Set `VLM_API_KEY`, `VLM_BASE_URL` (HTTPS API base URL), `JUDGE_MODEL`, and
`JUDGE_BACKEND` (the exact model ID returned by your provider), then run:

```bash
python -m evals.vlm calibrate --export results/vlm/val \
  --base-url "$VLM_BASE_URL" --model "$JUDGE_MODEL" --expected-model "$JUDGE_BACKEND" \
  --output results/vlm/calibration
python -m evals.vlm evaluate --export results/vlm/test \
  --calibration results/vlm/calibration/frozen_threshold.json \
  --base-url "$VLM_BASE_URL" --model "$JUDGE_MODEL" --expected-model "$JUDGE_BACKEND" \
  --output results/vlm/evaluation
```

Results are saved under `results/vlm/evaluation/`. Use distinct output paths
for different models or datasets. More options: [evaluation guide](evals/README.md).

For LF-CBM, VLG-CBM, and SALF-CBM, see [baseline setup](spatial_cbm/README.md#baselines).
Configs are starting templates, not exact historical paper recipes; SALF-CBM
training is a paper-based reconstruction.
