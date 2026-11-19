# Where Concepts Fire: Spatially Grounded Concept Bottleneck Models

**SpG-CBM** predicts which concepts are present in an image, shows where they
appear, and uses the concept scores to predict the class. This repository
includes training, sparse classification, and evaluation code.

## Get the data

Download **one** dataset to get started. The example below uses CUB.

| Dataset | What to download |
| --- | --- |
| [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) | Extract `CUB_200_2011.tgz`. Keep the images and the supplied train/test split together. |
| [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) | Sign in and download the ILSVRC 2012 training images, validation images, and development kit (validation labels). |
| [Places365](https://places2.csail.mit.edu/download.html) | Use the official access form for **Places365-Standard**. Request the training images, validation images, and matching file lists with scene labels. |
| [PartImageNet++](https://huggingface.co/datasets/lixiao20/PartImageNetPP) | Download the human part annotations and obtain the corresponding images from ImageNet-1K. Human masks and detector boxes are separate inputs. |

**Grounding DINO annotations.** SpG-CBM learns its spatial maps from concept
boxes, not just image labels. For CUB, ImageNet, and Places365, start with the
[VLG-CBM annotation download](https://github.com/Trustworthy-ML-Lab/VLG-CBM#training-pipeline).
For missing images or a different concept list, use the upstream
[annotation generator](https://github.com/Trustworthy-ML-Lab/VLG-CBM#annotation-generation-optional).
PartImageNet++ is not supported by that generator out of the box: for
detector-supervised training, supply boxes for its own part vocabulary.

## Train on CUB

From the repository root, install the dependencies:

```bash
python -m pip install -r requirements-release.txt
```

1. [Pair the images with their boxes and prepare the splits](spatial_cbm/README.md#prepare-data). This produces `prepared/cub`.
2. In [the CUB config](configs/cub.yaml), set `model.checkpoint` to your CUB-trained ResNet-18 backbone (torchvision state-dict format; weights are not bundled).
3. Check the inputs and start training:

```bash
CONFIG=configs/cub.yaml
RUN=runs/cub/seed42  # training.output in the config
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

## Spatial Concept Intervention

Correct up to five missed concepts using supplied region masks, keeping the
global branch and NEC classifier frozen (paper Experiment V):

```bash
python -m evals.intervention export --config "$CONFIG" \
  --head-dir "$RUN/nec" --feedback /data/feedback-test.json \
  --split test --output results/intervention/test
python -m evals.intervention evaluate --export results/intervention/test \
  --rule replace --output results/intervention/results.json
```

Results include base/edited accuracy, percentage-point gain, and fixed/harmed
predictions over the full supplied cohort. See [feedback inputs and SALF-CBM
calibration](evals/README.md#spatial-concept-intervention). Evaluation of saved
exports is CPU-only; it does not rerun the backbone.

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
