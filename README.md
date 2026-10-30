# Spatial CBM: Unified Training Release

One training pipeline for **CUB, ImageNet, Places365, and PartImageNet++**.
Dataset differences belong in preparation/configuration, not in the model,
loss, optimizer, validation, or checkpoint-selection code.

This branch is a **release candidate**. Synthetic integration tests cover all
four dataset configurations. Real-data GPU parity and throughput must be
verified before replacing existing experiment pipelines.

## Scope

- ResNet-18/50/101 backbone training or explicit pretrained checkpoints.
- SG-CBM frozen-backbone training with conv4+5 or conv3+4+5 spatial maps.
- LF-CBM, VLG-CBM, and a paper-based SALF-CBM reconstruction in the same pipeline.
- Standard mean BCEWithLogitsLoss and vectorized spatial KL/soft BCE losses for SG-CBM.
- FP16/BF16 training, FP32 validation/loss accumulation, channels-last.
- Minimum validation loss selects `best.pt`; early-stopping patience defaults to 10.
- Validation concept/pixel AP each epoch; saved validation-only thresholds for final metrics.
- Shared feature export, training-only normalization, and GLM-SAGA/NEC fitting.
- Held-out ACC@NEC5/AVGACC, separate CUB-70 evaluation, and explicit PartImageNet++ human supervision.
- Node-local target caches and optional direct image reads from indexed archives.
- Saved split/data/source identities and guarded deterministic resume.

CUB-70 human-part evaluation lives in its own `cub70/` folder. Legacy dataset-specific
trainers are preserved in Git history; the new baseline heads use shared orchestration.
See [migration notes](docs/migration.md).

## Code Layout

There are two shared training/classification entry points, not separate scripts for each dataset:
`python -m spatial_cbm.training` and
`python -m spatial_cbm.training.classifier`.
Independent CUB-70 evaluation uses `python -m cub70`.

The remaining files are internal modules with distinct responsibilities:

- `layers.py`, `losses.py`, `targets.py`: model layers, objectives, and target caching/batching.
- `training/config.py`, `manifest.py`, `data.py`: configuration and dataset preparation/loading.
- `training/model.py`, `engine.py`, `backbone_recipe.py`: backbone/head construction and training.
- `training/baselines.py`, `teacher.py`: baseline heads/objectives and cached CLIP targets.
- `training/export.py`, `classifier.py`, `classifier_cache.py`: feature export and sparse classifier fitting.
- `training/metrics.py`: shared AP monitoring, calibration, and frozen-threshold evaluation.

Loss implementations, target helpers, and checkpoint selection are consolidated;
unused legacy helpers are removed. The GLM-SAGA solver remains separate.

## Install

```bash
pip install -r requirements-release.txt
python -m pytest -q tests/release
python -m spatial_cbm.training --help
```

Choose a compatible PyTorch/torchvision CUDA installation. No datasets,
checkpoints, credentials, or cluster endpoints are embedded in this interface.

## Baselines

Choose `model.method: sg_cbm`, `vlg_cbm`, `lf_cbm`, or `salf_cbm`.
The three baseline templates are in `configs/release/baselines/`. Edit their data,
backbone, output, and teacher paths; use a separate run directory per method.
All use the same dataset preparation, trainer, minimum-validation-loss selection,
feature export, calibration, and NEC classifier. No new training scripts are needed.

LF-CBM and SALF-CBM require the optional official OpenAI CLIP package and an explicit
local ViT-B/16 checkpoint. Prepare their teacher cache once before training:

```bash
pip install git+https://github.com/openai/CLIP.git
python -m spatial_cbm.training teacher --config configs/release/baselines/lf_cbm.yaml
python -m spatial_cbm.training train --config configs/release/baselines/lf_cbm.yaml
python -m spatial_cbm.training calibrate --config configs/release/baselines/lf_cbm.yaml
python -m spatial_cbm.training evaluate --config configs/release/baselines/lf_cbm.yaml
python -m spatial_cbm.training export --config configs/release/baselines/lf_cbm.yaml
```

VLG-CBM uses detector labels and training-only box crops; it needs no teacher cache.
LF/SALF regress CLIP scores without detector labels. AP/calibration against GDINO or
human annotations still requires those annotations explicitly; teacher scores are
never substituted for evaluation ground truth. Regression validation is centered
over the full validation cohort, not separately over minibatches. Training batches
must contain at least three images; one/two-image tails merge into the previous batch.

For LF's upstream full-training-set projection updates, enable
`training.cache_regression_features: true` with FP32 training. Frozen features
are cached once and identity-checked; this option also supports SALF but rejects
VLG's random-crop training. `feature_batch_size` controls cache extraction and
`evaluation_batch_size` separately limits validation/export/test image batches.
The CUB LF template uses a 50,000-image cap (one update per epoch on CUB),
1,000 maximum epochs and patience 50. Validation runs every epoch, unlike
upstream's every-50-step checkpoint checks; this is not identical stopping behavior.
The ImageNet controlled recipes and their separate settings are documented in
[the baseline audit](docs/baselines.md#imagenet-controlled-launch-2026-10-30).

LF/VLG linear heads expose **post-hoc linear CAMs**, not native spatial explanations.
A nonlinear VLG head (`model.vlg_hidden > 0`) reports localization as unavailable.
SALF uses native spatial maps. LF's two vocabulary filters are saved with the selected
checkpoint; exported concepts, metrics, and classifiers use the retained order.

These are method implementations under our shared protocol, **not exact published-result
reproductions**. In particular, full SALF training code was not released; its training
path is a documented paper-based reconstruction. Supported student backbones remain
torchvision ResNets, not CLIP-RN50 or arbitrary Hugging Face checkpoints. See the
[pinned upstream audit, objectives, and limitations](docs/baselines.md).

## Data Contract

Prepare a JSONL manifest containing the **training pool only**, never the
official test split. Each row has an exact image identity and zero-based class:

```json
{"id":"bird/001.jpg","image":"images/bird/001.jpg","class_id":0,"detections":[{"concept":"black bill","confidence":0.91,"box":[0.2,0.1,0.4,0.3],"coordinates":"normalized_xyxy"}]}
```

Paths are relative to the source manifest. Class ids must be contiguous;
concept names must match the vocabulary exactly (VLG additionally applies upstream
lowercase/punctuation/article normalization and rejects duplicate normalized names). Explicit `detections: []`
means a verified negative image. Missing annotations cannot silently become
negative labels. Box coordinates are explicitly `normalized_xyxy` or
`pixel_xyxy`, never inferred from their numerical range.

Alternatively provide `annotation`, `annotation_image_id`, and
`box_coordinates`. Supported annotation formats are GDINO's
`[metadata, detection, ...]` or `{"image_id":...,"detections":[...]}`.
The header filename/id must match `annotation_image_id` exactly.

```bash
python -m spatial_cbm.training prepare --source manifests/pinpp-train.jsonl \
  --test-source manifests/pinpp-test.jsonl \
  --concepts concept_files/partimagenetpp_generic_parts.txt --dataset partimagenetpp \
  --output prepared/partimagenetpp --fraction 0.05 --seed 6885
```

This saves separate training-pool/test databases, a vocabulary, and explicit
`split.json`/`indices.npz`. Test image IDs and image references must be disjoint
from the entire training pool. Test annotations are packaged during preparation,
but training and calibration never load test samples.

The split follows VLG-CBM's rule: seeded `torch.randperm` in manifest order,
with `int(N * fraction)` validation images taken from the permutation's tail.
Defaults are 10% for CUB/ImageNet and 5% for Places365/PartImageNet++, seed 6885.
PartImageNet++ uses this as our internal split of its 90K training pool, not an
author-released VLG split; its 10K test set is separate. The split seed is
independent of training seeds. Prepare once and reuse the **same directory for
all methods/seeds, backbone training, and concept training**.

Supply `--split existing-split.json` to reuse audited image IDs. Matching the
upstream rule/seed is not proof of identical author-side membership unless the
ordered image roster also matches. Manifest IDs must be canonical across train
and test; this check does not detect duplicate image bytes under different IDs.

Native-layout adapters emit the same schema:

```bash
python -m spatial_cbm.training manifest --layout cub --root /data/CUB_200_2011 --output manifests/cub.jsonl
python -m spatial_cbm.training manifest --layout cub --partition test --root /data/CUB_200_2011 --output manifests/cub-test.jsonl
python -m spatial_cbm.training manifest --layout imagefolder --root /data/imagenet/train --output manifests/imagenet.jsonl
python -m spatial_cbm.training manifest --layout imagefolder --partition test --root /data/imagenet/val --class-map manifests/imagenet.classes.json --output manifests/imagenet-test.jsonl
python -m spatial_cbm.training manifest --layout places365 --root /data/places365 --filelist /data/places365_train_standard.txt --output manifests/places365.jsonl
python -m spatial_cbm.training manifest --layout manifest --root /data/pinpp --filelist /data/train90k.jsonl --output manifests/pinpp.jsonl
```

ImageFolder manifests save a sibling `*.classes.json` mapping. Evaluation must
reuse the training mapping with `--class-map`; missing evaluation classes retain
their original IDs and unknown classes are rejected. Rows also retain class names
so preparation rejects inconsistent train/test mappings. Existing manifests must
be audited or regenerated; the adapter does not repair their labels in place.

For concept training, add `--annotation-map mapping.json`. It maps each exact
sample id to an object with `annotation`, `annotation_image_id`, and
`box_coordinates`; annotation paths are relative to that mapping file. Never
infer annotation ids from filename sort positions. Already-indexed archives
can use `image: {"archive":"train.tar","offset":123,"size":456}` to address
JPEG bytes directly, including inside a nested tar.

## Train And Resume

The default `model.fusion: residual` retains `global + alpha * spatial`.
For fusion ablations, set `model.fusion: convex` to use
`alpha * global + (1 - alpha) * spatial`. Fixed alpha must lie in `[0, 1]`. Set
`model.learn_alpha: true` and `model.alpha: 0.5` to learn one shared scalar via
a sigmoid parameterization. History records its value after each epoch, and
`complete.json` records `best_alpha` at the minimum-validation-loss checkpoint.
The KL and spatial-BCE objectives are unchanged, including at alpha 0 and 1.
Use fresh runs for these ablations, not altered evaluation of existing checkpoints.

Spatial logits use normalized log-sum-exp pooling by default
(`model.spatial_pooling: lse`). Set `model.spatial_pooling: topk` and
`model.topk_fraction: 0.05` to average the highest 5% of each concept's raw
spatial logits. The selected count is `ceil(H * W * topk_fraction)`, with at
least one cell. Fractions must be in `(0, 1]`; negative logits are retained.
This changes only spatial pooling, not the maps, fusion coefficient, KL loss,
or spatial BCE. The saved configuration controls training and all subsequent
evaluation and feature export; changing pooling requires a fresh run.

Edit the paths and hyperparameters in `configs/release/{dataset}.yaml`:

```bash
python -m spatial_cbm.training check --config configs/release/partimagenetpp_backbone.yaml
python -m spatial_cbm.training train --config configs/release/partimagenetpp_backbone.yaml
# Set model.checkpoint in the concept config to the backbone run's best.pt.
python -m spatial_cbm.training train --config configs/release/partimagenetpp.yaml
python -m spatial_cbm.training train --config configs/release/partimagenetpp.yaml --resume
```

Use exactly the same commands with `cub.yaml`, `imagenet.yaml`, or
`places365.yaml`. SG-CBM concept prediction uses PyTorch's mean `BCEWithLogitsLoss`
(via its functional equivalent), with `positive_weight: 1.0` in all default
recipes: no positive-class reweighting or automatic class balancing. Optional
positive weights use PyTorch's element-count normalization, not division by
the sum of weights. This replaces the earlier release's weight-sum-normalized
BCE; old checkpoints must not be resumed under the changed objective.

Validation runs after **every epoch**, using the same combined training
objective and native target resolution. Strict improvements reset
patience; ties retain the earlier checkpoint. F1/AP and test accuracy never
select the training checkpoint. The test set is not loaded by training.
The same validation forward pass also logs exact concept AP and spatial AP when
annotation labels and spatial maps are available.
F1 and threshold searches are omitted from the epoch loop.

Outputs: `protocol.json`, `history.json`, `best.pt`, `last.pt`, `complete.json`.
Resume checks configuration, source hashes, data, split, backbone identity, and
best/last consistency. Cache paths must be node-local, not NFS/PVC. Stage images
and prepared data locally when throughput matters.

## Calibration And Evaluation

After training finishes, calibrate only the minimum-validation-loss checkpoint:

```bash
python -m spatial_cbm.training calibrate --config configs/release/cub.yaml
python -m spatial_cbm.training evaluate --config configs/release/cub.yaml
```

The first command writes `calibration.json` from the internal validation split.
The second writes `test_metrics.json` using the saved thresholds unchanged.
Both are tied to checkpoint, vocabulary, source, and split/data hashes. Missing
or mismatched calibration is an error, never an invitation to tune on test data.

- Concept AP: exact rank-based macro AP over concepts with positive labels;
  supported/total concept counts are recorded. This is not pooled micro AP.
- Concept Precision, Recall, F1: micro-averaged over image-concept pairs, using
  `sigmoid(concept_score) >= threshold` (SG-CBM uses its fused logit). One global threshold maximizes validation
  micro F1 on the fixed grid 0, 0.001, ..., 1 (plus a predict-none candidate).
  This is a grid maximum, not an exact optimum over every observed score.
- Spatial AP: exact pixel AP averaged over visible detected image-concept pairs,
  after bilinear upsampling to input resolution. A pixel is positive if its
  image region overlaps a retained GDINO box. This uses unthresholded maps.
- Localization threshold: per-map min-max normalized upsampled logits, with
  one threshold chosen for maximum validation mean foreground IoU on the grid
  0, 0.02, ..., 1. Constant maps normalize to zero. Maps are evaluated for
  detected concepts independently of concept-presence thresholding.

Ties choose the highest threshold. Scores are stored on the 0-to-1 scale, along
with the grids, objectives, aggregation definitions, and selected thresholds.
By default these metrics measure agreement with GDINO, not independent human
annotations. The explicit PartImageNet++ human mode instead uses human part
labels/regions for training and validation. Neither mode tunes on test masks.

Exact AP ranking/sorting adds validation compute; it may cost more than an F1
grid search. Concept logits/labels are spooled to node-local scratch rather than
kept on GPU, and spatial sorts are chunked. Full-scale performance is unverified.

## Sparse Classifiers

```bash
python -m spatial_cbm.training export --config configs/release/partimagenetpp.yaml
python -m spatial_cbm.training.classifier --artifact_dir runs/partimagenetpp/seed42 \
  --output_dir runs/partimagenetpp/seed42/nec --device cuda --seed 42 \
  --saga_workers 0 --saga_batch_size 512 --table_device cuda \
  --cache_features_device cuda_preserve_sampler --n_iters 1000 \
  --max_glm_steps 150 --lam_max 0.1 --epsilon 0.00001 \
  --nec_values 5,10,15,20,25,30 --skip_train_eval
```

The classifier uses the internal validation split, never test data. CUDA feature
caching preserves sampler order and checks available memory. Use
`cpu_preserve_sampler` or `none` if necessary. NEC selects the first path point
reaching each sparsity, then keeps exactly `NEC * number_of_classes` weights.
Unreached targets fail explicitly; selection does not use test accuracy.

Evaluate the saved heads on the entire held-out test split in one feature pass:

```bash
python -m spatial_cbm.training classify --config configs/release/partimagenetpp.yaml \
  --classifier-dir runs/partimagenetpp/seed42/nec --nec-values 5,10,15,20,25,30
```

`classification_test.json` contains percent accuracy at each NEC, `ACC_NEC5`,
and `AVGACC` (the arithmetic mean over the listed NEC values). Classifier fitting
saves `classifier_identity.json`; evaluation verifies those head hashes,
training-only normalization, selected checkpoint, and feature provenance. It
does not select/refit heads on test data. Class-only test images do not require
concept annotations. Results are per fitted seed; multi-seed aggregation is
separate. Older heads without an identity manifest are not silently trusted.

## CUB-70

Use CUB-trained models, the frozen concept-to-part mapping, and the saved CUB
validation localization threshold. No human test-mask calibration is performed.

```bash
python -m cub70 --config configs/release/cub.yaml --mapping cub70/mapping.json \
  --masks-root /data/CUB-70/masks --output results/cub70.json
```

See [CUB-70 data formats and protocol](cub70/README.md). This evaluator selects
the highest-fused-score concept **among those mapped to each part**, not among
the entire bank, and reports per-part and aggregate pixel AP/foreground IoU.

## PartImageNet++ Human Supervision

Use the same trainer with `data.supervision: human`; all other datasets remain
GDINO-only. This is a **supervised ablation**, not a label-free result. The
vocabulary must contain human part names, not unverified color/texture attributes.
Keep the image roster, split IDs, backbone, and vocabulary fixed for a controlled
GDINO-versus-human comparison.

Human manifests use `human_parts` instead of `detections`:

```json
{"id":"bird.jpg","image":"bird.jpg","class_id":0,"human_parts":[{"concept":"head","coordinates":"pixel_xy","polygons":[[10,10,30,10,30,30,10,30]]}]}
```

All polygons for a concept are unioned in the original image, resized/cropped
with nearest-neighbor interpolation, then average-pooled into soft grid targets.
Part presence is defined on the full image. Explicit `human_parts: []` means no
annotated concepts; missing labels are errors. Human box targets are also
accepted with `box` and explicit `pixel_xyxy`/`normalized_xyxy` coordinates.
Human labels ignore detector confidence and never fall back to GDINO.

For COCO-style polygon annotations, use the existing manifest adapter with
`--human-annotations annotations.json --category-map category_to_concept.json`.
The category map is an explicit object such as `{"8":"head","9":null}`; null
excludes a category. Every encountered category must be listed. Images join by
exact `id == file_name`, or an explicit `annotation_image_id`, never row position.
RLE masks require conversion to polygons before this adapter; they are rejected.

```bash
python -m spatial_cbm.training prepare --source manifests/pinpp-human-train.jsonl \
  --test-source manifests/pinpp-human-test.jsonl --dataset partimagenetpp \
  --supervision human --concepts concept_files/partimagenetpp_generic_parts.txt \
  --split prepared/partimagenetpp/split.json --output prepared/partimagenetpp-human
python -m spatial_cbm.training train --config configs/release/partimagenetpp_human.yaml
```

The human config uses a separate output directory. Release-format backbones may
be shared across annotation modes only when their ordered image/class roster and
internal split match exactly. Use the same `calibrate`, `evaluate`, `export`,
classifier-fitting, and `classify` commands afterward. Output metadata identifies
human supervision explicitly; do not mix these numbers with GDINO-only results.

## Migration Boundaries

These are **prospective recipes**, not claimed reproductions of historical
paper results. Reuse audited manifests to match prior splits; do not regenerate
partitions per training seed. ImageNet's example uses a 75-epoch maximum rather
than the historical 10, with patience 10. No experiment is launched by editing
these configurations.

The backbone template carries over SGD/cosine, label smoothing, random resized
crops/flips, EMA, mixup/CutMix, RandAugment, and erasing. It now selects by
internal validation loss, uses epochwise learning-rate updates, and evaluates
label-smoothed cross-entropy without image mixing. Real-data GPU parity and
throughput checks are still required before definitive retraining. External
checkpoints cannot certify an unseen internal holdout; release-format backbone
checkpoints enforce matching prepared data and splits. ViT and arbitrary
MMPretrain checkpoint conversion are not supported yet.

See [architecture and readiness gates](docs/release-architecture.md).
