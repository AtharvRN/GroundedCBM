# SG-CBM: Unified Training Release

One training pipeline for **CUB, ImageNet, Places365, and PartImageNet++**.
Dataset differences belong in preparation/configuration, not in the model,
loss, optimizer, validation, or checkpoint-selection code.

This branch is a **release candidate**. Synthetic integration tests cover all
four dataset configurations. Real-data GPU parity and throughput must be
verified before replacing existing experiment pipelines.

## Scope

- ResNet-18/50/101 backbone training or explicit pretrained checkpoints.
- Frozen-backbone concept training with conv4+5 or conv3+4+5 spatial maps.
- Existing normalized fused BCE and vectorized spatial KL/soft BCE losses.
- FP16/BF16 training, FP32 validation/loss accumulation, channels-last.
- Minimum validation loss selects `best.pt`; early-stopping patience defaults to 10.
- Validation concept/pixel AP each epoch; saved validation-only thresholds for final metrics.
- Shared feature export, training-only normalization, and GLM-SAGA/NEC fitting.
- Node-local target caches and optional direct image reads from indexed archives.
- Saved split/data/source identities and guarded deterministic resume.

Independent evaluation modules will be added later. Legacy dataset-specific
trainers, baseline implementations, and their evaluation scripts are not part
of this training-only SG-CBM release. Their source is preserved in Git history.
See [migration notes](docs/migration.md).

## Code Layout

There are two command entry points, not separate scripts for each dataset:
`python -m spatial_cbm.training` and
`python -m spatial_cbm.training.classifier`.

The remaining files are internal modules with distinct responsibilities:

- `layers.py`, `losses.py`, `targets.py`: model layers, objectives, and target caching/batching.
- `training/config.py`, `manifest.py`, `data.py`: configuration and dataset preparation/loading.
- `training/model.py`, `engine.py`, `backbone_recipe.py`: backbone/head construction and training.
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

## Data Contract

Prepare a JSONL manifest containing the **training pool only**, never the
official test split. Each row has an exact image identity and zero-based class:

```json
{"id":"bird/001.jpg","image":"images/bird/001.jpg","class_id":0,"detections":[{"concept":"black bill","confidence":0.91,"box":[0.2,0.1,0.4,0.3],"coordinates":"normalized_xyxy"}]}
```

Paths are relative to the source manifest. Class ids must be contiguous;
concept names must match the vocabulary exactly. Explicit `detections: []`
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
python -m spatial_cbm.training manifest --layout places365 --root /data/places365 --filelist /data/places365_train_standard.txt --output manifests/places365.jsonl
python -m spatial_cbm.training manifest --layout manifest --root /data/pinpp --filelist /data/train90k.jsonl --output manifests/pinpp.jsonl
```

For concept training, add `--annotation-map mapping.json`. It maps each exact
sample id to an object with `annotation`, `annotation_image_id`, and
`box_coordinates`; annotation paths are relative to that mapping file. Never
infer annotation ids from filename sort positions. Already-indexed archives
can use `image: {"archive":"train.tar","offset":123,"size":456}` to address
JPEG bytes directly, including inside a nested tar.

## Train And Resume

Edit the paths and hyperparameters in `configs/release/{dataset}.yaml`:

```bash
python -m spatial_cbm.training check --config configs/release/partimagenetpp_backbone.yaml
python -m spatial_cbm.training train --config configs/release/partimagenetpp_backbone.yaml
# Set model.checkpoint in the concept config to the backbone run's best.pt.
python -m spatial_cbm.training train --config configs/release/partimagenetpp.yaml
python -m spatial_cbm.training train --config configs/release/partimagenetpp.yaml --resume
```

Use exactly the same commands with `cub.yaml`, `imagenet.yaml`, or
`places365.yaml`. Validation runs after **every epoch**, using the weighted
training objective and native target resolution. Strict improvements reset
patience; ties retain the earlier checkpoint. F1/AP and test accuracy never
select the training checkpoint. The test set is not loaded by training.
The same validation forward pass also logs exact concept AP and spatial AP.
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
  `sigmoid(fused_logit) >= threshold`. One global threshold maximizes validation
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
These metrics measure agreement with GDINO, not independent human annotations.
Human-mask evaluation remains separate and must not tune on human test masks.

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
