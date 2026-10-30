# Unified Training Architecture

`spatial_cbm/training/config.py` defines one strict YAML schema. `manifest.py` contains
native-layout adapters; `data.py` creates identity-keyed SQLite annotations and
one reusable split. `model.py` reuses the shared multiscale concept layer.
`baselines.py` supplies LF/VLG/SALF heads and objectives; `teacher.py` prepares
factorized, identity-checked CLIP targets. See [baseline fidelity](baselines.md).
`engine.py` owns the only release epoch loop, precision policy, validation-loss
selection, checkpointing, and early stopping. `export.py` creates selected-head
features and training-only normalization. `classifier.py` runs shared GLM-SAGA
and evaluates the saved NEC heads on held-out images in one feature pass.
`metrics.py` monitors AP without selecting checkpoints, calibrates thresholds on
validation only, and applies frozen thresholds to the separate test database.
`cub70/` independently evaluates CUB-trained models against human part masks.
PartImageNet++ human supervision is an explicit data option in the same trainer,
not a second training pipeline.

There is no dataset-name dispatch in the training engine, model, or objective.
ImageNet's useful archive reading is a generic optional image backend, not
another model or trainer. All SG-CBM datasets use the vectorized spatial losses;
baseline objectives likewise do not dispatch on dataset names.
Legacy executable paths and their unused dependencies have been removed from
the release tree; see [migration notes](migration.md) for archived source.

## Provenance

Base: `seeds` commit `8d355ebb`. A separate worktree preserves all uncommitted
research experiments. Incorporated changes were deliberately scoped:

- Existing vectorized losses and lossless target cache from `seeds`.
- Validation-loss selection/early-stopping helper from this audit.
- Established conv3/4/5 head algebra from the completed ImageNet/PINPP runs.
- Existing GLM-SAGA solver and sampler-preserving feature-cache implementation.
- Solver convergence metadata; optimization equations unchanged.
- Historical backbone mixup/CutMix, RandAugment, erasing, and warm-started EMA.

Historical feature heads are not silently remapped into the new checkpoint
schema. Explicit torchvision ResNet state dictionaries are accepted as backbone
inputs; incompatible keys fail. The supported release backbones are ResNet-18,
ResNet-50, and ResNet-101.

## Readiness Gates

1. Synthetic integration tests cover all dataset configurations, model/loss
   parity, cached targets, archive reads, native adapters, split/identity checks,
   resume determinism, feature export, cached SAGA, and exact NEC truncation.
2. Repeat these tests in the pinned NRP training container. Completed for this
   earlier SG-CBM candidate; see [validation record](release-validation.md).
   The new baseline integration still needs container/GPU validation.
3. Before large jobs, run real-data GPU/mixed-precision smoke tests, compare
   targets/model outputs against pinned historical inputs, and benchmark I/O.
4. Before backbone retraining, verify the ported strong augmentation and EMA
   recipe on real data and fix the internal split in advance. The new trainer
   selects by validation loss, not held-out test accuracy. Its cosine learning
   rate is updated per epoch rather than per batch; validation cross-entropy
   uses the configured label smoothing, without mixup or CutMix.
5. Check the CUB-70 mapping against the actual checkpoint vocabulary and audit
   mask coverage before reporting benchmark results. Other independent
   evaluations remain outside this release. Baseline training is now shared,
   with explicit upstream/reconstruction differences documented separately.

All datasets now require a saved, test-disjoint internal partition before
training. The VLG permutation/truncated-count rule replaces the initial
stratified default. A training seed does not regenerate the partition.
Calibration uses the selected checkpoint: global concept threshold for maximum
micro F1, separate map threshold for maximum mean foreground IoU. Both grids and
selected thresholds are recorded. See the README for precise AP definitions.

Synthetic test success is not a claim of reproduced paper accuracy or measured
GPU throughput. Existing results stay attributed to their original code and
configuration until the release parity checks are complete.

## Recovery and Aggregation

Epoch checkpoint publication first atomically saves `epoch.pending.pt`, containing
the resumable state and any newly selected best checkpoint. Under the run writer
lock, resume replays this journal into `best.pt`, `last.pt`, and `history.json`
before continuing or marking completion. An interruption before the journal is
published leaves the previous epoch authoritative. Stale history is rebuilt from
`last.pt`; legacy best/last disagreements without a journal still fail closed.
Source-provenance checks remain strict: these fixes do not silently migrate runs
created with older code.

SG-CBM epoch loss aggregates concept BCE over image-concept elements and spatial
KL/BCE over images with valid region slots, then combines those cohort means with
the configured coefficients. Empty valid targets retain their existing denominator
slots; images with no valid slots do not enter the spatial denominator. Component
means and denominators are recorded. Minibatch optimization is unchanged; validation
selection no longer depends on the partitioning of annotation-sparse images.

Frozen regression feature caches use a blocking, identity-specific file lock.
Builders write a private sibling directory and atomically rename it into place
only after writing the completion manifest and content hashes. Waiting consumers
recheck and verify that published cache. A killed builder releases the lock; its
unpublished staging directory cannot be mistaken for a usable cache. Nonempty
partial caches from the older implementation require a fresh cache directory.
