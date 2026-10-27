# Unified Training Architecture

`gcbm/training/config.py` defines one strict YAML schema. `manifest.py` contains
native-layout adapters; `data.py` creates identity-keyed SQLite annotations and
one reusable split. `model.py` reuses the shared multiscale concept layer.
`engine.py` owns the only release epoch loop, precision policy, validation-loss
selection, checkpointing, and early stopping. `export.py` creates selected-head
features and training-only normalization. `classifier.py` runs shared GLM-SAGA.

There is no dataset-name dispatch in the training engine, model, or objective.
ImageNet's useful archive reading is a generic optional image backend, not
another model or trainer. All datasets use the vectorized spatial losses.

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
   candidate; see [validation record](release-validation.md).
3. Before large jobs, run real-data GPU/mixed-precision smoke tests, compare
   targets/model outputs against pinned historical inputs, and benchmark I/O.
4. Before backbone retraining, verify the ported strong augmentation and EMA
   recipe on real data and fix the internal split in advance. The new trainer
   selects by validation loss, not held-out test accuracy. Its cosine learning
   rate is updated per epoch rather than per batch; validation cross-entropy
   uses the configured label smoothing, without mixup or CutMix.
5. Add the independent evaluations later, as requested.

Synthetic test success is not a claim of reproduced paper accuracy or measured
GPU throughput. Existing results stay attributed to their original code and
configuration until the release parity checks are complete.
