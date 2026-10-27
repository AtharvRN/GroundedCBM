# Legacy Code And Release Scope

The release tree contains one SG-CBM training pipeline, shared by CUB, ImageNet,
Places365, and PartImageNet++. The initial release candidate left old executable
paths alongside it; those paths have now been removed rather than renamed.

Historical source and commands remain available at
[commit 3106615f](https://github.com/AtharvRN/GroundedCBM/tree/3106615f)
and on the research branches. No running experiment or research worktree was
modified by this cleanup.

| Historical path | Release path |
| --- | --- |
| `train_cbm.py`, `gcbm/train_imagenet.py`, `methods/savlg.py` | `python -m spatial_cbm.training train --config ...` |
| `gcbm/imagenet_config.py`, old dataset configs | `spatial_cbm/training/config.py`, `configs/release/` |
| ImageNet loaders, annotation indexes, target precomputation | Shared `manifest`, `prepare`, and target cache in `spatial_cbm/training/data.py` |
| `gcbm/imagenet_models.py` | `spatial_cbm/training/model.py` and shared `spatial_cbm/layers.py` |
| `gcbm/imagenet_final_layers.py`, `gcbm/imagenet_glm_path.py`, `scripts/train_sparse_nec.py` | Shared `export` and `python -m spatial_cbm.training.classifier` |
| Legacy evaluation scripts and baseline trainers | Deferred, not silently redirected or supported by this release |

Dataset-specific vocabulary files and YAML configuration are intentional.
Native CUB/Places365 layouts still need small manifest adapters; they do not
have separate training loops, losses, model implementations, or solvers.

This is not a backward-compatible CLI replacement. Use the documented prepared
manifest and explicit checkpoints; old experimental checkpoint formats are not
automatically converted. Existing published results retain their original code
and configuration provenance.

The Python package is now `spatial_cbm`, not `gcbm`; update imports and commands
accordingly. The old namespace is not kept as a duplicate compatibility layer.
Losses are in `spatial_cbm/losses.py`, target caching/batching in `targets.py`,
and validation-loss selection in `training/engine.py`. The memory-mapped feature
dataset lives alongside classifier caching. Numerical algorithms are unchanged.

Source hashes intentionally prevent resuming an old run across this refactor.
Use its original pinned commit to resume it; this rename does not migrate or
modify existing run artifacts.

Preparation now requires `--test-source`, audits identity/reference disjointness,
and saves test records in a separate database. The generated split now follows
VLG's seeded permutation and truncated validation count, not the initial
stratified rule. Existing prepared directories without a verified test manifest
cannot train under this protocol; create a new directory, optionally passing
audited `--split` IDs, rather than altering an existing run's split in place.
