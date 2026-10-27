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
| `train_cbm.py`, `gcbm/train_imagenet.py`, `methods/savlg.py` | `python -m gcbm.training train --config ...` |
| `gcbm/imagenet_config.py`, old dataset configs | `gcbm/training/config.py`, `configs/release/` |
| ImageNet loaders, annotation indexes, target precomputation | Shared `manifest`, `prepare`, and target cache in `gcbm/training/data.py` |
| `gcbm/imagenet_models.py` | `gcbm/training/model.py` and shared `gcbm/sg_model.py` |
| `gcbm/imagenet_final_layers.py`, `gcbm/imagenet_glm_path.py`, `scripts/train_sparse_nec.py` | Shared `export` and `python -m gcbm.training.classifier` |
| Legacy evaluation scripts and baseline trainers | Deferred, not silently redirected or supported by this release |

Dataset-specific vocabulary files and YAML configuration are intentional.
Native CUB/Places365 layouts still need small manifest adapters; they do not
have separate training loops, losses, model implementations, or solvers.

This is not a backward-compatible CLI replacement. Use the documented prepared
manifest and explicit checkpoints; old experimental checkpoint formats are not
automatically converted. Existing published results retain their original code
and configuration provenance.
