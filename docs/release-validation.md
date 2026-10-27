# Release Candidate Validation

Validated on 2026-10-27:

- Local Python 3.12 / PyTorch 2.9.1: 43 tests passed (19.61 seconds).
- NRP Python 3.10 / PyTorch 2.2.2 / torchvision 0.17.2: 43 tests passed (31.65 seconds).
- NRP job: `spatial-cbm-release-smoke-20261027-r8`, CPU-only, no experiment jobs changed.
- Container: `ghcr.io/atharvrn/savlgcbm-cbm@sha256:888fc0f5d7e326c6ed18ffae20060ca2b529ac03acb97b1350aa525a7d96235d`.
- `git diff --cached --check`: clean.

The suite exercises synthetic end-to-end training/export for all four dataset
configurations, minimum-loss selection, split isolation, checkpoint provenance,
cached targets, archive reading, model/loss parity, deterministic interrupted
resume (including backbone EMA/mixing), and a small cached GLM-SAGA fit. It also
checks that legacy executable paths are absent and all internal imports resolve
within the retained source tree. The NRP payload excludes Python bytecode caches
and the deleted modules. Shared training/classification commands use the
`spatial_cbm` namespace; CUB-70 uses `cub70`. Source-tree checks reject imports
from the old `gcbm` package.

Validation/calibration tests check VLG-style permutation membership and default
fractions, train/test identity separation, exact AP with tied scores, constant
maps, absent concept positives, maximum micro F1 over the declared grid, saved
threshold reuse, and rejection of mismatched calibration. The integration test
observes calibration reading only the training-pool database and evaluation
reading only the separate held-out test database.

Additional integration tests cover identity-checked NEC classification without
concept test labels, ACC@NEC5/AVGACC output, human-supervised PartImageNet++ using
the same trainer and split-matched backbone, polygon union before pooling, and
explicit COCO image/category joins. CUB-70 tests verify mapped-concept selection,
held-out image restrictions, coverage reporting, and frozen validation thresholds.

It does not establish full-dataset numerical reproduction, CUDA AMP parity,
GPU throughput, or convergence of the paper experiments. These remain required
before promoting this candidate to the definitive experiment pipeline.

Run locally with `python -m pytest -q tests/release`.
