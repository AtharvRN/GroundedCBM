# Release Candidate Validation

Validated on 2026-10-27:

- Local Python 3.12 / PyTorch 2.9.1: 28 tests passed after the package rename and module consolidation.
- NRP Python 3.10 / PyTorch 2.2.2 / torchvision 0.17.2: 28 tests passed.
- NRP job: `spatial-cbm-release-smoke-20261027-r6`, CPU-only, no experiment jobs changed.
- Container: `ghcr.io/atharvrn/savlgcbm-cbm@sha256:888fc0f5d7e326c6ed18ffae20060ca2b529ac03acb97b1350aa525a7d96235d`.
- `git diff --cached --check`: clean.

The suite exercises synthetic end-to-end training/export for all four dataset
configurations, minimum-loss selection, split isolation, checkpoint provenance,
cached targets, archive reading, model/loss parity, deterministic interrupted
resume (including backbone EMA/mixing), and a small cached GLM-SAGA fit. It also
checks that legacy executable paths are absent and all internal imports resolve
within the retained source tree. The NRP payload excludes Python bytecode caches
and the deleted modules. Both command entry points use the `spatial_cbm`
namespace; the source-tree checks reject imports from the old `gcbm` package.

It does not establish full-dataset numerical reproduction, CUDA AMP parity,
GPU throughput, or convergence of the paper experiments. These remain required
before promoting this candidate to the definitive experiment pipeline.

Run locally with `python -m pytest -q tests/release`.
