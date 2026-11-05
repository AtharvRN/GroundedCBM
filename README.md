# SpG-CBM

Spatially Grounded Concept Bottleneck Models.

A shared training and evaluation pipeline for **CUB, ImageNet, Places365, and
PartImageNet++**, with LF-CBM, VLG-CBM, and SALF-CBM baselines.

## Installation

Use a CUDA-compatible PyTorch/torchvision installation, then:

```bash
pip install -r requirements-release.txt
```

## Usage

- [Data preparation, training, and classification](spatial_cbm/README.md)
- [CUB-70, region deletion, and VLM evaluation](evals/README.md)
- [Dataset and baseline configs](configs/release/)

The configs are starting templates, not exact historical paper recipes.
SALF-CBM training is a documented paper-based reconstruction.
