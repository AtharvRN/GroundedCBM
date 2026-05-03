# Included

- `scripts/train_imagenet_gcbm.py`: final ImageNet concept-layer training entrypoint.
- `scripts/run_glm_path.py`: sparse GLM-SAGA path sweep on saved ImageNet concept features.
- `scripts/eval_imagenet_nec.py`: ImageNet NEC evaluation against val root or val tar assets.
- `scripts/eval_imagenet_localization.py`: ImageNet localization evaluation using concept-head checkpoints.
- `scripts/train_cub_gcbm.py`: CUB G-CBM training entrypoint through the preserved SAVLG trainer.
- `scripts/eval_cub_nec.py`: CUB sparse/NEC evaluation.
- `scripts/eval_cub_localization.py`: CUB part-localization evaluation for the final G-CBM checkpoint.
- `scripts/render_imagenet_spatial_grid.py`: paper figure helper for six curated ImageNet examples.
- `gcbm/imagenet_core.py`: minimal copied ImageNet training/eval core, including bbox transforms and target precompute.
- `glm_saga/elasticnet.py`: only the GLM-SAGA implementation used by sparse evaluation.
- `methods/`, `model/`, `data/`, `evaluations/`: preserved legacy CUB dependency slice required by the final SAVLG/G-CBM path.
- `concept_files/`: released ImageNet and CUB concept sets.
- `configs/`: example configs for ImageNet and CUB runs.
- `tests/`: focused bbox, precompute, annotation-mapping, and localization-metric smoke tests.

# Intentionally Excluded

- Generated results, saved checkpoints, logs, TensorBoard runs, cached activations, memmaps.
- Notebooks, docs, paper artifacts, slide assets, and local scratch exports.
- K8s jobs, pod bootstrap scripts, cluster helpers, and environment-specific shell wrappers.
- Historical ablation scripts not needed for the final ImageNet/CUB paper tables.
- Download helpers, dataset unpack helpers, and one-off annotation audit utilities.
- Baseline-specific visualization scripts not required to run the final G-CBM evaluations.
