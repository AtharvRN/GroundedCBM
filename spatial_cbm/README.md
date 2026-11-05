# Training and Classification

Run from the repository root. The same commands support CUB, ImageNet,
Places365, and PartImageNet++; select the matching config in `configs/release/`.

## Prepare Data

Create separate training-pool and held-out test JSONL manifests. Each row contains
an image ID, image path, zero-based class ID, and annotations:

```json
{"id":"bird/001.jpg","image":"images/bird/001.jpg","class_id":0,"detections":[{"concept":"black bill","confidence":0.91,"box":[0.2,0.1,0.4,0.3],"coordinates":"normalized_xyxy"}]}
```

Image paths are relative to the manifest. Concept names must match the vocabulary.
An empty detection list means a verified negative, not missing annotations.
Use `python -m spatial_cbm.training manifest --help` for native-layout adapters.

```bash
python -m spatial_cbm.training prepare --dataset cub \
  --source manifests/cub-train.jsonl --test-source manifests/cub-test.jsonl \
  --concepts concept_files/cub_filtered.txt --output prepared/cub --seed 6885
```

Preparation saves an internal validation split. Reuse the same prepared directory
across methods and seeds; never use test images for validation.

## Train and Evaluate

Set the prepared-data, backbone checkpoint, node-local cache, and output paths
in the config first. Use a separate output directory per method and seed.

```bash
CONFIG=configs/release/cub.yaml
python -m spatial_cbm.training check --config "$CONFIG"
python -m spatial_cbm.training train --config "$CONFIG"
python -m spatial_cbm.training calibrate --config "$CONFIG"
python -m spatial_cbm.training evaluate --config "$CONFIG"
```

Add `--resume` to `train` after an interruption. Minimum validation loss selects
`best.pt`; calibration uses validation only. Test results are in `test_metrics.json`.
For PartImageNet++ human supervision, use `partimagenetpp_human.yaml` and prepare
human-annotated manifests with `--supervision human`.

## Classification: ACC@NEC5 and AVGACC

Set `RUN` to the config's `training.output`:

```bash
RUN=runs/cub/seed42
python -m spatial_cbm.training export --config "$CONFIG"
python -m spatial_cbm.training.classifier --artifact_dir "$RUN" \
  --output_dir "$RUN/nec" --device cuda --seed 42 --saga_workers 0 \
  --table_device cuda --cache_features_device cuda_preserve_sampler \
  --n_iters 1000 --max_glm_steps 150 --lam_max 0.1 --epsilon 0.00001 --skip_train_eval
python -m spatial_cbm.training classify --config "$CONFIG" --classifier-dir "$RUN/nec"
```

Defaults evaluate NEC values 5, 10, 15, 20, 25, and 30. Results are saved to
`$RUN/nec/classification_test.json`. Change the classifier seed for additional runs.

## Baselines

Choose a config in `configs/release/baselines/`, edit its paths, and use the same
training and classification commands above. VLG-CBM needs no teacher cache.
LF-CBM and SALF-CBM require OpenAI CLIP and a local CLIP checkpoint:

```bash
pip install git+https://github.com/openai/CLIP.git
CONFIG=configs/release/baselines/lf_cbm.yaml
python -m spatial_cbm.training teacher --config "$CONFIG"
python -m spatial_cbm.training train --config "$CONFIG"
```

Teacher preparation also accepts `--resume`. Set `RUN` to the baseline config's
output before fitting its classifier. SALF-CBM is a paper-based reconstruction,
not the authors' training code. The [upstream audit](https://github.com/AtharvRN/GroundedCBM/blob/f6131365/docs/baselines.md)
is preserved in Git history. Use `--help` on any command for additional options.
