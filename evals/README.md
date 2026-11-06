# Evaluations

Run from the repository root. Replace the example config and data paths with
your own. Deletion and VLM evaluation require [feature export and NEC fitting](../spatial_cbm/README.md#classification-accnec5-and-avgacc).

## CUB-70

Use human part masks and the saved CUB validation calibration:

```bash
python -m spatial_cbm.training calibrate --config configs/cub.yaml
python -m evals.cub70 --config configs/cub.yaml \
  --mapping evals/cub70_assets/mapping.json --masks-root /data/CUB-70/masks \
  --output results/cub70.json
```

## Deletion

```bash
python -m evals.deletion --config configs/cub.yaml \
  --head-dir runs/cub/seed42/nec --output results/deletion/cub
```

Defaults: NEC=5, 32x32 patches, three random controls, full test split.
Add `--count 100` for a quick check. Use a new output directory for each run.

## VLM Judge

Export validation and test examples (release-trained SpG-CBM or SALF-CBM):

```bash
python -m evals.vlm export --config configs/cub.yaml \
  --head-dir runs/cub/seed42/nec --split validation --count 200 --output results/vlm/val
python -m evals.vlm export --config configs/cub.yaml \
  --head-dir runs/cub/seed42/nec --split test --count 100 --output results/vlm/test
```

Set `VLM_API_KEY`, `VLM_BASE_URL` (HTTPS API base URL), `JUDGE_MODEL`, and
`JUDGE_BACKEND` (the exact model ID returned by your provider). Then calibrate
on validation examples and evaluate the test examples:

```bash
python -m evals.vlm calibrate --export results/vlm/val \
  --base-url "$VLM_BASE_URL" --model "$JUDGE_MODEL" --expected-model "$JUDGE_BACKEND" \
  --output results/vlm/calibration
python -m evals.vlm evaluate --export results/vlm/test \
  --calibration results/vlm/calibration/frozen_threshold.json \
  --base-url "$VLM_BASE_URL" --model "$JUDGE_MODEL" --expected-model "$JUDGE_BACKEND" \
  --output results/vlm/evaluation
```

Results are saved as JSON in the specified output locations.
Use `--help` on any command for more options.

[Protocol details and limitations](PROTOCOL.md) | [CUB-70 mask format](cub70_assets/README.md)
