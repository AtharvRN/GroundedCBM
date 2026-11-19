# Training and Classification

Run from the repository root. The same commands support CUB, ImageNet,
Places365, and PartImageNet++; select the matching config in `configs/`.

## Prepare Data

Start with the [image and annotation downloads](../README.md#get-the-data).
For CUB, extract the image archive and the detector annotations separately.
Create one JSON index for the official training images and one for the test
images. Each key is the image path from CUB's `images.txt` (relative to
`images/`); its value points to the detector JSON, relative to the index.
For example, **if** that JSON's header contains the following filename and its
boxes use pixel coordinates, an index entry looks like this:

```json
{
  "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg": {
    "annotation": "cub_train/0.json",
    "annotation_image_id": "Black_Footed_Albatross_0046_18.jpg",
    "box_coordinates": "pixel_xyxy"
  }
}
```

Match by image identity, not by sorting numbered JSON files. Use the exact
`file_name` or `image_id` from the detector header. If an archive lacks image
identities, recover them from the generator's original image order or
regenerate those annotations. Use `normalized_xyxy` instead for boxes in
the range 0 to 1.

```bash
CUB=/data/CUB_200_2011
for split in train test; do
  python -m spatial_cbm.training manifest --layout cub --root "$CUB" \
    --partition "$split" --annotation-map "/data/annotations/cub-$split.json" \
    --output "manifests/cub-$split.jsonl"
done
```

Alternatively, supply two JSONL manifests directly. Each row contains an image
ID, image path, zero-based class ID, and detections:

```json
{"id":"bird/001.jpg","image":"images/bird/001.jpg","class_id":0,"detections":[{"concept":"black bill","confidence":0.91,"box":[0.2,0.1,0.4,0.3],"coordinates":"normalized_xyxy"}]}
```

Image paths are relative to the manifest. Concept names must match the vocabulary.
An empty detection list means a verified negative, not missing annotations.
Use `python -m spatial_cbm.training manifest --help` for native-layout adapters.

```bash
python -m spatial_cbm.training prepare --dataset cub \
  --source manifests/cub-train.jsonl --test-source manifests/cub-test.jsonl \
  --concepts concepts/cub_filtered.txt --output prepared/cub --seed 6885
```

Preparation saves an internal validation split. Reuse the same prepared directory
across methods and seeds; never use test images for validation.

## Train and Evaluate

Set the prepared-data, backbone checkpoint, node-local cache, and output paths
in the config first. Use a separate output directory per method and seed.

```bash
CONFIG=configs/cub.yaml
python -m spatial_cbm.training check --config "$CONFIG"
python -m spatial_cbm.training train --config "$CONFIG"
python -m spatial_cbm.training calibrate --config "$CONFIG"
python -m spatial_cbm.training evaluate --config "$CONFIG"
```

Add `--resume` to `train` after an interruption. Minimum validation loss selects
`best.pt`; calibration uses validation only. Test results are in `test_metrics.json`.
For PartImageNet++ human supervision, use `partimagenetpp_human.yaml` and prepare
human-annotated manifests with `--supervision human`.

Concept metrics following Table 3's definitions are `concept_ap` and
`concept_macro.{precision,recall,f1}` in `test_metrics.json`. AP and macro scores
average concepts with test positives. Each concept threshold maximizes its own
validation F1 (highest threshold on ties). Concepts with no validation positives
always predict absent; all-positive concepts always predict present. The
`concept_macro.micro` field reports pooled scores at those same thresholds;
`concept` retains the separate, single-threshold micro-F1 protocol for comparison.
Archive legacy micro-only calibration/results JSON before recalibrating; the
new evaluator will not silently relabel old results as macro scores.
Metrics use the prepared annotations: GDINO targets give proxy scores, not
independent human-label results. The human-supervision config is a training
ablation, not a switch that reproduces Table 3's detector-trained comparison.

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

Choose a config in `configs/baselines/`, edit its paths, and use the same
training and classification commands above. VLG-CBM needs no teacher cache.
LF-CBM and SALF-CBM require OpenAI CLIP and a local CLIP checkpoint:

```bash
pip install git+https://github.com/openai/CLIP.git
CONFIG=configs/baselines/lf_cbm.yaml
python -m spatial_cbm.training teacher --config "$CONFIG"
python -m spatial_cbm.training train --config "$CONFIG"
```

Teacher preparation also accepts `--resume`. Set `RUN` to the baseline config's
output before fitting its classifier. SALF-CBM training is a reconstruction
of the published method, not the original training implementation.

## Paper Settings

Use the same prepared splits and exact vocabulary across methods. The supplied
concept files are source vocabularies, not a guarantee of the final paper's
filtered banks (671 CUB, 4,306 ImageNet, 2,534 Places365, 783 PartImageNet++).
Keep the prepared `concepts.txt` and split hashes with each reported model.
Disable baseline vocabulary filtering with `training.filter_concepts: false`
for a matched comparison.

The current paper uses residual fusion `g + 0.1s` except on PartImageNet++, where
`model.fusion: convex` and `model.alpha: 0.75` mean `0.75g + 0.25s`.
The dataset templates now reflect the paper's stated training budgets. Places365
uses the archived 10-epoch AdamW/cosine KL+BCE recipe; PartImageNet++ uses the
archived convex recipe. ImageNet's template uses the manuscript's 20-epoch
budget, not the longer histories of some existing checkpoints. Do not resume an
old run with a changed recipe. CUB defaults to a 10% internal holdout (599 images);
pass `--fraction 0.05` to `prepare` for the other three datasets. Reuse the exact
saved split when comparing to an existing result.
ImageNet's main SALF-CBM result uses the authors' released checkpoint and stronger
backbone, not this repository's reconstructed SALF training recipe. Do not
replace its backbone at inference or label a reconstructed run as official.
Recipe templates do not include historical data, weights or exact cohorts.

## Optional ViT and Multiple GPUs

Install `pip install -r requirements-vit.txt`. For a frozen CLIP ViT-B/16 set
`model.backbone: clip_vit_b16`, `model.stages: vit4812`, `model.weights: none`
and an explicit local `model.checkpoint` (OpenCLIP full/visual-only weights).
Use input size 224 and the native 14x14 grid. The global branch reads the
512-dimensional image embedding; spatial projections read patch tokens after
blocks 4, 8 and 12. CLIP normalization and bicubic resizing are automatic.
This option supports SpG-CBM only, not backbone fine-tuning or baseline heads.

For multi-GPU SpG concept training, set `training.device: cuda` and
`training.devices: 2`. `training.batch_size` is the total batch, not per GPU;
losses are computed after gathering. Checkpoint keys remain single-device
compatible. LF/SALF validation uses streaming full-cohort moments, without
materializing all spatial predictions on disk.
