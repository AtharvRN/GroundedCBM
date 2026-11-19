# Evaluations

Run from the repository root. Replace the example config and data paths with
your own. Spatial intervention and VLM evaluation require [feature export and NEC fitting](../spatial_cbm/README.md#classification-accnec5-and-avgacc).

## CUB-70

Use human part masks and the saved CUB validation calibration:

```bash
python -m spatial_cbm.training calibrate --config configs/cub.yaml
python -m evals.cub70 --config configs/cub.yaml \
  --mapping evals/cub70_assets/mapping.json --masks-root /data/CUB-70/masks \
  --output results/cub70.json
```

## Spatial Concept Intervention

Paper Experiment V corrects up to five missed, present true-class concepts
(annotated parts on PartImageNet++). Supply the same complete image roster and
mask source across models, including images with no edits. First run validation
calibration with `python -m spatial_cbm.training calibrate --config "$CONFIG"`.

```bash
python -m evals.intervention export --config "$CONFIG" --head-dir "$RUN/nec" \
  --feedback /data/feedback-test.json --split test --output results/intervention/test
python -m evals.intervention evaluate --export results/intervention/test \
  --rule replace --output results/intervention/results.json
```

Feedback JSON contains a source (`gdino`, `sam3`, or `human`), a training-derived
class vocabulary, and the complete cohort. Masks are binary PNGs already aligned
to the model-input crop, not original-resolution images. Paths are relative to
the JSON file; an empty `edits` list keeps the image in the accuracy denominator.

```json
{"source":"gdino","class_concepts":{"0":["yellow wing"]},"images":[{"id":"bird.jpg","edits":[{"concept":"yellow wing","mask":"masks/wing.png","confidence":0.9}]}]}
```

For PartImageNet++ human masks, omit `class_concepts`. GDINO boxes must be unioned
into crop-aligned masks before export. GDINO confidence must exceed 0.15; SAM3
confidence must be at least 0.5. Human masks need no confidence. Rank eligible
edits by confidence, breaking ties by feedback order. Missed concepts are selected
using saved validation-only concept thresholds, never test accuracy.

SpG-CBM replaces the map with `logit(clip(cell_occupancy, .001, .999))` and
re-pools it. SALF-CBM uses `map + beta * support`. Calibrate SALF's beta on a
disjoint validation cohort (200 images in the paper), then apply it unchanged:

```bash
python -m evals.intervention export --config "$CONFIG" --head-dir "$RUN/nec" \
  --feedback /data/feedback-val.json --split validation --output results/intervention/val
python -m evals.intervention calibrate --export results/intervention/val \
  --betas 0 0.5 1 2 4 --output results/intervention/beta.json
python -m evals.intervention evaluate --export results/intervention/test \
  --rule additive --calibration results/intervention/beta.json \
  --output results/intervention/salf.json
```

Prespecify the beta grid; the example is not a claim about historical settings.
Use the original cohort, checkpoints, thresholds and masks for exact paper replay.
The exporter supports release-trained SpG-CBM (LSE pooling) and SALF-CBM; official
external checkpoints need an adapter. Exports hash masks, model/classifier inputs
and saved arrays. Calibration/evaluation run locally on CPU and reject changed
exports or overlapping calibration/test images. These are simulated corrections,
not a user study or proof of causal faithfulness.

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

### VLG-CBM Scene Exports

For ImageNet and Places365, `evals.vlg_scenes` exports concept-targeted Grad-CAM
from a pinned release-trained VLG checkpoint. It differentiates through the
actual concept head, including nonlinear layers, rather than assuming every
model has a linear concept projection.

```bash
python -m evals.vlg_scenes --profiles /data/vlg_judge_profiles.json \
  --dataset places365 --source /data/pinned_training_source \
  --output results/vlg_places_smoke --smoke
```

A study profile pins model/classifier paths and hashes, the matching training
archive and backbone, official validation labels, and existing disjoint
200/5,000-image cohorts. The smoke check exports two images from each cohort
without querying a VLM. For the full export, omit `--smoke`, choose a fresh
output directory, and optionally add `--device cuda` or `--device mps`. Use
`--split validation` to publish the calibration export first, then `--split test`
with the same arguments and output directory. This lets API judging start while
test assets are being prepared. The resulting
`validation/` and `test/` directories use the same `evals.vlm calibrate` and
`evaluate` commands above. Never use smoke exports for reported judge results.

### LF-CBM Exports

`evals.lf_export` exports the release-trained LF-CBM baseline's own top positive
concept contributions and post-hoc concept-targeted Grad-CAM maps. LF-CBM does
not produce native spatial explanations. Maps target the raw concept output,
without multiplying by its class weight. They are checked against autograd.

```bash
PYTHONPATH=/data/pinned_training_source:$PYTHONPATH \
python -m evals.lf_export --profiles /data/lf_profiles.json --dataset cub \
  --device cpu --smoke --output results/lf_cub_smoke
```

Profiles pin the minimum-validation-loss checkpoint, matching training archive,
backbone, concept vocabulary, training normalization, NEC=5 classifier and prior
cohort exports. The inputs reuse verified 224-pixel crops, without cropping a
second time. Run a CPU/accelerator agreement check before omitting `--smoke` for
full exports. CUB uses 200 calibration and 1,000 test images; ImageNet and
Places365 use 200 and 5,000. The output format is compatible with `evals.vlm`;
exporting does not send API requests.

### GroundingDINO Concept Presence

`evals.gdino_presence` scores a fixed, complete concept vocabulary on every image.
It retains the maximum phrase confidence over all detector queries without a
confidence floor, rounded scores, or image-specific concept hints. Phrase
confidence is the mean sigmoid confidence over its tokens. Prompt batches are
fixed across images and checked against the model's token limit.

The prepared study directory contains `inputs/{concepts.txt,roster.json,
provenance.json,split_reference.json,calibration_labels.npy,test_labels.npy,
image_probes.json}`, the pinned `detector/` source, the official
`groundingdino_swinb_cogcoor.pth`, and `staged_hashes.json`. The roster records
filenames, split-local indices and original label-array indices; image probes
verify source JPEGs against the CBM evaluation inputs. The PartImageNet++ study
uses 4,500 fixed validation images and all 10,000 test images, with the same
783-concept bank and human-presence labels as the CBM table.

```bash
python -m evals.gdino_presence infer --root /data/gdino_study \
  --images /data/images --output results/gdino --shards 2 --shard 0
python -m evals.gdino_presence infer --root /data/gdino_study \
  --images /data/images --output results/gdino --shards 2 --shard 1
python -m evals.gdino_presence evaluate --root /data/gdino_study \
  --output results/gdino
```

The two inference commands can run on separate GPUs. Only label-independent
image-backbone features are cached across prompts; cached and ordinary forwards
are compared before accepting a worker's outputs. Inference uses the CBM's
Resize256/CenterCrop224 image content, followed by GroundingDINO's Resize800.
Incomplete or smoke exports cannot produce reported metrics.

F1 thresholds are fitted independently per concept on validation only, choosing
the highest threshold on ties. No validation positives means predict absent;
all validation positives means predict present. AUROC pools all image-concept
pairs; AP and F1 average over concepts with test positives. This is one frozen
pretrained detector, not three independently trained models. Existing
image-specific, confidence-truncated detection caches are not substitutes.

[Protocol details and limitations](PROTOCOL.md) | [CUB-70 mask format](cub70_assets/README.md)
