# Evaluation Protocols

These workflows consume frozen release checkpoints; they do not train models or
fit classifiers. Shared detector-based concept/localization evaluation remains
in `spatial_cbm.training`.

| Evaluation | Entry point | Additional inputs |
| --- | --- | --- |
| CUB-70 human parts | `python -m evals.cub70` | Human masks, concept-to-part mapping, saved validation calibration |
| Spatial concept intervention | `python -m evals.intervention` | Native maps, region feedback, validation calibration, saved NEC head |
| VLM-as-a-judge | `python -m evals.vlm` | Saved NEC head, frozen exports, judge endpoint |

## CUB-70

```bash
python -m evals.cub70 --config configs/cub.yaml \
  --mapping evals/cub70_assets/mapping.json --masks-root /data/CUB-70/masks \
  --output results/cub70.json
```

The evaluator is `python -m evals.cub70`, with mapping assets under
`evals/cub70_assets/`. See [mask formats, mapping and aggregation](cub70_assets/README.md).
Selection uses the highest fused score among concepts mapped to each part,
not oracle overlap. Thresholds come from saved CUB training-pool validation
calibration, never the CUB-70 human test masks. Missing masks are unknown, not
background. The primary metrics average visible image-part pairs within each
part, then supported parts equally; retain coverage when reporting them.

## Spatial Concept Intervention

See [commands and feedback format](README.md#spatial-concept-intervention).
The workflow follows Experiment V: select up to five missed, present true-class
concepts (annotated parts on PartImageNet++), edit only their maps and re-pool
with the native pooling function. The global activation, other concepts,
training normalization and classifier remain unchanged. Accuracy includes
every image in the supplied cohort, even when no concept is edited.

Replacement uses the logit of clipped cell occupancy. SALF's additive edit uses
its resized binary support and a validation-selected beta. Calibration and test
cohorts must be disjoint; beta selection never inspects test accuracy. Export
records the actual model, classifier, masks, thresholds, selection order and
input hashes. Saved arrays can be evaluated on CPU without the original image
files, checkpoint or GPU; their hashes are verified before replay.

Use the historical artifacts for exact reported cohorts and calibration. The
release exports new runs and does not bundle paper checkpoints or datasets.
This is simulated concept correction, not proof that a highlighted region
causally mediates the original prediction.

## VLM-as-a-Judge Evaluation

Independent judgments of **concept presence** and **spatial localization** for
CUB, ImageNet and Places365. This is separate from training, detector-based
evaluation, human-mask overlap, and spatial concept intervention.

```bash
python -m evals.vlm --help
```

## 1. Freeze Images, Concepts and Maps

After concept training, feature export and NEC fitting, use the same training
configuration and saved classifier. This exporter supports release-trained
SpG-CBM and SALF-CBM adaptations. It does not load the official authors' SALF
checkpoint; see the external-export contract below.

```bash
python -m evals.vlm export --config configs/cub.yaml \
  --head-dir runs/cub/nec --nec 5 --split validation --count 200 \
  --seed 0 --output output/vlm/cub/calibration

python -m evals.vlm export --config configs/cub.yaml \
  --head-dir runs/cub/nec --nec 5 --split test --count 1000 \
  --seed 0 --output output/vlm/cub/test
```

For ImageNet and Places365, supply the corresponding configuration and use
`--count 5000` for test. Image selection is class-balanced hash ordering before
inference; no correctness, detector-eligibility or judge-result filtering.
Concept selection retains up to five positive contributions to the clean
predicted class, using training-standardized activations and the saved head.
There is no separate positive-activation filter. Images without eligible
concepts remain in the image roster and are explicitly reported.

Exports contain lossless model-input crops, native maps, a hashed pair manifest,
the complete image roster and model/classifier/preprocessing identities. No
model inference or concept selection is repeated during judge runs. The sampler
defines new reproducible cohorts; it does not claim to reconstruct previously
reported experiment rosters. Retain the original exports for exact replay.

## 2. Calibrate Each Judge Separately

Use an HTTPS Chat Completions-compatible provider with image input and strict
JSON-schema output. Set its credential in `VLM_API_KEY` through your normal
secret-management mechanism. No key is stored in the repository or protocol.
Set `VLM_BASE_URL`, `JUDGE_MODEL` and `JUDGE_BACKEND` to your provider's base URL,
requested model ID and exact returned backend ID. A changed backend is rejected.

```bash
python -m evals.vlm calibrate --export output/vlm/cub/calibration \
  --output output/vlm/cub/qwen/calibration \
  --base-url "$VLM_BASE_URL" --model "$JUDGE_MODEL" \
  --expected-model "$JUDGE_BACKEND" --workers 8
```

Presence uses the original crop and concept phrase. Localization uses that crop
plus an image showing the thresholded region in RGB and everything else in
grayscale at 35% brightness. Model identity and classifier outputs are not shown
to the judge. Full prompts are in `vlm.py` and saved in every run protocol.

Native maps are bilinearly resized to the crop dimensions, independently min-max
normalized and thresholded with `>=`. Constant maps produce empty regions.
Calibration tests `.10, .20, ..., .90, .95`, maximizing localization-yes divided
by presence-yes, with the smaller threshold on exact ties. Presence responses
are cached across thresholds. Unsure responses remain in the relevant
denominators; API failures never become unsure judgments.

## 3. Evaluate and Repeat

```bash
python -m evals.vlm evaluate --export output/vlm/cub/test \
  --calibration output/vlm/cub/qwen/calibration/frozen_threshold.json \
  --output output/vlm/cub/qwen/repeat1 \
  --base-url "$VLM_BASE_URL" --model "$JUDGE_MODEL" \
  --expected-model "$JUDGE_BACKEND" --workers 8
```

Run the identical command with fresh `repeat2` and `repeat3` output directories.
Each repeat re-queries presence and then judges localization only for concepts
judged present in that repeat. Maps stay fixed, but localization subsets may
differ. Existing responses are reused only when resuming the same output
directory. Never copy response caches between repeats or judges.

Use Qwen with eight workers or DeepSeek with sixteen for the current study;
worker count is configurable. Each judge needs its own calibration and run
directories. Defaults are temperature zero, low reasoning effort and a 2,048
completion-token cap. Providers must support these settings. Transport has
bounded retries and timeouts; resume the exact command after failed tasks.
Parallelism does not change images, concepts or thresholds.

Evaluation rejects calibration/test image overlap, changed model identity,
changed prompts and changed calibration outputs. Export files and successful
responses are hashed; resumes reject changed protocols. Keep output directories
private if images or image identifiers are subject to dataset restrictions.

```bash
python -m evals.vlm summarize \
  --runs output/vlm/cub/qwen/repeat1 output/vlm/cub/qwen/repeat2 output/vlm/cub/qwen/repeat3 \
  --output output/vlm/cub/qwen/repeats.json
```

## Metrics and Interpretation

- **Concept presence:** presence-yes / all selected concept-image pairs.
- **Conditional localization:** localization-yes / presence-yes.
- **Grounded concept precision:** presence-yes and localization-yes / all pairs.

Rates are fractions, pair-pooled rather than image-macro averages. Repeat
summaries report means, sample standard deviations, unanimous presence agreement
and localization agreement on the common subset judged present in every repeat.
These are judge repeats, not independently trained checkpoints or confidence
intervals over images. Stable repeat averages do not establish human agreement.
For statistical uncertainty use image-cluster resampling. These judgments assess
visible evidence and spatial alignment, not causal reliance or classifier faithfulness.

## External Models and Official SALF-CBM

The judge accepts frozen exports from other models without loading their weights.
An adapter must preserve that model's native crops, concept order, normalization,
classifier, selection rule and maps. Record whether an official ImageNet SALF
export uses the authors' original classifier or a refitted NEC=5 classifier;
those are different explanation-selection protocols and must not be silently
mixed. Retain the original backbone and concept projection in either case.
The public checkpoint exporter above is not an official-checkpoint adapter.

Create `manifest.jsonl` with one row per selected image/concept:

```json
{"id":"image:concept","image_id":"image.jpg","concept":"yellow wing","clean":"images/image/clean.png","native":"images/image/map.npy","clean_sha256":"...","native_sha256":"..."}
```

`clean` is a PNG of the actual model-input crop; `native` is a finite 2D float
NumPy array. Paths must stay within the export directory. Include `export.json`:

```json
{"identity":{"dataset":"imagenet","method":"SALF_RELEASED","checkpoint_sha256":"...","classifier_sha256":"...","preprocessing":"authors native transform","concept_selection":"frozen original positive contributions"},"split":"test","images":1,"image_ids":["image.jpg"],"pairs":1,"manifest_sha256":"..."}
```

The identity dictionary must match exactly between calibration and test exports;
image IDs must not overlap. Populate SHA-256 fields from actual files, and include
all image IDs even when no concept was selected. Keep each adapter's source and
checkpoint provenance with its exports. The interface supports reproducing an
official export, but does not by itself certify that an adapter used official weights.
