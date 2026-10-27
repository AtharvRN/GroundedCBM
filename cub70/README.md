# CUB-70 Human-Part Evaluation

This is a separate evaluator, not another trainer. It accepts only CUB-trained
release checkpoints and image IDs from their saved held-out test database.

```bash
python -m spatial_cbm.training calibrate --config configs/release/cub.yaml
python -m cub70 --config configs/release/cub.yaml --mapping cub70/mapping.json \
  --masks-root /data/CUB-70/masks --output results/cub70.json
```

## Inputs

Native mask layout: `<masks-root>/<one-based-class-id>/<image-stem>_<part>.png`.
Parts are body, head, neck, beak, tail, right/left wing, right/left leg, and
right/left eye. The latter six are unioned into wing, leg, and eye. Masks must
match the original image dimensions. The evaluator requires extracted image
files and applies the model's resize/center crop, using nearest-neighbor for
human masks. Only classes 1 through 70 and held-out image IDs are eligible.

Alternatively, supply `--manifest masks.jsonl` with explicit complete masks:

```json
{"id":"001.Black_footed_Albatross/example.jpg","masks":{"head":["masks/head.png"],"wing":["masks/left.png","masks/right.png"],"eye":null,"leg":[]}}
```

Paths are relative to the manifest. Missing keys or null mean unknown. An empty
list explicitly means absent. Missing files are errors, not negative masks.
In native-layout mode a bilateral group with either component missing is unknown
and excluded; it is not scored against an incomplete union. Coverage is reported.
This stricter rule differs from historical evaluations that unioned available
sides despite missing components.

## Mapping And Selection

The bundled mapping preserves the existing Qwen3 exact-part-set majority result:
360 rows, 303 nonempty assignments, eight anatomical groups. It is not generated
or tuned using evaluation images. The original prompt is in `mapping_prompt.md`.
Source: the research tree's `data/cub70_qwen3_majority/mapping.json`.

The mapping must match the trained checkpoint's concept vocabulary. Matching
lowercases, removes the punctuation `- , . ( )`, collapses whitespace, and strips
a leading article. Duplicate canonical bank names and missing mapped concepts
are errors, not silent intersections or arbitrary channel choices. Use the
model's actual saved vocabulary, not a raw candidate concept list.

For each image and part, select the highest **fused concept logit** among only
its mapped concepts. No GT overlap, positivity filter, classifier weight, or
oracle choice is used. Ties follow mapping order. Upsample that native map,
min-max normalize it, and apply the saved CUB validation threshold. Constant
maps are zero. Multi-part concepts may be candidates for multiple parts.

## Results

The JSON records checkpoint/mapping/calibration/input hashes, image coverage,
part coverage, selected concepts, exact per-pair pixel AP, foreground IoU,
pooled foreground IoU, and pixel accuracy. Primary aggregates equally average
supported parts, each of which averages visible image-part pairs. Unsupported
parts remain explicit; a partial result is not a complete eight-part benchmark.
All metric values are fractions, not percentages.

No thresholds or mappings are selected on human test masks. Calibration comes
from the CUB training-pool validation split. Human masks provide anatomical
localization targets; they do not verify colors, textures, or other concept
qualifiers. These results are not interchangeable with earlier pooled-pixel,
fixed-threshold, all-concept-max, or oracle protocols.
