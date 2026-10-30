# Baseline Fidelity and Upstream Audit

The release implements VLG-CBM, LF-CBM, and a corrected SALF-CBM reconstruction in the shared training workflow; this is not a claim of reproduced paper accuracy. Baseline-specific heads, targets, and losses are independent equation implementations; this integration does not vendor upstream training code. Existing shared GLM-SAGA fitting retains its separate provenance.

## Pinned Sources

| Method | Official upstream | Audited commit |
| --- | --- | --- |
| VLG-CBM | [Trustworthy-ML-Lab/VLG-CBM][vlg] | `84031a293a7153ad89a739ef2096b9335ba66790` |
| LF-CBM | [Trustworthy-ML-Lab/Label-free-CBM][lf] | `fa2869d46c454f406af69b19d3d5427f2d427492` |
| SALF-CBM | [itaybenou/show-and-tell][salf] | `e45f8645afde37a8a7ae66db6987b5a8a53552a0` |

## Method Fidelity

| Method | Head and supervision contract | Pooling and limitations |
| --- | --- | --- |
| VLG | Biased `Linear(D,C)`, then `vlg_hidden` repetitions of `ReLU -> Linear(C,C)`; no output sigmoid. Binary detection-presence targets; mean BCE-with-logits with one fixed scalar `positive_weight` shared across concepts. Upstream automatic classwise positive weighting is not available. | Global average backbone features. Nonlinear global heads support classification but concept maps are explicitly unavailable; do not substitute a spatial application of the MLP. |
| LF | Frozen backbone and bias-free `Linear(D,C)`. CLIP image/text cosine targets. Center each concept across images, cube values elementwise, then compute cosine across images; average negative similarities over concepts. | Global average backbone features; no target softmax or sigmoid. Training strings/order and filtering must not inherit VLG annotation canonicalization. |
| SALF | Frozen spatial backbone; bilinear feature resize to `salf_grid`, then bias-free `1x1 Conv`. Prompted CLIP cosine targets; centered-cubic cosine across images independently for every concept/cell. | Spatial softmax-weighted average of raw maps, followed by feature standardization. No averaging of backbone features before the concept head. |

VLG source: [`model/cbm.py:85-200`][vlg-head], [`loss.py:35-51`][vlg-loss]. LF source: [`similarity.py:5-23`][lf-loss], [`train_cbm.py:114-173`][lf-train]. SALF released inference: [`cbm.py:8-83`][salf-model]; training equations: [paper sections 3.2-3.4][salf-paper].

## Targets and Teacher Cache

- VLG keeps annotations with confidence strictly greater than the configured threshold (upstream `.15`) and concepts with at least one positive training image. Cropping chooses uniformly among present vocabulary concepts after selecting one random eligible box per concept. Crop the original image to that box, retain the selected concept plus other concepts with box IoU strictly greater than `.5`, then apply preprocessing. No eligible concepts means full image/all-zero labels; upstream exceptions fall back to full-image labels. CBL optimization alone uses crops; filtering, validation, and classifier export do not. Published CUB/Places/ImageNet crop probability is `.5`, unlike CLI default `0`. See [`data/concept_dataset.py:48-146,253-294`][vlg-data].
- Normalize both VLG annotation names and vocabulary lookup keys: lowercase, replace `- , . ( )` with spaces, remove a leading `a ` or `an `, and collapse whitespace. Use NumPy random draws for crop decisions and concept/box selection, as upstream does; preserve NumPy RNG state and worker seeding for reproducibility. This normalization does not apply to LF/SALF literal teacher strings.
- LF teacher is CLIP `ViT-B/16`, distinct from the student. Encode literal concept strings, L2-normalize both embeddings, and use their dot product without logit scale. Teacher preprocessing is shorter-side bicubic Resize224, CenterCrop224, RGB, tensor, CLIP normalization: mean `(0.48145466,0.4578275,0.40821073)`, std `(0.26862954,0.26130258,0.27577711)`. Student preprocessing is independent. See [`utils.py:93-114`][lf-targets] and [`clip/clip.py:79-86`][lf-transform].
- LF retains mean top-five training-image cosine strictly above `lf_clip_cutoff`: upstream default `.25`, CUB `.26`, Places/ImageNet `.28`. After selecting the head, retain per-concept validation centered-cubic similarity strictly above `lf_interpretability_cutoff` (upstream `.45`). See [`train_cbm.py:91-173`][lf-train] and [`training_commands.txt:4-17`][lf-commands].
- SALF paper specifies ViT-B/16, literal concept text, interior grid centers with integer strides `H//(gh+1), W//(gw+1)`, and two-pixel red circle outlines. Reported classification uses grid `7x7`, radius `32`; spatial top-500/quantile filtering is not verified author behavior. Shared Resize256/CenterCrop224 teacher/student geometry is a reconstruction choice, not recovered author training preprocessing. See [paper section 4.1 and Appendix A][salf-paper].

`teacher.prompt_grid` selects `interior` (default, preserving the previous placement)
or `edge`. For a 224-pixel image and seven positions, `edge` uses integer stride
`224//(7-1)=37`, starting at zero: `0,37,74,111,148,185,222`. Coordinates are
clamped to the final valid pixel when an integer stride lands exactly at the image
size; circles crossing the boundary are clipped by PIL. This is the explicitly
chosen edge-origin interpretation of the appendix denominator, not a literal
implementation of its inconsistent loop bounds. Layout and exact centers are
recorded in the teacher identity; changing them requires regenerating targets.
The head remains a bias-free 1x1 convolution and the loss remains centered-cubic
cosine across images separately for each concept and spatial cell.

Configuration uses `Model.method`, `Model.salf_grid`, `Model.vlg_hidden`; `Teacher(cache, model, checkpoint, batch_size, prompt_radius)`; and `Training.vlg_crop_probability`, `Training.lf_clip_cutoff`, `Training.lf_interpretability_cutoff`. Baseline recipes are provided under `configs/release/baselines/`. Set their prepared-data, backbone-checkpoint, teacher-checkpoint, cache, and output paths before running:

For fixed-vocabulary controls, set `training.filter_concepts: false`. This retains
the complete supplied vocabulary in its original order: LF skips both the
training CLIP support filter and the post-training interpretability filter; VLG
skips its vocabulary support filter, without changing detection-label confidence
thresholding. The default remains `true`. SG/SALF have no additional head-level
vocabulary filter. Record this as a fixed-vocabulary variant, not the unchanged
native LF/VLG pipeline. The flag is part of checkpoint/configuration provenance.

```sh
python -m spatial_cbm.training teacher --config configs/release/baselines/lf_cbm.yaml
python -m spatial_cbm.training train --config configs/release/baselines/lf_cbm.yaml
python -m spatial_cbm.training teacher --config configs/release/baselines/salf_cbm.yaml
python -m spatial_cbm.training train --config configs/release/baselines/salf_cbm.yaml
python -m spatial_cbm.training train --config configs/release/baselines/vlg_cbm.yaml
```

Install official OpenAI CLIP separately and supply an explicit local teacher checkpoint; a model name is not permission to download or substitute weights. Teacher preprocessing/cache identity is strict: ordered samples, literal vocabulary, checkpoint hash, method, resize/crop/interpolation, normalization, prompt geometry, and precision must match. Reject incompatible or partial caches rather than silently reuse them. Encode only training plus internal validation, excluding test. LF uses official CLIP preprocessing on original RGB; the SALF reconstruction uses bilinear Resize256/CenterCrop224, then red prompts and CLIP normalization only. VLG uses annotations and does not run the teacher command.

## Deliberate Protocol Differences

- The release uses a saved internal validation split disjoint from test, epoch-based optimization, minimum-validation-loss checkpoints, and configured patience. Original LF uses up to 1000 randomly sampled projection steps (batch up to 50000), evaluates every 50 steps and at the final step, stops at the first non-improvement including ties, and restores its best weights. Its `dataset_val` is the official evaluation split, including CUB's test set, used for projection selection and interpretability filtering. The release does not reproduce that selection protocol. See [`train_cbm.py:48-49,119-160`][lf-train] and [`data_utils.py:8-12`][lf-splits].
- LF/SALF centered-cubic training requires at least three images per batch and at least three training images. Merge a final tail of one or two images into the preceding batch; do not drop it or train on it separately. A singleton has zero centered signal, and a centered pair has degenerate cosine geometry with no useful regression gradient. This is a release safeguard, not an upstream batching claim.
- `RegressionMeter` stores predictions and targets in disk-backed float32 arrays, then reads concept chunks spanning the complete validation cohort and computes centered-cubic similarities in float64. LF retains one similarity per concept; SALF retains one per concept/cell. This is full-cohort computation, not a two-pass streaming accumulator or an average of independently centered minibatch losses. Ordinary first/second-moment regression statistics are insufficient.
- The shared cubic similarity centers along the image axis, L2-normalizes each centered vector with `eps=1e-12` before cubing, then L2-normalizes the cubic vector with `eps=1e-8` before taking the dot product. Pre-cube normalization is a positive rescaling that preserves nondegenerate cosine while limiting underflow; epsilon floors are deliberate numerical safeguards. A zero-variance vector yields similarity zero. Training, full-cohort validation, and LF interpretability filtering use this policy.
- The old research `methods/lf.py:118-121` cubes an uncentered cosine; training/validation at lines 198/210 also transpose `[N,C]`, reducing over concepts instead of images. It is not the original LF objective. The research `lf_original_protocol` flag does not fix that route. The release must use the centered, elementwise-cubed objective above for training, validation, and filtering.
- Corrected SALF behavior comes from research `methods/salf_paper.py:19-70`, used by `scripts/train_places365_salf.py`, not the old `methods.salf.train_salf_cbm` dispatch in `methods/registry.py:12-15`. The release and corrected helper average over concepts/cells; paper Eq. 1 sums them. This scales loss/gradients by `1/(C*H*W)` and is an explicit deviation, not exact optimizer parity.
- VLG upstream intends best-validation selection but stores live `state_dict()` references at `model/cbm.py:475-476`. Release checkpoints must clone or serialize immutable model state, fixing this aliasing bug rather than reproducing it. Upstream averages validation batch means equally and trains all configured epochs; release selection/patience is a documented change. See [`model/cbm.py:348-496`][vlg-selection].
- SALF upstream publishes inference and demos, not the exact author training pipeline ([README:1-4][salf-readme]). Optimizer, batching, checkpoint rules, filtering implementation, and exact training transforms remain reconstruction choices. Released inference uses Hugging Face `microsoft/resnet-50`, Resize224/CenterCrop224, and default map size `12x12`; do not conflate that checkpoint profile with the paper's `7x7` training recipe. See [`data_utils.py:10-36`][salf-preprocess] and [`cbm.py:40-83`][salf-model].

## Backbone and Classifier Scope

### ImageNet Controlled Launch (2026-10-30)

The `imagenet_*` recipes reuse the saved SpG-CBM cohort: 1,153,050 fitting
images, 10,000 internal-validation images, and 4,306 ordered concepts. The
50,000-image evaluation split is separate. This is the existing reduced
training cohort, not the complete 1,281,167-image official training pool.
Additional vocabulary filters are disabled for both methods.

VLG uses its released ImageNet `args.txt` settings: seven epochs, batch 64,
SGD with momentum 0.9, LR 0.01, weight decay 1e-5, positive weight 8000,
crop probability 0.5, confidence >0.15, and no hidden layers. This positive
weight is intentionally different from the SpG-CBM value of 100.

LF uses Adam, LR 0.001, zero weight decay, batch 50,000, literal ViT-B/16
teacher scores, and frozen pooled ResNet-50 features. Its controlled cap is
75 epochs with patience 10 after at least 10 epochs, not 1,000 upstream
random projection updates. There are 24 projection updates per full epoch
on this cohort. Both methods select minimum internal-validation loss,
retain the earliest tie, and use FP32 with a frozen ResNet-50 V1 backbone.
Validation loss and annotation-based concept/spatial AP run every epoch.
Post-training thresholds are fitted only on internal validation.

The local release suite passed 243 tests before adding these recipes. Full
ImageNet baseline execution is not yet validated: launch-time tests and
real-image/GPU-memory probes must pass before full optimization. The saved
prepared input has no held-out GDINO annotations; training/calibration and
feature export do not constitute held-out concept/localization evaluation.

Release students are torchvision-compatible ResNet-18/50/101 with explicit weights or compatible checkpoints; frozen during concept fitting. Upstream CLIP student backbones, ViTs, pytorchcv CUB checkpoints, and Hugging Face SALF checkpoints are not automatically supported or equivalent. CLIP as a teacher does not imply CLIP student support.

Export raw concept scores (SALF after pooling), fit training-only column means/sample standard deviations, and reuse those statistics on validation/test. Record any standard-deviation floor. Reuse the existing shared NEC fitting and selection, not a new baseline classifier loop. Original LF/VLG fixed-lambda fitting zero-initializes a biased classifier and calls GLM-SAGA with alpha `.99`, step `.1`, `k=1`, `epsilon=1`, taking `path[0]`; the solver divides supplied `max_reg` by alpha. Shared NEC evaluation is not reproduction of that full fixed-lambda pipeline. See [`model/cbm.py:532-568`][vlg-classifier] and [`glm_saga/elasticnet.py:528-559`][vlg-solver].

The baseline workflow test suite covers CUB, ImageNet, Places365, and PartImageNet++ using real ResNet execution and GLM-SAGA fitting with mock teacher targets. Its 16 passing workflow tests validate integration, not real CLIP teacher quality, full-data training, or published accuracy.

## Attribution and Licensing

Cite VLG-CBM (Srivastava, Yan, Weng; NeurIPS 2024), LF-CBM (Oikarinen, Das, Nguyen, Weng; ICLR 2023), SALF-CBM (Benou, Riklin-Raviv; CVPR 2025), and the sparse-layer/GLM-SAGA work where used. No repository-wide license was found in the pinned VLG/LF trees; citation is not permission to redistribute their source or modifications. Independent baseline equations do not resolve the provenance of already inherited code. Review that separately before release; [public visibility is not a reuse license][license-guidance].

SALF is [MIT, copyright 2024 Itay Benou][salf-license]. Preserve applicable notices for any reused components, including [GLM-SAGA, 2021 Eric Wong][glm-license] and [CLIP, 2021 OpenAI][clip-license]. Do not apply those component licenses to all VLG/LF code. Check checkpoint and dataset terms separately.

[vlg]: https://github.com/Trustworthy-ML-Lab/VLG-CBM/tree/84031a293a7153ad89a739ef2096b9335ba66790
[lf]: https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/fa2869d46c454f406af69b19d3d5427f2d427492
[salf]: https://github.com/itaybenou/show-and-tell/tree/e45f8645afde37a8a7ae66db6987b5a8a53552a0
[vlg-head]: https://github.com/Trustworthy-ML-Lab/VLG-CBM/blob/84031a293a7153ad89a739ef2096b9335ba66790/model/cbm.py#L85-L200
[vlg-loss]: https://github.com/Trustworthy-ML-Lab/VLG-CBM/blob/84031a293a7153ad89a739ef2096b9335ba66790/loss.py#L35-L51
[vlg-data]: https://github.com/Trustworthy-ML-Lab/VLG-CBM/blob/84031a293a7153ad89a739ef2096b9335ba66790/data/concept_dataset.py#L48-L294
[vlg-selection]: https://github.com/Trustworthy-ML-Lab/VLG-CBM/blob/84031a293a7153ad89a739ef2096b9335ba66790/model/cbm.py#L348-L496
[vlg-classifier]: https://github.com/Trustworthy-ML-Lab/VLG-CBM/blob/84031a293a7153ad89a739ef2096b9335ba66790/model/cbm.py#L532-L568
[vlg-solver]: https://github.com/Trustworthy-ML-Lab/VLG-CBM/blob/84031a293a7153ad89a739ef2096b9335ba66790/glm_saga/elasticnet.py#L528-L559
[lf-loss]: https://github.com/Trustworthy-ML-Lab/Label-free-CBM/blob/fa2869d46c454f406af69b19d3d5427f2d427492/similarity.py#L5-L23
[lf-train]: https://github.com/Trustworthy-ML-Lab/Label-free-CBM/blob/fa2869d46c454f406af69b19d3d5427f2d427492/train_cbm.py#L48-L173
[lf-targets]: https://github.com/Trustworthy-ML-Lab/Label-free-CBM/blob/fa2869d46c454f406af69b19d3d5427f2d427492/utils.py#L93-L114
[lf-transform]: https://github.com/Trustworthy-ML-Lab/Label-free-CBM/blob/fa2869d46c454f406af69b19d3d5427f2d427492/clip/clip.py#L79-L86
[lf-commands]: https://github.com/Trustworthy-ML-Lab/Label-free-CBM/blob/fa2869d46c454f406af69b19d3d5427f2d427492/training_commands.txt#L4-L17
[lf-splits]: https://github.com/Trustworthy-ML-Lab/Label-free-CBM/blob/fa2869d46c454f406af69b19d3d5427f2d427492/data_utils.py#L8-L12
[salf-model]: https://github.com/itaybenou/show-and-tell/blob/e45f8645afde37a8a7ae66db6987b5a8a53552a0/cbm.py#L8-L83
[salf-preprocess]: https://github.com/itaybenou/show-and-tell/blob/e45f8645afde37a8a7ae66db6987b5a8a53552a0/data_utils.py#L10-L36
[salf-readme]: https://github.com/itaybenou/show-and-tell/blob/e45f8645afde37a8a7ae66db6987b5a8a53552a0/README.md#L1-L4
[salf-paper]: https://arxiv.org/html/2502.20134v1
[salf-license]: https://github.com/itaybenou/show-and-tell/blob/e45f8645afde37a8a7ae66db6987b5a8a53552a0/LICENSE#L1-L21
[glm-license]: https://github.com/MadryLab/glm_saga/blob/4016cfe8f529bb67ebf1ff40135cf9e0181a77e8/LICENSE
[clip-license]: https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE
[license-guidance]: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository
