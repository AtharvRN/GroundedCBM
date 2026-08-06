# CUB SG-CBM and SALF-CBM Seed Results

Date: 2026-07-25

This note records the corrected CUB seed results for SG-CBM and SALF-CBM. For localization, `Pointing Acc.` means the paper definition: whether the single strongest activation falls inside the annotated target region. This is reported as `point_hit` in the eval JSONs.

## Paper-Facing Summary

Seeds:

```text
42, 123, 1, 6885, 0
```

Sparse classification:

| Method | Test Acc | ANEC-5 | ANEC-avg |
|---|---:|---:|---:|
| SG-CBM | 0.760933 +/- 0.001842 | 0.760069 +/- 0.002707 | 0.760518 +/- 0.001953 |
| SALF-CBM | 0.720829 +/- 0.001572 | 0.621278 +/- 0.004534 | 0.691957 +/- 0.000691 |

GDINO box localization. Metric mapping: `RMA = mass_in_gt`, `Pointing Acc. = point_hit`, `mIoU = threshold_metrics[thr].mean_iou`.

| Method | RMA | Pointing Acc. | mIoU @ 0.7 | Best mIoU |
|---|---:|---:|---:|---:|
| SG-CBM | 0.354313 +/- 0.071639 | 0.641586 +/- 0.035413 | 0.254169 +/- 0.021835 | 0.300599 +/- 0.016185 |
| SALF-CBM | 0.161187 +/- 0.000041 | 0.277087 +/- 0.002250 | 0.192497 +/- 0.001528 | 0.192497 +/- 0.001528 |

CUB part localization, concept-oracle mode. Here `Pointing Acc. @ 0.1` is the strongest activation within `0.1 * image_diagonal` of the annotated part point. `Point-in-mask` is a separate thresholded-mask metric and should not be used as paper pointing accuracy.

| Method | Pointing Acc. @ 0.1 | Best Point-in-Mask | Best Mask IoU | Mask IoU @ 0.7 |
|---|---:|---:|---:|---:|
| SG-CBM | 0.995934 +/- 0.000781 | 0.999202 +/- 0.000000 | 0.275859 +/- 0.020187 | 0.096641 +/- 0.017015 |
| SALF-CBM | 0.909971 +/- 0.001416 | 0.999202 +/- 0.000000 | 0.191791 +/- 0.000639 | 0.073719 +/- 0.000228 |

Concept prediction against GDINO-labeled concept presence:

| Method | AUROC | AUPRC | Macro AP | P@5 | Best F1 |
|---|---:|---:|---:|---:|---:|
| SG-CBM | 0.961425 +/- 0.003227 | 0.591765 +/- 0.027258 | 0.543934 +/- 0.012534 | 0.618183 +/- 0.020192 | 0.595931 +/- 0.020474 |
| SALF-CBM | 0.537274 +/- 0.016274 | 0.010401 +/- 0.000509 | 0.017878 +/- 0.000474 | 0.016014 +/- 0.001046 | 0.021257 +/- 0.001309 |

CUB70 part segmentation localization, corrected mask protocol from
`/workspace/SAVLGCBM/scripts/eval_cub70_localization.py`. Metric mapping:
`RMA = concept_region_metrics.mass_in_gt`, `Pointing Acc. = concept_region_metrics.point_hit`,
`mIoU@mean = concept_region_metrics.miou_at_mean`, `MaskIoU@0.5 = concept_region_metrics.mask_iou_at_0p5`,
and `oracle mAP@0.5 = oracle_mAP.overall["0.5"]`.

| Method | Selection | RMA | Pointing Acc. | mIoU@mean | MaskIoU@0.5 | Best IoU | Oracle mAP@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| SG-CBM | activation | 0.108583 +/- 0.028250 | 0.243622 +/- 0.028659 | 0.065351 +/- 0.002192 | 0.069428 +/- 0.006219 | 0.131360 +/- 0.011553 | - |
| SG-CBM | concept oracle | 0.112324 +/- 0.025654 | 0.428018 +/- 0.019496 | 0.070930 +/- 0.001327 | 0.076568 +/- 0.008109 | 0.301400 +/- 0.017282 | 0.129655 +/- 0.031342 |
| SALF-CBM | activation | 0.036388 +/- 0.000063 | 0.110630 +/- 0.006381 | 0.056771 +/- 0.001603 | 0.062201 +/- 0.002466 | 0.079220 +/- 0.004281 | - |
| SALF-CBM | concept oracle | 0.036870 +/- 0.000023 | 0.295824 +/- 0.003882 | 0.071458 +/- 0.000114 | 0.086131 +/- 0.000253 | 0.244020 +/- 0.001506 | 0.068580 +/- 0.005098 |

Per-seed CUB70 concept-oracle rows:

| Method | Seed | RMA | Pointing Acc. | mIoU@mean | MaskIoU@0.5 | Best IoU | Oracle mAP@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SG-CBM | 42 | 0.157956 | 0.462794 | 0.073261 | 0.091017 | 0.332200 | 0.185073 |
| SG-CBM | 123 | 0.099484 | 0.418565 | 0.070591 | 0.073687 | 0.295300 | 0.113629 |
| SG-CBM | 1 | 0.104970 | 0.421507 | 0.070193 | 0.072088 | 0.291500 | 0.110439 |
| SG-CBM | 6885 | 0.101562 | 0.419704 | 0.070014 | 0.072411 | 0.293200 | 0.115799 |
| SG-CBM | 0 | 0.097649 | 0.417521 | 0.070589 | 0.073635 | 0.294800 | 0.123333 |
| SALF-CBM | 42 | 0.036838 | 0.296128 | 0.071336 | 0.085947 | 0.245500 | 0.069513 |
| SALF-CBM | 123 | 0.036883 | 0.289294 | 0.071401 | 0.085977 | 0.242400 | 0.071430 |
| SALF-CBM | 1 | 0.036881 | 0.297836 | 0.071640 | 0.086342 | 0.245100 | 0.068280 |
| SALF-CBM | 6885 | 0.036854 | 0.299450 | 0.071478 | 0.086465 | 0.242400 | 0.060174 |
| SALF-CBM | 0 | 0.036892 | 0.296412 | 0.071433 | 0.085927 | 0.244700 | 0.073501 |

## SG-CBM Configuration

The SG-CBM CUB rerun used this paper-style configuration:

```text
dataset=cub
backbone=resnet18_cub
feature_layer=layer4
model_name=savlg_cbm
savlg_branch_arch=dual
savlg_global_head_mode=vlg_linear
savlg_freeze_global_head=False
savlg_residual_spatial_alpha=0.1
loss_mask_w=0.25
loss_global_spatial_align_w=0
grid_h=14
grid_w=14
cbl_batch_size=32
cbl_epochs=75
cbl_early_stop_patience=8
cbl_min_epochs=15
cbl_min_delta=0.001
saga_lam=0.0002
```

Operational defaults used with this config:

```text
cbl_lr=5e-4
cbl_optimizer=adam
saga_batch_size=512
saga_n_iters=4000
saga_step_size=0.1
savlg_spatial_branch_mode=multiscale_conv45
savlg_spatial_stage=conv5
savlg_target_mode=soft_box
mask_h=14
mask_w=14
annotation_dir=/workspace/SAVLGCBM/annotations
concept_set=concept_files/cub_filtered.txt
savlg_residual_spatial_pooling=lse
```

## Run Locations

Corrected SG-CBM run root:

```text
/workspace/sgcbm_paper_config_seed_runs
```

SALF-CBM run root:

```text
/workspace/salf_paper_config_seed_runs
```

Completed seeds:

```text
42, 123, 1, 6885, 0
```

Each seed has:

- trained SG-CBM checkpoint
- sparse NEC metrics
- GDINO concept prediction metrics
- GDINO box localization metrics
- CUB part localization metrics, activation-based
- CUB part localization metrics, concept-oracle mode

## Sparse Classification

| Metric | Mean | Std |
|---|---:|---:|
| Test acc | 0.760933 | 0.001842 |
| ANEC-5 | 0.760069 | 0.002707 |
| ANEC-avg | 0.760518 | 0.001953 |

Paper target for SG-CBM CUB RN18: ANEC-5 `0.7617`, ANEC-avg `0.7622`.

Per seed:

| Seed | Test acc | ANEC-5 | ANEC-avg |
|---:|---:|---:|---:|
| 42 | 0.763558 | 0.762867 | 0.762953 |
| 123 | 0.759931 | 0.762349 | 0.761572 |
| 1 | 0.759585 | 0.759240 | 0.759470 |
| 6885 | 0.762176 | 0.756131 | 0.757858 |
| 0 | 0.759413 | 0.759758 | 0.760737 |

## Concept Prediction

Ground truth: GDINO-labeled CUB concept presence.

| Metric | Mean | Std |
|---|---:|---:|
| AUROC | 0.961425 | 0.003227 |
| AUPRC | 0.591765 | 0.027258 |
| Macro AP | 0.543934 | 0.012534 |
| P@5 | 0.618183 | 0.020192 |
| Best F1 | 0.595931 | 0.020474 |

Per seed:

| Seed | AUROC | AUPRC | Macro AP | P@5 | Best F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.966105 | 0.632725 | 0.562903 | 0.647565 | 0.626995 |
| 123 | 0.958525 | 0.565377 | 0.533160 | 0.597720 | 0.577062 |
| 1 | 0.962070 | 0.595614 | 0.545291 | 0.622314 | 0.597820 |
| 6885 | 0.962211 | 0.596933 | 0.546406 | 0.622763 | 0.600136 |
| 0 | 0.958215 | 0.568177 | 0.531911 | 0.600553 | 0.577643 |

## GDINO Box Localization

These metrics use GDINO boxes as the spatial target.

Paper-report field mapping for this repo's `scripts/eval_gdino_localization.py` output:

```text
RMA           = metrics.distribution_metrics.mass_in_gt
Pointing Acc. = metrics.distribution_metrics.point_hit
mIoU          = metrics.threshold_metrics[thr].mean_iou
Mask IoU@0.5  = metrics.distribution_metrics.mask_iou_at_0p5
```

Important naming caveat: this script's `metrics.best_mean_iou.value` is not the paper mIoU field. In the current implementation it stores the best thresholded `mask_iou`. For paper mIoU, use `threshold_metrics[thr].mean_iou`, either at the table's fixed threshold or best over thresholds.

| Metric | Mean | Std |
|---|---:|---:|
| RMA / mass-in-GT | 0.354313 | 0.071639 |
| Pointing | 0.641586 | 0.035413 |
| Soft IoU | 0.209918 | 0.033562 |
| Mask IoU at 0.5 | 0.197568 | 0.012236 |
| mIoU at activation threshold 0.7 | 0.254169 | 0.021835 |
| Best mIoU over thresholds | 0.300599 | 0.016185 |
| Best mask IoU over thresholds | 0.279246 | 0.024676 |
| Best box acc @ 0.5 | 0.204522 | 0.024403 |

Per seed:

| Seed | RMA | Pointing | mIoU @ 0.7 | Best mIoU | Best mIoU thr | Mask IoU @ 0.5 | Best mask IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.482330 | 0.704893 | 0.293174 | 0.329522 | 0.9 | 0.219420 | 0.323347 |
| 123 | 0.321951 | 0.626092 | 0.245496 | 0.294155 | 0.9 | 0.193069 | 0.269543 |
| 1 | 0.325011 | 0.627244 | 0.242557 | 0.293303 | 0.9 | 0.191138 | 0.266589 |
| 6885 | 0.325181 | 0.625991 | 0.244433 | 0.293809 | 0.9 | 0.191848 | 0.268461 |
| 0 | 0.317091 | 0.623709 | 0.245185 | 0.292206 | 0.9 | 0.192364 | 0.268289 |

Important caveat: RMA is not stable across seeds. Seed `42` is close to the paper table (`0.4960` RMA and `0.3238` mIoU), but the other four seeds land around `0.317-0.325` RMA. If the paper table uses best-threshold mIoU, seed `42` is `0.329522`; if it uses fixed threshold `0.7`, seed `42` is `0.293174`.

## Legacy CUB Keypoint/Disk Part Localization: Activation-Based

This section is not the CUB70 mask protocol used for the paper-facing part segmentation table. It comes from the earlier keypoint/disk evaluator and is retained only for traceability.

These metrics use the mapped concept's own activation map for each part target. They are not concept-oracle metrics. `Point hit @ 0.1` is the part-point analogue of pointing accuracy: whether the strongest activation is within `0.1 * image_diagonal` of an annotated part point. `Best point-in-mask` is a separate thresholded-mask metric. The `best_*` fields are best over activation thresholds `0.3` through `0.9`.

| Metric | Mean | Std |
|---|---:|---:|
| Best point-in-mask | 0.991906 | 0.002039 |
| Point hit @ 0.1 | 0.609109 | 0.040062 |
| Best mask IoU | 0.076726 | 0.010634 |

Per seed:

| Seed | Best point-in-mask | Point hit @ 0.1 | Best mask IoU |
|---:|---:|---:|---:|
| 42 | 0.995326 | 0.680651 | 0.095656 |
| 123 | 0.992210 | 0.589816 | 0.071877 |
| 1 | 0.990728 | 0.589261 | 0.070417 |
| 6885 | 0.990310 | 0.590591 | 0.073319 |
| 0 | 0.990956 | 0.595227 | 0.072359 |

## Legacy CUB Keypoint/Disk Part Localization: Concept Oracle

This section is not the CUB70 mask protocol used for the paper-facing part segmentation table. It comes from the earlier keypoint/disk evaluator and is retained only for traceability.

These metrics were rerun with:

```text
--compute_concept_oracle
```

In this mode, for each part target, all concept maps are evaluated and the best concept score is selected independently for each metric. `Point hit @ 0.1` is the part-point pointing metric; `Best point-in-mask` is the separate thresholded-mask metric.

| Metric | Mean | Std |
|---|---:|---:|
| Best point-in-mask | 0.999202 | 0.000000 |
| Point hit @ 0.1 | 0.995934 | 0.000781 |
| Best mask IoU | 0.275859 | 0.020187 |

Per seed:

| Seed | Best point-in-mask | Point hit @ 0.1 | Best mask IoU |
|---:|---:|---:|---:|
| 42 | 0.999202 | 0.994946 | 0.311853 |
| 123 | 0.999202 | 0.996428 | 0.264984 |
| 1 | 0.999202 | 0.996922 | 0.267509 |
| 6885 | 0.999202 | 0.995934 | 0.269185 |
| 0 | 0.999202 | 0.995440 | 0.265765 |

## Interpretation

The corrected config reproduces sparse classification closely: five-seed ANEC is within about `0.001-0.002` absolute of the paper CUB RN18 SG-CBM row.

The part-localization `point_hit @ 0.1` metrics are high under the oracle protocol. The separate `point_in_mask` metric is near-saturated under oracle mode because it can choose the best concept map for each part target and thresholded-mask hit criterion.

The main unresolved issue is GDINO-box RMA variance. Seed `42` reproduces the paper-like localization behavior, while the other four seeds achieve similar classification and concept-prediction quality but place substantially less softmax mass inside GDINO boxes.
