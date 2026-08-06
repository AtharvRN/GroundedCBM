# PartImageNet++ SG-CBM/SALF Recovery Notes

Date: 2026-07-25

## Concept Set

- Canonical generic concept file: `concept_files/partimagenetpp_generic_parts.txt`
- Local concept source summary: `local_docs/partimagenetpp/generic_part_concepts_summary.json`
- Generic source list contains 785 concepts.
- `data_utils.get_concepts(...)` canonicalizes/deduplicates that source to 784 concepts.
- The completed SG-CBM run kept 783 concepts after the configured spatial-threshold concept filter.
- The one canonical concept filtered from SG-CBM was `screen or other meter`.

## PVC Payload

The recovered payload is on the Nautilus PVC mounted at:

```text
/workspace/partimagenetpp_eval_payload
```

Important files:

- `pinpp_train_images_90k.tar`
- `pinpp_val_images.tar`
- `pinpp_gdino_splits.tar`
- `partimagenetpp_gdino_thr0.1_splits/`
- `pinpp_val_images/`
- `pinpp_train_images_pvc_backup_20260725T011818Z/`
- `pinpp_val_gt_boxes_generic.jsonl`
- `pinpp_val_gt_boxes_generic.summary.json`

The original `/root/partimagenetpp_eval_payload` staging directory is not present in the currently running A100 test pods. Several older manifests point at `/root/...` or Delta NVMe paths and should not be used for new jobs unless that staging is recreated.

Canonical PVC-resolving manifests were added:

```text
/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_train_90k_manifest_pvc.jsonl
/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_val_10k_manifest_pvc.jsonl
```

Both were validated on the first 200 rows:

- train: 90,000 rows, 200/200 sampled image paths exist
- val: 10,000 rows, 200/200 sampled image paths exist

Use these environment variables for new runs:

```bash
export PARTIMAGENETPP_TRAIN_MANIFEST=/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_train_90k_manifest_pvc.jsonl
export PARTIMAGENETPP_VAL_MANIFEST=/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_val_10k_manifest_pvc.jsonl
```

## Completed SG-CBM Run

Run:

```text
/workspace/partimagenetpp_runs/sgcbm_imagenet_hparams/savlg_cbm_partimagenetpp_2026_07_25_01_18_40
```

Training configuration:

- SG-CBM internal name: `savlg_cbm`
- backbone: `resnet50`
- concept file: `concept_files/partimagenetpp_generic_parts.txt`
- GDINO supervision: `/workspace/partimagenetpp_eval_payload/partimagenetpp_gdino_thr0.1_splits`
- CBL epochs: 15
- batch size: 256
- GDINO confidence threshold in training: 0.15
- spatial target mode: `soft_box`
- mask/grid: 14x14
- global BCE positive weight: 100.0
- `loss_mask_w=1.0`
- `savlg_residual_spatial_alpha=0.1`

Training result from `metrics.txt`:

- train accuracy: 0.8240
- val/test accuracy: 0.7784
- selected sparse lambda: 0.000707
- nonzero final-layer weights: 11,128 / 783,000

Tensor-level verification at 2026-07-26 00:12 UTC:

- saved `concepts.txt`: 783 unique concepts
- `W_g.pt`: 1000 x 783, finite
- `b_g.pt`: 1000, finite
- `proj_mean.pt` / `proj_std.pt`: 1 x 783, finite
- concept-layer global weight: 783 x 2048, finite
- concept-layer spatial weight: 783 x 2048 x 1 x 1, finite

The run saved the full checkpoint tensors plus NEC-truncated final layers:

- `concept_layer.pt`
- `proj_mean.pt`
- `proj_std.pt`
- `W_g.pt`
- `b_g.pt`
- `W_g@NEC={5,10,15,20,25,30}.pt`
- `b_g@NEC={5,10,15,20,25,30}.pt`

NEC results from `nec_metrics.json`:

| NEC | Accuracy |
| --- | ---: |
| 5 | 0.7057 |
| 10 | 0.7789 |
| 15 | 0.8296 |
| 20 | 0.8389 |
| 25 | 0.8447 |
| 30 | 0.8473 |

## Black-Box Classification Reference

Pretrained torchvision ResNet50 ImageNet-v1 was evaluated directly on the
PartImageNet++ 10k validation manifest, without any CBM concept bottleneck:

```text
/workspace/partimagenetpp_results/blackbox_resnet50_imagenet1k_v1_val10k_20260726T0605Z.json
```

The run used `/tmp`-staged validation images on `a100-gpu-test-v3` after
verifying the rewritten manifest had 10,000 rows and 0 missing images.

Important leakage caveat: this PartImageNet++ validation manifest uses
ImageNet train-style filenames such as `n01440764/n01440764_1113.JPEG`, not
official ImageNet validation filenames. An exact overlap check confirmed at
least sampled entries appear inside the ImageNet train class tars, e.g.
`n01440764_3724.JPEG` is present in `n01440764.tar`. Therefore this black-box
number is not a held-out estimate for an ImageNet-pretrained torchvision
ResNet50; it should be treated as a sanity/reference score on images the
backbone likely saw during ImageNet pretraining.

- top-1: 0.8974
- top-5: 0.9864
- correct top-1: 8,974 / 10,000
- correct top-5: 9,864 / 10,000

## Human GT Evaluation

PartImageNet++ val/test evaluation should use dataset-provided GT boxes and labels:

```text
/workspace/partimagenetpp_eval_payload/pinpp_val_gt_boxes_generic.jsonl
```

Do not use GDINO pseudo-labels as evaluation GT for PartImageNet++ val/test.

Completed SG-CBM GT-box localization:

```text
/workspace/partimagenetpp_results/sgcbm_gtbox_localization_20260725T030723Z.json
```

Key metrics:

- images seen: 10,000
- target instances: 26,743
- point hit: 0.6433
- mass in GT: 0.4448
- soft IoU: 0.2871
- best mask IoU: 0.3756 at activation threshold 0.7
- best box accuracy at IoU 0.5: 0.3233 at activation threshold 0.7

Completed SG-CBM concept prediction against human part presence from human GT
boxes:

```text
/workspace/partimagenetpp_results/sgcbm_concept_metrics_humanbox_20260726T012451Z.json
```

Key metrics:

- images: 10,000
- concepts: 783
- GT source: `partimagenetpp_boxes`
- GT positive rate: 0.003415
- AUROC: 0.9977
- AP: 0.6748
- macro AP: 0.6923
- P@5: 0.4701
- best F1: 0.7043 at threshold 3.8378

Older SG-CBM concept-prediction artifacts such as
`sgcbm_concept_metrics_20260725T025208Z.json` and
`sgcbm_concept_metrics_pvcmanifest_workers0_20260725T202515Z.json` used
manifest/class-valid concept labels rather than box-derived human part
presence. Keep those only as historical checks; the human-box result above is
the preferred PartImageNet++ concept metric.

PVC-manifest smoke verification:

```text
/workspace/partimagenetpp_results/sgcbm_gtbox_localization_pvcmanifest_smoke64_20260725T201106Z.json
```

This confirmed the canonical manifests work from a fresh pod without `/root` staging.

Full PVC-manifest SG-CBM GT-box localization rerun:

```text
/workspace/partimagenetpp_results/sgcbm_gtbox_localization_pvcmanifest_workers0_20260725T202332Z.json
```

This matched the earlier full GT-box result:

- images seen: 10,000
- target instances: 26,743
- point hit: 0.6433
- mass in GT: 0.4448
- soft IoU: 0.2871
- best mask IoU: 0.3756 at activation threshold 0.7
- best box accuracy at IoU 0.5: 0.3233 at activation threshold 0.7

Full PVC-manifest SG-CBM concept prediction rerun using manifest/class-valid
labels:

```text
/workspace/partimagenetpp_results/sgcbm_concept_metrics_pvcmanifest_workers0_20260725T202515Z.json
```

This matched the earlier full manifest-label concept result:

- images: 10,000
- concepts: 783
- AUROC: 0.9979
- AP: 0.8292
- macro AP: 0.7812
- P@5: 0.5723
- best F1: 0.7712

The concept rerun was executed with `--num_workers 0`. A prior default-worker attempt showed no progress after model load on the shared PVC/DataLoader path and was interrupted, so `--num_workers 0` is the safer setting for these evaluation scripts on the current pod setup.

## GDINO vs Human GT Sanity Baseline

Completed GDINO-to-human-GT eval:

```text
/workspace/partimagenetpp_results/gdino_gt_eval_rootio_20260725T040918Z.json
```

Detection mAP:

- IoU 0.1: 0.4934
- IoU 0.3: 0.3932
- IoU 0.5: 0.3345

## SALF Status

Completed older SALF checkpoint:

```text
/workspace/partimagenetpp_eval_payload/partimagenetpp_runs/salf_cbm_partimagenetpp_2026_07_24_02_07_13
```

This checkpoint finished and wrote `concept_layer.pt`, `W_g.pt`, `b_g.pt`, `proj_mean.pt`, `proj_std.pt`, `concepts.txt`, and metrics files. It used `clip_RN50` and 761 concepts, so it is useful as an older baseline but is not concept-bank matched to the completed 783-concept SG-CBM run. Its own `metrics.txt` reports validation accuracy 58.14.

Existing older SALF evals:

- `/workspace/partimagenetpp_results/salf_concept_metrics_20260725T034609Z.json`
- `/workspace/partimagenetpp_results/salf_gtbox_localization_20260725T034450Z.json`

Older SALF key metrics:

- concept AUROC: 0.7114
- concept AP: 0.0146
- concept macro AP: 0.0614
- concept P@5: 0.0549
- concept best F1: 0.0441
- localization point hit: 0.4876
- localization mass in GT: 0.4260
- localization soft IoU: 0.1827
- localization best box accuracy at IoU 0.5: 0.2716

Canonical PVC-manifest rerun of the older unmatched SALF baseline:

```text
/workspace/partimagenetpp_results/salf_old_unmatched_concept_metrics_pvcmanifest_workers0_20260725T224117Z.json
/workspace/partimagenetpp_results/salf_old_unmatched_gtbox_localization_pvcmanifest_workers0_20260725T224117Z.json
```

Concept metrics from the canonical rerun:

- concepts: 761 named concepts; the old checkpoint has one extra unnamed score column, and the evaluator drops it
- AUROC: 0.7057
- AP: 0.0135
- macro AP: 0.0497
- P@5: 0.0560
- best F1: 0.0416

Localization metrics from the canonical rerun match the earlier old-baseline localization artifact:

- images seen: 10,000
- images with targets: 9,947
- target instances: 26,407
- point hit: 0.4876
- mass in GT: 0.4260
- soft IoU: 0.1827
- best box accuracy at IoU 0.5: 0.2716 at activation threshold 0.5

Two later SALF attempts did not complete:

- `/workspace/partimagenetpp_runs/salf_cliprn50_dense/salf_cbm_partimagenetpp_2026_07_25_08_42_22`
- `/workspace/partimagenetpp_runs/salf_resnet50_clipvitb16_dense/salf_cbm_partimagenetpp_2026_07_25_08_53_26`

Both only wrote `args.txt` and `train.log`; no `concept_layer.pt` was produced.

## Active SALF Rerun

Submitted Kubernetes job:

```text
partimagenetpp-salf-clipvitb16-a100-r1
```

Pod observed at launch:

```text
partimagenetpp-salf-clipvitb16-a100-r1-xp7bs
```

Job spec:

```text
k8s/partimagenetpp_salf_clipvitb16_a100_job.yaml
```

Runtime config path:

```text
/workspace/partimagenetpp_configs/partimagenetpp_salf_clipvitb16_dense_<RUN_TS>.json
```

Output/log roots:

```text
/workspace/partimagenetpp_runs/salf_clipvitb16_dense
/workspace/partimagenetpp_runs/salf_clipvitb16_dense_activations
/workspace/logs/partimagenetpp_salf_clipvitb16_dense_<RUN_TS>.log
```

This run uses:

- `backbone=clip_ViT-B/16`
- `lf_clip_name=clip_ViT-B/16`
- no `clip_cutoff`, so the generic concept file is kept intact before any downstream model-specific behavior
- canonical PVC train/val manifests
- human PartImageNet++ labels/boxes for later evaluation

Status at 2026-07-25 20:29 UTC:

- job is running on pod `partimagenetpp-salf-clipvitb16-a100-r1-xp7bs`
- no pod restarts observed
- output directory: `/workspace/partimagenetpp_runs/salf_clipvitb16_dense/salf_cbm_partimagenetpp_2026_07_25_20_14_00`
- live log: `/workspace/logs/partimagenetpp_salf_clipvitb16_dense_20260725T201355Z.log`
- current stage: `SALF P train` prompt-grid similarities
- observed progress: 153 / 2813 train batches, about 5%
- observed speed: about 5.7 seconds per batch on A100 80GB, GPU at 100%
- no checkpoint yet; only startup files are expected at this stage

Correction at 2026-07-25 20:57 UTC:

- this job uses `backbone=clip_ViT-B/16`, so it is not the desired ResNet50 ImageNet-v1 SALF-CBM comparison
- the expensive prompt-grid `P_train`/`P_val` targets are still reusable because they depend on CLIP ViT-B/16, the generic concepts, grid/radius, and image order, not on the downstream SALF backbone
- the job is being allowed to finish the CLIP ViT-B/16 prompt-cache generation on the A100 80GB path
- a duplicate direct ResNet50 run was briefly started on `a100-gpu-test-v2` with `spatial_batch_size=64` and `prompt_batch_size=1536`, but it was slower per image on the A100 40GB node and was stopped before completion

Corrected ResNet50 ImageNet-v1 + CLIP ViT-B/16 SALF waiter:

```text
pod: a100-gpu-test-v2
pid: 3917
config: /workspace/partimagenetpp_configs/partimagenetpp_salf_resnet50_clipvitb16_dense_matched783_20260725T211127Z.json
log: /workspace/logs/partimagenetpp_salf_resnet50_clipvitb16_dense_matched783_20260725T211127Z.log
waiter: /workspace/scripts/wait_start_salf_resnet50_clipvitb16_matched783_20260725T211127Z.sh
```

This waiter sleeps until the 784-concept CLIP ViT-B/16 prompt-cache files from the 80GB job exist, slices them down to the SG-CBM matched 783-concept bank, then launches the corrected SALF training with:

- `backbone=resnet50`
- Torchvision ResNet50 `IMAGENET1K_V1` through `data_utils.get_target_model("resnet50")`
- `lf_clip_name=clip_ViT-B/16`
- `filter_set=/workspace/partimagenetpp_eval_payload/partimagenetpp_sgcbm_filtered_out.txt`
- dropped concept: `screen or other meter`
- `cbl_batch_size=256`
- canonical PVC manifests

Later at 2026-07-25 21:53 UTC, `a100-gpu-test-v2` and `a100-gpu-test` were killed by the cluster and separate Places365 jobs were scheduled onto those nodes. The v2 waiter did not survive. `a100-gpu-test-v3` was not used because it had unrelated CUB localization work running.

At 2026-07-25 22:29 UTC, `a100-gpu-test` reappeared as an idle 2xA100-80GB pod. The same matched783 waiter was restarted there:

```text
pod: a100-gpu-test
pid: 1285
config: /workspace/partimagenetpp_configs/partimagenetpp_salf_resnet50_clipvitb16_dense_matched783_20260725T211127Z.json
log: /workspace/logs/partimagenetpp_salf_resnet50_clipvitb16_dense_matched783_20260725T211127Z.log
waiter: /workspace/scripts/wait_start_salf_resnet50_clipvitb16_matched783_20260725T211127Z.sh
```

At 2026-07-25 23:08 UTC, unrelated CUB70 localization jobs were observed using both GPUs in `a100-gpu-test`, so the sleeping matched783 waiter was stopped to avoid later contention. Use the Kubernetes job spec below after the reusable caches exist, or restart the waiter only on a confirmed idle pod.

At 2026-07-26 00:41 UTC, the reusable CLIP ViT-B/16 train prompt cache completed:

```text
/workspace/partimagenetpp_runs/salf_clipvitb16_dense_activations/partimagenetpp_train_salf_clip_ViT-B/16_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_0ef4831614c7cee8_P.pt
/workspace/partimagenetpp_runs/salf_clipvitb16_dense_activations/partimagenetpp_train_salf_clip_ViT-B/16_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_0ef4831614c7cee8_meta.json
```

The train cache file is about 13GB. The same job then moved to `SALF P val` for the 10,000-image val split.

Prepared corrected Kubernetes job spec:

```text
k8s/partimagenetpp_salf_resnet50_clipvitb16_matched783_a100_job.yaml
```

This job validates with `kubectl apply --dry-run=client`. Submit it after the reusable CLIP ViT-B/16 `P_train` and `P_val` caches exist so it does not reserve an A100 while sleeping. It performs the same 784-to-783 cache slicing and then trains the corrected ResNet50/ImageNet-v1 + CLIP ViT-B/16 SALF-CBM.

Monitor with:

```bash
kubectl get job partimagenetpp-salf-clipvitb16-a100-r1
kubectl get pod -l job-name=partimagenetpp-salf-clipvitb16-a100-r1
kubectl logs -f job/partimagenetpp-salf-clipvitb16-a100-r1
```

After completion, evaluate with:

```bash
export PARTIMAGENETPP_TRAIN_MANIFEST=/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_train_90k_manifest_pvc.jsonl
export PARTIMAGENETPP_VAL_MANIFEST=/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_val_10k_manifest_pvc.jsonl
RUN_DIR=$(ls -td /workspace/partimagenetpp_runs/salf_clipvitb16_dense/salf_cbm_partimagenetpp_* | head -1)

python scripts/eval_concept_accuracy.py \
  --dataset partimagenetpp \
  --load_paths "${RUN_DIR}" \
  --model_names salf_cbm \
  --names SALF-CLIP-ViT-B16 \
  --partimagenetpp_train_manifest "${PARTIMAGENETPP_TRAIN_MANIFEST}" \
  --partimagenetpp_val_manifest "${PARTIMAGENETPP_VAL_MANIFEST}" \
  --normalization minmax \
  --output /workspace/partimagenetpp_results/salf_clipvitb16_concept_metrics.json

python scripts/eval_partimagenetpp_gtbox_localization.py \
  --gcbm_path "${RUN_DIR}" \
  --model_name salf_cbm \
  --train_manifest "${PARTIMAGENETPP_TRAIN_MANIFEST}" \
  --val_manifest "${PARTIMAGENETPP_VAL_MANIFEST}" \
  --gt_boxes_jsonl /workspace/partimagenetpp_eval_payload/pinpp_val_gt_boxes_generic.jsonl \
  --output /workspace/partimagenetpp_results/salf_clipvitb16_gtbox_localization.json \
  --map_normalization concept_zscore_minmax
```

## Current VLG/SALF status at 2026-07-26 02:50 UTC

The wrong-backbone SALF cache-producing Kubernetes job was stopped after the
reusable CLIP ViT-B/16 train/val prompt caches were complete:

```text
kubectl delete job -n wenglab-interpretable-ai partimagenetpp-salf-clipvitb16-a100-r1
```

The reusable cache artifacts are:

```text
/workspace/partimagenetpp_runs/salf_clipvitb16_dense_activations/partimagenetpp_train_salf_clip_ViT-B/16_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_0ef4831614c7cee8_P.pt
/workspace/partimagenetpp_runs/salf_clipvitb16_dense_activations/partimagenetpp_val_salf_clip_ViT-B/16_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_0ef4831614c7cee8_P.pt
```

The first corrected SALF-CBM attempt on `a100-gpu-test-v3` did not finish. The
pod hit its `activeDeadlineSeconds` and completed/killed the sshd container,
which also killed the background training and watcher. This was not a model
exception. Last persisted progress from the shared logs was early CBL epoch 2;
the run directory contained only `args.txt` and `train.log`, with no final
`concept_layer.pt`, so there are no valid corrected SALF metrics from that
attempt.

The reusable SG-matched cache slices are complete:

```text
/workspace/partimagenetpp_runs/salf_resnet50_clipvitb16_dense_activations/partimagenetpp_train_salf_resnet50_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_5431310002d47ebf_P.pt
/workspace/partimagenetpp_runs/salf_resnet50_clipvitb16_dense_activations/partimagenetpp_val_salf_resnet50_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_5431310002d47ebf_P.pt
```

Corrected SALF-CBM was relaunched as durable Kubernetes job r2, but r2 was
stopped before any checkpoint because its workers blocked in Ceph/PVC random
image reads (`folio_wait_bit_common` / `ceph_mdsc_wait_request`) before the
first CBL progress line. The reusable tensor caches had loaded, but image I/O
was the bottleneck and no `concept_layer.pt` was written.

Corrected SALF-CBM r3 confirmed that local staging fixed the PVC random-read
stall, completed all 10 CBL epochs, and entered dense final-layer training.
However, r3 had been launched with `saga_n_iters=1000`, which the dense final
layer treats as 1000 epochs. By final-layer epoch 100 the validation accuracy
had plateaued around 81.25-81.28% and the learning rate had decayed to about
`5e-6`; no final artifacts are written until the loop exits. r3 was therefore
stopped before final artifact write and replaced with r4 using the corrected
`saga_n_iters=100`.

Corrected SALF-CBM r4 completed successfully with local image staging:

```text
job: partimagenetpp-salf-r50-clipvitb16-matched783-r4
pod: partimagenetpp-salf-r50-clipvitb16-matched783-r4-rxdmt
node: rci-nrp-gpu-03.sdsu.edu
run_ts: 20260726T023652Z
state: Succeeded at 2026-07-26 02:58 UTC
config: /workspace/partimagenetpp_configs/partimagenetpp_salf_resnet50_clipvitb16_dense_matched783_20260726T023652Z.json
log: /workspace/logs/partimagenetpp_salf_resnet50_clipvitb16_dense_matched783_20260726T023652Z.log
local stage root: /tmp/partimagenetpp_salf_20260726T023652Z
run dir: /workspace/partimagenetpp_runs/salf_resnet50_clipvitb16_dense/salf_cbm_partimagenetpp_2026_07_26_02_39_25
```

The r4 job has `activeDeadlineSeconds=172800` and runs post-training evaluation
inside the same job after `train_cbm.py` exits successfully. Startup/status
logs verified:

- `backbone=resnet50`
- `lf_clip_name=clip_ViT-B/16`
- `dataset=partimagenetpp`
- `filter_set=/workspace/partimagenetpp_eval_payload/partimagenetpp_sgcbm_filtered_out.txt`
- dropped concept: `screen or other meter`
- matched prompt-grid caches already exist and were reused
- before training, r4 extracts `pinpp_train_images_90k.tar` and
  `pinpp_val_images.tar` to local `/tmp`, then rewrites train/val manifests to
  point at the staged files
- r4 uses `saga_n_iters=100` for the dense final layer
- r4 finished staging and manifest rewrite, loaded 90,000 train and 10,000 val
  examples from local manifests, and loaded the matched cached spatial tensors
  at 2026-07-26 02:40 UTC

Completed r4 training:

- CBL epoch 0 completed with `train_loss=-0.901843`,
  `val_loss=-0.946136`, `best_val=-0.946136`
- CBL epoch 1 completed with `train_loss=-0.957324`,
  `val_loss=-0.959787`, `best_val=-0.959787`
- CBL epoch 2 completed with `train_loss=-0.965001`,
  `val_loss=-0.964481`, `best_val=-0.964481`
- CBL epoch 3 completed with `train_loss=-0.968599`,
  `val_loss=-0.967574`, `best_val=-0.967574`
- CBL epoch 4 completed with `train_loss=-0.971123`,
  `val_loss=-0.970230`, `best_val=-0.970230`
- CBL epoch 5 completed with `train_loss=-0.973440`,
  `val_loss=-0.972778`, `best_val=-0.972778`
- CBL epoch 6 completed with `train_loss=-0.975748`,
  `val_loss=-0.975330`, `best_val=-0.975330`
- CBL epoch 7 completed with `train_loss=-0.978042`,
  `val_loss=-0.977747`, `best_val=-0.977747`
- CBL epoch 8 completed with `train_loss=-0.980151`,
  `val_loss=-0.979815`, `best_val=-0.979815`
- CBL epoch 9 completed with `train_loss=-0.981910`,
  `val_loss=-0.981394`, `best_val=-0.981394`
- dense final-layer training completed 100 epochs; final reported
  `val_accuracy=81.24`
- saved classifier artifacts: `concept_layer.pt`, `W_g.pt`, `b_g.pt`,
  `proj_mean.pt`, `proj_std.pt`, `concepts.txt`, `metrics.txt`,
  `test_metrics.json`
- `test_metrics.json` reports image classification accuracy `0.8124`

r3 progress before it was stopped:

- at 2026-07-26 02:20 UTC, r3 completed the first CBL epoch:
  `train_loss=-0.901843`, `val_loss=-0.946136`, `best_val=-0.946136`,
  `stale_epochs=0`
- at 2026-07-26 02:21 UTC, r3 completed the second CBL epoch:
  `train_loss=-0.957324`, `val_loss=-0.959787`, `best_val=-0.959787`,
  `stale_epochs=0`
- at 2026-07-26 02:23 UTC, r3 completed the third CBL epoch:
  `train_loss=-0.965001`, `val_loss=-0.964481`, `best_val=-0.964481`,
  `stale_epochs=0`
- at 2026-07-26 02:24 UTC, r3 completed the fourth CBL epoch:
  `train_loss=-0.968599`, `val_loss=-0.967574`, `best_val=-0.967574`,
  `stale_epochs=0`
- at 2026-07-26 02:25 UTC, r3 completed the fifth CBL epoch:
  `train_loss=-0.971123`, `val_loss=-0.970230`, `best_val=-0.970230`,
  `stale_epochs=0`
- at 2026-07-26 02:27 UTC, r3 completed the sixth CBL epoch:
  `train_loss=-0.973440`, `val_loss=-0.972778`, `best_val=-0.972778`,
  `stale_epochs=0`
- at 2026-07-26 02:28 UTC, r3 completed the seventh CBL epoch:
  `train_loss=-0.975748`, `val_loss=-0.975330`, `best_val=-0.975330`,
  `stale_epochs=0`
- at 2026-07-26 02:30 UTC, r3 completed the eighth CBL epoch:
  `train_loss=-0.978042`, `val_loss=-0.977747`, `best_val=-0.977747`,
  `stale_epochs=0`
- at 2026-07-26 02:31 UTC, r3 completed the ninth CBL epoch:
  `train_loss=-0.980151`, `val_loss=-0.979815`, `best_val=-0.979815`,
  `stale_epochs=0`
- at 2026-07-26 02:32 UTC, r3 completed the tenth CBL epoch:
  `train_loss=-0.981910`, `val_loss=-0.981394`, `best_val=-0.981394`,
  `stale_epochs=0`
- dense final-layer epoch 100 had `val_acc=81.28` before r3 was stopped

After training, r4 wrote latest pointers and ran:

- `scripts/eval_concept_accuracy.py` with `--gt_source partimagenetpp_boxes`
- `scripts/eval_partimagenetpp_gtbox_localization.py` against `pinpp_val_gt_boxes_generic.jsonl`

Corrected SALF outputs:

```text
/workspace/partimagenetpp_results/salf_resnet50_clipvitb16_matched783_concept_metrics_20260726T023652Z.json
/workspace/partimagenetpp_results/salf_resnet50_clipvitb16_matched783_gtbox_localization_20260726T023652Z.json
```

Corrected SALF concept prediction against human box-derived concept presence:

- images: 10,000
- concepts: 783
- GT source: `partimagenetpp_boxes`
- GT positive rate: 0.003415
- AUROC: 0.4299
- AP: 0.0030
- macro AP: 0.0348
- P@5: 0.0042
- best F1: 0.0070 at threshold 1.8007

Corrected SALF native-spatial GT-box localization:

- images seen: 10,000
- target instances: 26,743
- localization source: `native_spatial_maps`
- map normalization: `concept_zscore_minmax`
- point hit: 0.3679
- mass in GT: 0.3119
- soft IoU: 0.2187
- best mask IoU: 0.2779 at activation threshold 0.3
- best box accuracy at IoU 0.5: 0.2450 at activation threshold 0.5

SALF dense-head truncation sanity check, not SG-style NEC:

```text
/workspace/partimagenetpp_results/salf_resnet50_clipvitb16_matched783_nec_metrics_20260726T044128Z.json
```

The SALF NEC pass used `/tmp`-staged val images because direct PVC small-file
reads repeatedly blocked in Ceph I/O. This artifact evaluates the saved dense
SALF classifier after NEC truncation. It should not be compared to SG-CBM NEC,
because SG-CBM NEC uses sparse GLM heads trained along the regularization path.
Keep this table only as a sanity check for dense-head sensitivity to truncation.

| NEC | Top-1 | Top-5 |
| --- | ---: | ---: |
| 5 | 0.0113 | 0.0315 |
| 10 | 0.0219 | 0.0576 |
| 15 | 0.0337 | 0.0818 |
| 20 | 0.0447 | 0.1091 |
| 25 | 0.0587 | 0.1327 |
| 30 | 0.0751 | 0.1568 |

Corrected SALF sparse-path NEC classification:

```text
feature artifact: /workspace/partimagenetpp_runs/salf_resnet50_clipvitb16_nec_path_artifact_20260726T0520Z
path run: /workspace/partimagenetpp_runs/salf_resnet50_clipvitb16_nec_path_20260726T0524Z
saved-head eval: /workspace/partimagenetpp_results/salf_resnet50_clipvitb16_sparse_nec_path_eval_20260726T0535Z.json
```

This run extracts normalized SALF concept features from the corrected
ResNet50/ImageNet-v1 + CLIP ViT-B/16 checkpoint using explicit
PartImageNet++ 90k train and 10k validation manifests, then trains sparse GLM
heads along the path. These are the comparable SALF NEC numbers.

| NEC | Top-1 | Top-5 |
| --- | ---: | ---: |
| 5 | 0.1095 | 0.2147 |
| 10 | 0.2034 | 0.3600 |
| 15 | 0.3339 | 0.5508 |
| 20 | 0.4193 | 0.6613 |
| 25 | 0.5244 | 0.7623 |
| 30 | 0.6012 | 0.8231 |

VLG-CBM for PartImageNet++ completed on `a100-gpu-test`, GPU 0:

```text
run dir: /workspace/partimagenetpp_runs/vlg_resnet50_global_only/vlg_cbm_partimagenetpp_r50v1_global_only_matched783_20260726T010939Z
log: /workspace/logs/partimagenetpp_vlg_resnet50_global_only_tmpstaged_20260726T010939Z.log
runner: /workspace/GroundedCBM/scripts/run_partimagenetpp_vlg_resnet50_global_only.sh
```

This VLG run uses:

- ResNet50 `IMAGENET1K_V1`
- `branch_arch=global_only`
- 15 CBL epochs via `gcbm/train_imagenet.py --epochs 15`
- SG-matched 783 generic PartImageNet++ concepts from the completed SG-CBM checkpoint
- `/tmp`-staged train/val images to avoid PVC random-read bottlenecks
- `batch_size=256`, `workers=8`, `prefetch_factor=2`
- sparse final layer enabled after CBL

The first stable training interval showed epoch 1 progress through step
120/351 at about 115-120 images/second.

Completed VLG-CBM run at 2026-07-26 04:12 UTC:

- CBL training finished all 15 epochs
- epoch 1 completed with validation loss 0.0546
- epoch 2 completed with validation loss 0.0460
- epoch 3 completed with validation loss 0.0427
- epoch 4 completed with validation loss 0.0418
- epoch 5 completed with validation loss 0.0414
- epoch 6 completed with validation loss 0.0424
- epoch 7 completed with validation loss 0.0418
- epoch 8 completed with validation loss 0.0429
- epoch 9 completed with validation loss 0.0427
- epoch 10 completed with validation loss 0.0425
- epoch 11 completed with validation loss 0.0433
- epoch 12 completed with validation loss 0.0436
- epoch 13 completed with validation loss 0.0440
- epoch 14 completed with validation loss 0.0440
- epoch 15 completed with validation loss 0.0441
- sparse final-layer training started from `concept_head_best.pt`
- checkpoint `checkpoint_epoch_015.pt` and `concept_head_latest.pt` were
  refreshed at 2026-07-26 03:45 UTC
- throughput was unstable in the recent interval, ranging from about 110 to
  780 images/second depending on cache/loader state
- current run directory already has `config.json`, `concepts.txt`, and
  `concept_head_best.pt`; `concept_head_best.pt` was refreshed at
  2026-07-26 02:26 UTC
- sparse final layer completed with train top-1 0.6362, val top-1 0.6144,
  train top-5 0.8552, val top-5 0.8390
- selected sparse lambda: 0.000707
- nonzero final-layer weights: 144,304 / 783,000
- train feature extraction: 90,000 examples, 111.0 images/second
- val feature extraction: 10,000 examples, 65.6 images/second

VLG-CBM dense/saved-head truncation sanity check, not SG-style NEC:

```text
/workspace/partimagenetpp_results/vlg_resnet50_global_only_nec_metrics_20260726T042800Z.json
```

This table thresholds an already-selected sparse final layer and is useful only
as a sanity check. For SG-style NEC comparison, use the sparse-path VLG table
below.

| NEC | Top-1 | Top-5 |
| --- | ---: | ---: |
| 5 | 0.2782 | 0.4976 |
| 10 | 0.3451 | 0.5788 |
| 15 | 0.4031 | 0.6439 |
| 20 | 0.4377 | 0.6809 |
| 25 | 0.4711 | 0.7135 |
| 30 | 0.4976 | 0.7342 |

Corrected VLG-CBM sparse-path NEC classification:

```text
feature artifact: /workspace/partimagenetpp_runs/vlg_resnet50_global_only_nec_path_artifact_20260726T0449Z
path run: /workspace/partimagenetpp_runs/vlg_resnet50_global_only_nec_path_20260726T0449Z
saved-head eval: /workspace/partimagenetpp_results/vlg_resnet50_global_only_nec_path_eval_20260726T0510Z.json
```

| NEC | Top-1 | Top-5 |
| --- | ---: | ---: |
| 5 | 0.4954 | 0.6664 |
| 10 | 0.5751 | 0.7775 |
| 15 | 0.6730 | 0.8769 |
| 20 | 0.7561 | 0.9400 |
| 25 | 0.7998 | 0.9576 |
| 30 | 0.8147 | 0.9626 |

Completed VLG-CBM human-box concept prediction:

```text
/workspace/partimagenetpp_results/vlg_resnet50_global_only_humanbox_concept_metrics_20260726T013603Z.json
```

Key metrics:

- images: 10,000
- concepts: 783
- GT source: `partimagenetpp_boxes`
- GT positive rate: 0.003415
- AUROC: 0.9978
- AP: 0.6864
- macro AP: 0.7038
- P@5: 0.4757
- best F1: 0.7164 at threshold 4.0266

Completed VLG-CBM CAM-style GT-box localization:

```text
/workspace/partimagenetpp_results/vlg_resnet50_global_only_cam_gtbox_localization_20260726T013603Z.json
```

Key metrics:

- images seen: 10,000
- target instances: 26,743
- point hit: 0.5661
- mass in GT: 0.5570
- soft IoU: 0.1158
- best mask IoU: 0.3569 at activation threshold 0.3
- best box accuracy at IoU 0.5: 0.3226 at activation threshold 0.5

The VLG watcher now runs both:

- `scripts/eval_concept_accuracy.py` against human box-derived PartImageNet++ concept presence
- `scripts/eval_partimagenetpp_gtbox_localization.py` with `model_name=vlg_cbm`

The VLG localization result is CAM-style because the VLG model is global-only:
the evaluator applies the learned global linear concept weights over conv5
feature maps and writes `localization_source=global_head_conv5_cam`. Treat this
as a diagnostic localization comparison, not as a native spatial-branch metric
like SG-CBM/SALF-CBM.

At 2026-07-26 01:24 UTC, the PartImageNet++ concept evaluator was updated to
support explicit human box-derived concept GT:

```bash
python scripts/eval_concept_accuracy.py \
  --dataset partimagenetpp \
  --gt_source partimagenetpp_boxes \
  --partimagenetpp_gt_boxes_jsonl /workspace/partimagenetpp_eval_payload/pinpp_val_gt_boxes_generic.jsonl \
  ...
```

This is the preferred concept-prediction source for PartImageNet++ because it
marks a generic part positive only when the human GT box annotations contain at
least one box for that part in the image. Fresh box-derived concept evaluations
were launched/wired for:

```text
SG-CBM result pointer: /workspace/partimagenetpp_results/sgcbm_humanbox_concept_latest_result.txt
VLG watcher log pointer: /workspace/logs/partimagenetpp_vlg_resnet50_global_only_latest_eval_log.txt
SALF r4 job log: /workspace/logs/partimagenetpp_salf_resnet50_clipvitb16_dense_matched783_20260726T023652Z.log
SALF r4 expected concept result: /workspace/partimagenetpp_results/salf_resnet50_clipvitb16_matched783_concept_metrics_20260726T023652Z.json
SALF r4 expected localization result: /workspace/partimagenetpp_results/salf_resnet50_clipvitb16_matched783_gtbox_localization_20260726T023652Z.json
```
