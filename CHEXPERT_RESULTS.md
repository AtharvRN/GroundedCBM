# CheXpert Results Tracker

Last updated: 2026-05-31.

This file tracks CheXpert integration, annotation/cache status, completed quantitative results, and in-progress runs. Metrics are multilabel unless otherwise noted: mean AUROC and mAP/mean AP are the primary metrics.

## Concept-Set Comparison

This section is the first place to update when a new concept set or NEC sweep finishes. It compares concept sets at fixed NEC values using mean AUROC and mAP.

### Best Overall Per Concept Set

| Concept set | Method / source | Train split | Concept threshold | Final layer / sweep | Best reported point | Mean AUROC | mAP | Status |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.50 | Natural GLM-SAGA lambda path | NEC@50 | 0.7964 | 0.5572 | Complete |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.50 | One GLM-SAGA solution, post-hoc NEC truncation | NEC@30 | 0.7731 | 0.5582 | Complete |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.70 | One GLM-SAGA solution, post-hoc NEC truncation | NEC@30 | 0.7673 | 0.5134 | Complete |
| queries520 | Old Medical_CBM LF-CBM | Full CheXpert | 0.25 cutoff in old LF setup | Upstream sparse/NEC | NEC@30 | 0.7930 | 0.5393 | Reference only |
| queries520 | Old Medical_CBM LF-CBM | 25k CheXpert | 0.25 cutoff in old LF setup | Upstream sparse/NEC | NEC@30 | 0.8087 | 0.5056 | Reference only |
| gpt225 | Integrated VLG-CBM | Full CheXpert | 0.70 | One GLM-SAGA solution, post-hoc NEC truncation | Dense/SAGA final | 0.8167 | 0.5430 | Complete |

### AUROC By NEC

| Concept set / run | NEC@5 | NEC@10 | NEC@15 | NEC@20 | NEC@25 | NEC@30 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| queries520, integrated VLG, thr0.50 | 0.7947 | 0.7961 | 0.7952 | 0.7943 | 0.7951 | 0.7962 | Natural GLM-SAGA lambda path |
| queries520, integrated VLG, thr0.50 | 0.6190 | 0.6248 | 0.6295 | 0.6975 | 0.7504 | 0.7731 | One SAGA solution, post-hoc truncation |
| queries520, integrated VLG, thr0.70 | 0.6067 | 0.6610 | 0.6372 | 0.6697 | 0.7253 | 0.7673 | One SAGA solution, post-hoc truncation |
| queries520, old Medical_CBM LF full | TBD | TBD | TBD | TBD | TBD | 0.7930 | Only NEC@30 currently recorded |
| queries520, old Medical_CBM LF 25k | TBD | TBD | TBD | TBD | TBD | 0.8087 | Only NEC@30 currently recorded |
| gpt225, integrated VLG, thr0.70 | 0.5972 | 0.6696 | 0.7368 | 0.7115 | 0.7466 | 0.7819 | One SAGA solution, post-hoc truncation |

### mAP By NEC

| Concept set / run | NEC@5 | NEC@10 | NEC@15 | NEC@20 | NEC@25 | NEC@30 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| queries520, integrated VLG, thr0.50 | 0.5364 | 0.5519 | 0.5508 | 0.5511 | 0.5557 | 0.5603 | Natural GLM-SAGA lambda path |
| queries520, integrated VLG, thr0.50 | 0.3788 | 0.4475 | 0.4505 | 0.4902 | 0.5414 | 0.5582 | One SAGA solution, post-hoc truncation |
| queries520, integrated VLG, thr0.70 | 0.3535 | 0.4157 | 0.4118 | 0.4387 | 0.4651 | 0.5134 | One SAGA solution, post-hoc truncation |
| queries520, old Medical_CBM LF full | TBD | TBD | TBD | TBD | TBD | 0.5393 | Only NEC@30 currently recorded |
| queries520, old Medical_CBM LF 25k | TBD | TBD | TBD | TBD | TBD | 0.5056 | Only NEC@30 currently recorded |
| gpt225, integrated VLG, thr0.70 | 0.3320 | 0.3934 | 0.4499 | 0.4490 | 0.4714 | 0.5116 | One SAGA solution, post-hoc truncation |

### Dense / Untruncated Final-Layer Comparison

| Concept set | Method / source | Train split | Concept threshold | Mean AUROC | mAP | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.50 | 0.7388 | 0.4810 | Sparse/SAGA final layer before NEC truncation; `5358/7280` nnz |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.70 | 0.7339 | 0.4922 | Sparse/SAGA final layer |
| queries520 | Old Medical_CBM LF-CBM | Full CheXpert | 0.25 cutoff in old LF setup | 0.7471 | 0.4965 | Reference only |
| queries520 | Old Medical_CBM LF-CBM | 25k CheXpert | 0.25 cutoff in old LF setup | 0.8176 | 0.5301 | Reference only, not directly comparable to full-data runs |
| gpt225 | Integrated VLG-CBM | Full CheXpert | 0.70 | 0.8167 | 0.5430 | Sparse/SAGA final layer before NEC truncation; filtered `225 -> 223` |

Notes:
- `TBD` means we do not yet have an artifact/log for that exact NEC value.
- The two integrated queries520 rows above are comparable at the dataset level, but their concept thresholds differ (`0.50` versus `0.70`).
- The old Medical_CBM LF-CBM rows are useful baselines but are not produced by the unified GroundedCBM entrypoint.
- The natural GLM/NEC lambda sweep is the preferred sparse comparison for queries520 threshold `0.50`, because it selects sparse heads along a lambda path rather than truncating one dense-ish SAGA solution.

## Data And Cache Status

| Asset | Status | Path / Notes |
| --- | --- | --- |
| CheXpert data | Available on pods | `/workspace/CHEXPERT_DATASET` |
| DenseNet121 CheXpert backbone | Available | `/workspace/Medical_CBM/checkpoints/densenet121_full_finetune_feb8/best_model.pth` |
| queries520 concept file | Integrated | `concept_files/chexpert_queries_520.txt` |
| queries520 threshold 0.50 train presence cache | Complete | `/workspace/CHEX_CBM/annotations_queries520_train/presence_cache_191027_520_thr050.pt` |
| queries520 threshold 0.50 valid presence cache | Complete | `/workspace/CHEX_CBM/annotations_queries520_valid/presence_cache_202_520_thr050.pt` |
| gpt225 full train annotations | Complete | `/workspace/CHEX_CBM/annotations_gpt225_train_full`, 191027/191027 JSONs |
| gpt225 full valid annotations | Complete | `/workspace/CHEX_CBM/annotations_gpt225_valid_full`, 202/202 JSONs |
| gpt225 threshold 0.70 train presence cache | Complete | `/workspace/CHEX_CBM/annotations_gpt225_train_full/presence_cache_191027_225_thr070.pt`, shape `(191027, 225)`, avg positives/image `34.31` |
| gpt225 threshold 0.70 valid presence cache | Complete | `/workspace/CHEX_CBM/annotations_gpt225_valid_full/presence_cache_202_225_thr070.pt`, shape `(202, 225)`, avg positives/image `27.22` |

Notes:
- The old monolithic gpt225 annotation job failed with OOM, but the later 4-shard train jobs plus valid job completed.
- The first gpt225 cache job failed due to `PYTHONPATH`; the rerun cache job completed with `0` unmatched rows.
- The old Medical_CBM gpt225 `.npz` cache was not row-order compatible with current full CheXpert train/official valid, so it should not be used for current training.

## Completed Runs

### Integrated GroundedCBM VLG-CBM, queries520, threshold 0.50

| Field | Value |
| --- | --- |
| Run | `chexpert_vlg_queries520_thr050_freq001_095_simplegpu_20260530T020815Z` |
| Run dir | `/workspace/chexpert_full_runs/chexpert_vlg_queries520_thr050_freq001_095_simplegpu_20260530T020815Z` |
| Log | `/workspace/logs/chexpert_vlg_queries520_thr050_freq001_095_simplegpu_20260530T020815Z.log` |
| Concepts | queries520 |
| Concept threshold | `0.50` |
| Frequency filtering | `min=0.001`, `max=0.95`; filtered `520 -> 520` |
| Presence mode | binary |
| Backbone | CheXpert-pretrained DenseNet121 |
| Final layer | GLM-SAGA, `saga_lam=0.0007`, `saga_iters=500` |
| GLM convergence | Did not converge at 500 iterations |
| Full SAGA sparsity | `5358 / 7280` nonzero weights, density `0.7360` |
| Val mean AUROC | `0.7388` |
| Val mAP | `0.4810` |

NEC metrics from post-hoc truncation of the learned final layer:

| NEC | nnz | Mean AUROC | mAP |
| ---: | ---: | ---: | ---: |
| 5 | 70 | 0.6190 | 0.3788 |
| 10 | 140 | 0.6248 | 0.4475 |
| 15 | 210 | 0.6295 | 0.4505 |
| 20 | 280 | 0.6975 | 0.4902 |
| 25 | 350 | 0.7504 | 0.5414 |
| 30 | 420 | 0.7731 | 0.5582 |

Interpretation:
- NEC target sparsity was hit exactly for each post-hoc NEC point, e.g. NEC@5 keeps `5 * 14 = 70` weights.
- NEC@5 is low because the full SAGA layer is still dense-ish and NEC@5 is a very aggressive truncation.
- The run used queries520 threshold-0.50 caches, not the newer gpt225 caches.

### Natural GLM/NEC Lambda Sweep, queries520, threshold 0.50

| Field | Value |
| --- | --- |
| Source run | `chexpert_vlg_queries520_thr050_freq001_095_simplegpu_20260530T020815Z` |
| Output dir | `/workspace/chexpert_full_runs/chexpert_vlg_queries520_thr050_freq001_095_simplegpu_20260530T020815Z/medical_glm_nec_natural_lamauto_eps0005_k50_it80_20260531T033124Z` |
| Log | `/workspace/logs/medical_glm_nec_natural_lamauto_eps0005_k50_it80_20260531T033124Z.log` |
| Selection | Natural GLM-SAGA sparse path first; truncation only as fallback |
| Runtime | `709.1` seconds |

Preferred NEC metrics from naturally sparse GLM path:

| NEC | nnz | Mean AUROC | mAP |
| ---: | ---: | ---: | ---: |
| 1 | 12 | 0.6117 | 0.3712 |
| 2 | 27 | 0.7251 | 0.4959 |
| 3 | 42 | 0.7474 | 0.5087 |
| 4 | 57 | 0.7771 | 0.5243 |
| 5 | 71 | 0.7947 | 0.5364 |
| 7 | 97 | 0.7587 | 0.5371 |
| 10 | 141 | 0.7961 | 0.5519 |
| 15 | 204 | 0.7952 | 0.5508 |
| 20 | 268 | 0.7943 | 0.5511 |
| 25 | 340 | 0.7951 | 0.5557 |
| 30 | 422 | 0.7962 | 0.5603 |
| 40 | 588 | 0.7961 | 0.5579 |
| 50 | 664 | 0.7964 | 0.5572 |

### Integrated GroundedCBM VLG-CBM, queries520, threshold 0.70

| Field | Value |
| --- | --- |
| Run | `chexpert_vlg_queries520_nec_from_cbl_cached_a100_resume_20260528T072102Z` |
| Run dir | `/workspace/sgcbm_medical_runs/chexpert_vlg_queries520_nec_from_cbl_cached_a100_resume_20260528T072102Z` |
| Concepts | queries520 |
| Concept threshold | `0.70` |
| Frequency filtering | none |
| Final layer | sparse/SAGA |
| Val mean AUROC | `0.7339` |
| Val mAP | `0.4922` |

NEC metrics:

| NEC | Mean AUROC | mAP |
| ---: | ---: | ---: |
| 5 | 0.6067 | 0.3535 |
| 10 | 0.6610 | 0.4157 |
| 15 | 0.6372 | 0.4118 |
| 20 | 0.6697 | 0.4387 |
| 25 | 0.7253 | 0.4651 |
| 30 | 0.7673 | 0.5134 |

Note: for `--use_saga`, `val_metrics.json` is the sparse/SAGA final-layer metric, not dense final-layer accuracy.

## Reference Medical_CBM Results

These are old-reference Medical_CBM LF-CBM results, useful as baselines but not produced by the current unified GroundedCBM entrypoint.

| Setting | Path | Val mean AUROC | Val mean AP / mAP | Upstream NEC@30 mean AUROC | Upstream NEC@30 mean AP / mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| Full CheXpert, LF-CBM, queries520 | `/workspace/Medical_CBM/checkpoints/lfcbm_chexpert_full_queries520_cxrclip_cut025_lam1e3_medicalcbm_r1` | 0.7471 | 0.4965 | 0.7930 | 0.5393 |
| 25k CheXpert, LF-CBM, queries520 | `/workspace/Medical_CBM/checkpoints/lfcbm_chexpert25k_queries520_cxrclip_cut025_lam1e3_medicalcbm_r1` | 0.8176 | 0.5301 | 0.8087 | 0.5056 |

## Annotation / Threshold Diagnostics

Score distribution from existing `presence_scores` showed low thresholds are too permissive:

| Split | Threshold | Avg positives/image |
| --- | ---: | ---: |
| Train random 5k | 0.10 | 516.6 |
| Train random 5k | 0.25 | 502.8 |
| Train random 5k | 0.50 | 354.8 |
| Train random 5k | 0.70 | 91.2 |
| Val | 0.50 | 306.1 |
| Val | 0.70 | 69.2 |

For gpt225 threshold 0.70, the new full-cache densities are much lower:

| Split | Avg positives/image |
| --- | ---: |
| Train full | 34.31 |
| Valid official | 27.22 |

## Runtime Validation Status

Smoke/runtime validation for all four CBM variants on CheXpert has passed on A100 pods:

| Model | Status | Notes |
| --- | --- | --- |
| VLG-CBM | Passed smoke/runtime validation | Unified `train_cbm.py` path |
| SGCBM | Passed smoke/runtime validation | Called SGCBM, not SAVLG, in user-facing naming |
| LF-CBM | Passed smoke/runtime validation | Medical defaults switched away from CLIP RN50; intended alignment models are CXR-CLIP / BiomedCLIP |
| SALF-CBM | Passed smoke/runtime validation | Uses medical LF alignment defaults |

Validation run dirs observed earlier:
- `/workspace/validation_runs/unified_vlg_smoke_20260529T030342Z`
- `/workspace/validation_runs/savlg_cbm_chexpert_2026_05_29_03_05_56`
- `/workspace/validation_runs/lf_cbm_chexpert_2026_05_29_03_07_31`
- `/workspace/validation_runs/salf_cbm_chexpert_2026_05_29_03_09_12`

## In Progress

### VLG-CBM gpt225 Full CheXpert Training

| Field | Value |
| --- | --- |
| Pod | `simple-gpu-test` |
| PID | `1515` |
| GPU | `1` |
| Run | `chexpert_vlg_gpt225_thr070_full_simplegpu_20260531T032423Z` |
| Log | `/workspace/logs/chexpert_vlg_gpt225_thr070_full_simplegpu_20260531T032423Z.log` |
| Status at last check | Complete; early stopped at epoch `8` |
| Val mean AUROC | `0.8167` |
| Val mAP | `0.5430` |

NEC metrics from post-hoc truncation of the learned final layer:

| NEC | nnz | Mean AUROC | mAP |
| ---: | ---: | ---: | ---: |
| 5 | 70 | 0.5972 | 0.3320 |
| 10 | 140 | 0.6696 | 0.3934 |
| 15 | 210 | 0.7368 | 0.4499 |
| 20 | 280 | 0.7115 | 0.4490 |
| 25 | 350 | 0.7466 | 0.4714 |
| 30 | 420 | 0.7819 | 0.5116 |

### SGCBM queries520 Full CheXpert Training

| Field | Value |
| --- | --- |
| Pod | `a100-gpu-test-v2` |
| PID | `1306` |
| GPU | `0` |
| Run dir | `/workspace/chexpert_full_runs/savlg_cbm_chexpert_2026_05_31_03_41_13` |
| Log | `/workspace/logs/chexpert_sgcbm_queries520_thr050_stream_a100v2_20260531T034054Z.log` |
| Status at last check | CBL epoch 1, `1401 / 1493` batches, about `94%`; no metrics yet |
| Runtime note | Uses on-the-fly spatial supervision; current bottleneck is dataloader JSON/PIL throughput |

### SALF-CBM queries520 Full CheXpert Training

| Field | Value |
| --- | --- |
| Pod | `simple-gpu-test` |
| PID | `2701` |
| GPU | `0` |
| Run dir | `/workspace/chexpert_full_runs/salf_cbm_chexpert_2026_05_31_03_56_13` |
| Log | `/workspace/logs/chexpert_salf_queries520_cxrclip_recompute_full_simplegpu_20260531T035608Z.log` |
| Alignment model | CXR-CLIP, `cxrclip_swint_mcc` |
| Status at last check | Computing SALF prompt-grid similarities for train, `2301 / 23879` batches, about `10%`; no metrics yet |
| Runtime note | Relaunched with `recompute_spatial_sims: true` because a stale smoke-test cache had length `9` instead of full train length `191027` |

### LF-CBM gpt225 Full CheXpert Training

| Field | Value |
| --- | --- |
| Pod | `atharv-gpu` |
| PID | `1333` |
| GPU | `0` |
| Run dir | `/workspace/chexpert_full_runs/lf_cbm_chexpert_2026_05_31_04_01_53` |
| Log | `/workspace/logs/chexpert_lf_gpt225_cxrclip_full_atharvgpu_20260531T040144Z.log` |
| Concept set | gpt225 |
| Alignment model | CXR-CLIP, `cxrclip_swint_mcc` |
| Status at last check | Failed at `937 / 10746` LF feature batches with PyTorch `Too many open files`; no metrics |

### SGCBM gpt225 Full CheXpert Training

| Field | Value |
| --- | --- |
| Pod | `atharv-gpu` |
| PID | `1334` |
| GPU | `1` |
| Run dir | `/workspace/chexpert_full_runs/savlg_cbm_chexpert_2026_05_31_04_01_54` |
| Log | `/workspace/logs/chexpert_sgcbm_gpt225_thr070_stream_atharvgpu_20260531T040144Z.log` |
| Concept set | gpt225 |
| Concept threshold | `0.70` |
| Status at last check | CBL epoch 6, `615 / 2985` batches, about `21%` of epoch 6; no metrics yet |
| Runtime note | Uses streamed spatial supervision from `/workspace/CHEX_CBM/gpt225_root` to avoid materializing a full mask cache |

## Next Results To Add

- Completed GLM/NEC lambda-sweep table for queries520 threshold-0.50.
- First full gpt225 VLG-CBM run using:
  - `/workspace/CHEX_CBM/annotations_gpt225_train_full/presence_cache_191027_225_thr070.pt`
  - `/workspace/CHEX_CBM/annotations_gpt225_valid_full/presence_cache_202_225_thr070.pt`
- Full-scale LF-CBM, SALF-CBM, and SGCBM CheXpert results from the unified entrypoint.
