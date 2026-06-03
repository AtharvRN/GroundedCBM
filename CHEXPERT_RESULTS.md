# CheXpert Results Tracker

Last updated: 2026-06-03.

This file tracks CheXpert integration, annotation/cache status, completed quantitative results, and in-progress runs. Metrics are multilabel unless otherwise noted: mean AUROC and mAP/mean AP are the primary metrics.

## Concept-Set Comparison

This section is the first place to update when a new concept set or NEC sweep finishes. It compares concept sets at fixed NEC values using mean AUROC and mAP.

### Best Overall Per Concept Set

| Concept set | Method / source | Train split | Concept threshold | Final layer / sweep | Best reported point | Mean AUROC | mAP | Status |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.50 | Natural GLM-SAGA lambda path | NEC@50 | 0.7964 | 0.5572 | Complete |
| queries520 | Integrated LF-CBM, CXR-CLIP cut0 | Full CheXpert | N/A | Natural GLM-SAGA lambda path | NEC@50 | 0.8099 | 0.5004 | Complete |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.50 | One GLM-SAGA solution, post-hoc NEC truncation | NEC@30 | 0.7731 | 0.5582 | Complete |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.70 | One GLM-SAGA solution, post-hoc NEC truncation | NEC@30 | 0.7673 | 0.5134 | Complete |
| queries520 | Old Medical_CBM LF-CBM | Full CheXpert | 0.25 cutoff in old LF setup | Upstream sparse/NEC | NEC@30 | 0.7930 | 0.5393 | Reference only |
| queries520 | Old Medical_CBM LF-CBM | 25k CheXpert | 0.25 cutoff in old LF setup | Upstream sparse/NEC | NEC@30 | 0.8087 | 0.5056 | Reference only |
| gpt225 | Integrated VLG-CBM | Full CheXpert | 0.70 | Natural GLM-SAGA lambda path | NEC@10 | 0.8371 | 0.5496 | Complete |
| gpt225 | Integrated VLG-CBM | Full CheXpert | 0.70 | One GLM-SAGA solution, post-hoc NEC truncation | Dense/SAGA final | 0.8167 | 0.5430 | Complete |
| gpt225 | Integrated LF-CBM, CXR-CLIP cut0 | Full CheXpert | N/A | Natural GLM-SAGA lambda path | NEC@50 | 0.8097 | 0.4995 | Complete |

### AUROC By NEC

| Concept set / run | NEC@5 | NEC@10 | NEC@15 | NEC@20 | NEC@25 | NEC@30 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| queries520, integrated VLG, thr0.50 | 0.7947 | 0.7961 | 0.7952 | 0.7943 | 0.7951 | 0.7962 | Natural GLM-SAGA lambda path |
| queries520, integrated LF-CBM, CXR-CLIP cut0 | 0.6963 | 0.7451 | 0.7796 | 0.7933 | 0.7989 | 0.8025 | Natural GLM-SAGA lambda path from LF concept cache |
| queries520, integrated VLG, thr0.50 | 0.6190 | 0.6248 | 0.6295 | 0.6975 | 0.7504 | 0.7731 | One SAGA solution, post-hoc truncation |
| queries520, integrated VLG, thr0.70 | 0.6067 | 0.6610 | 0.6372 | 0.6697 | 0.7253 | 0.7673 | One SAGA solution, post-hoc truncation |
| queries520, old Medical_CBM LF full | TBD | TBD | TBD | TBD | TBD | 0.7930 | Only NEC@30 currently recorded |
| queries520, old Medical_CBM LF 25k | TBD | TBD | TBD | TBD | TBD | 0.8087 | Only NEC@30 currently recorded |
| gpt225, integrated VLG, thr0.70 | 0.7922 | 0.8371 | 0.8350 | 0.8305 | 0.8274 | 0.8228 | Natural GLM-SAGA lambda path |
| gpt225, integrated VLG, thr0.70 | 0.5972 | 0.6696 | 0.7368 | 0.7115 | 0.7466 | 0.7819 | One SAGA solution, post-hoc truncation |
| gpt225, integrated LF-CBM, CXR-CLIP cut0 | 0.6927 | 0.7414 | 0.7750 | 0.7882 | 0.7961 | 0.8016 | Natural GLM-SAGA lambda path from LF concept cache |

### mAP By NEC

| Concept set / run | NEC@5 | NEC@10 | NEC@15 | NEC@20 | NEC@25 | NEC@30 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| queries520, integrated VLG, thr0.50 | 0.5364 | 0.5519 | 0.5508 | 0.5511 | 0.5557 | 0.5603 | Natural GLM-SAGA lambda path |
| queries520, integrated LF-CBM, CXR-CLIP cut0 | 0.3870 | 0.4304 | 0.4571 | 0.4753 | 0.4838 | 0.4891 | Natural GLM-SAGA lambda path from LF concept cache |
| queries520, integrated VLG, thr0.50 | 0.3788 | 0.4475 | 0.4505 | 0.4902 | 0.5414 | 0.5582 | One SAGA solution, post-hoc truncation |
| queries520, integrated VLG, thr0.70 | 0.3535 | 0.4157 | 0.4118 | 0.4387 | 0.4651 | 0.5134 | One SAGA solution, post-hoc truncation |
| queries520, old Medical_CBM LF full | TBD | TBD | TBD | TBD | TBD | 0.5393 | Only NEC@30 currently recorded |
| queries520, old Medical_CBM LF 25k | TBD | TBD | TBD | TBD | TBD | 0.5056 | Only NEC@30 currently recorded |
| gpt225, integrated VLG, thr0.70 | 0.5409 | 0.5496 | 0.5470 | 0.5465 | 0.5467 | 0.5479 | Natural GLM-SAGA lambda path |
| gpt225, integrated VLG, thr0.70 | 0.3320 | 0.3934 | 0.4499 | 0.4490 | 0.4714 | 0.5116 | One SAGA solution, post-hoc truncation |
| gpt225, integrated LF-CBM, CXR-CLIP cut0 | 0.3795 | 0.4288 | 0.4550 | 0.4702 | 0.4796 | 0.4874 | Natural GLM-SAGA lambda path from LF concept cache |

### Dense / Untruncated Final-Layer Comparison

| Concept set | Method / source | Train split | Concept threshold | Mean AUROC | mAP | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.50 | 0.7388 | 0.4810 | Sparse/SAGA final layer before NEC truncation; `5358/7280` nnz |
| queries520 | Integrated LF-CBM, CXR-CLIP cut0 | Full CheXpert | N/A | 0.7688 | 0.4507 | Dense/saved LF final-layer validation metric before separate GLM/NEC sweep |
| queries520 | Integrated VLG-CBM | Full CheXpert | 0.70 | 0.7339 | 0.4922 | Sparse/SAGA final layer |
| queries520 | Old Medical_CBM LF-CBM | Full CheXpert | 0.25 cutoff in old LF setup | 0.7471 | 0.4965 | Reference only |
| queries520 | Old Medical_CBM LF-CBM | 25k CheXpert | 0.25 cutoff in old LF setup | 0.8176 | 0.5301 | Reference only, not directly comparable to full-data runs |
| gpt225 | Integrated VLG-CBM | Full CheXpert | 0.70 | 0.8167 | 0.5430 | Sparse/SAGA final layer before NEC truncation; filtered `225 -> 223` |
| gpt225 | Integrated LF-CBM, CXR-CLIP cut0 | Full CheXpert | N/A | 0.7851 | 0.4730 | Dense/saved LF final-layer validation metric before separate GLM/NEC sweep |

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

### Natural GLM/NEC Lambda Sweep, gpt225, threshold 0.70

| Field | Value |
| --- | --- |
| Source run | `chexpert_vlg_gpt225_thr070_full_simplegpu_20260531T032423Z` |
| Output dir | `/workspace/chexpert_full_runs/chexpert_vlg_gpt225_thr070_full_simplegpu_20260531T032423Z/medical_glm_nec_natural_lamauto_eps0005_k50_it80_20260603T023008Z` |
| Log | `/workspace/logs/medical_glm_nec_vlg_gpt225_natural_lamauto_eps0005_k50_it80_20260603T023008Z.log` |
| Selection | Natural GLM-SAGA sparse path first; truncation only as fallback |
| Runtime | `508.4` seconds |

Preferred NEC metrics from naturally sparse GLM path:

| NEC | nnz | Mean AUROC | mAP |
| ---: | ---: | ---: | ---: |
| 5 | 72 | 0.7922 | 0.5409 |
| 10 | 138 | 0.8371 | 0.5496 |
| 15 | 217 | 0.8350 | 0.5470 |
| 20 | 290 | 0.8305 | 0.5465 |
| 25 | 355 | 0.8274 | 0.5467 |
| 30 | 411 | 0.8228 | 0.5479 |
| 40 | 538 | 0.8156 | 0.5442 |
| 50 | 620 | 0.8144 | 0.5426 |

### Integrated GroundedCBM LF-CBM, queries520, CXR-CLIP cut0

| Field | Value |
| --- | --- |
| Run dir | `/workspace/chexpert_full_runs/lf_queries520_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_50_00` |
| Training log | `/workspace/logs/chexpert_lf_queries520_cut000_workers8_20260601T184941Z.log` |
| Concept set | queries520 |
| Alignment model | CXR-CLIP, `cxrclip_swint_mcc` |
| Clip cutoff | `0.0` |
| Backbone | CheXpert-pretrained DenseNet121 |
| Val mean AUROC | `0.7688` |
| Val mAP | `0.4507` |
| Test mean AUROC | `0.7666` |
| Test mAP | `0.4928` |
| Concept cache train | `/workspace/chexpert_full_runs/lf_queries520_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_50_00/concept_cache_train.pt` |
| Concept cache valid | `/workspace/chexpert_full_runs/lf_queries520_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_50_00/concept_cache_valid.pt` |
| GLM/NEC output | `/workspace/chexpert_full_runs/lf_queries520_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_50_00/medical_glm_nec_lf_lamauto_eps0005_k50_it80` |
| GLM runtime | `263.5` seconds |

Preferred NEC metrics from naturally sparse GLM path:

| NEC | nnz | Mean AUROC | mAP |
| ---: | ---: | ---: | ---: |
| 5 | 69 | 0.6963 | 0.3870 |
| 10 | 135 | 0.7451 | 0.4304 |
| 15 | 211 | 0.7796 | 0.4571 |
| 20 | 285 | 0.7933 | 0.4753 |
| 25 | 351 | 0.7989 | 0.4838 |
| 30 | 434 | 0.8025 | 0.4891 |
| 40 | 558 | 0.8077 | 0.4968 |
| 50 | 636 | 0.8099 | 0.5004 |

### Integrated GroundedCBM LF-CBM, gpt225, CXR-CLIP cut0

| Field | Value |
| --- | --- |
| Run dir | `/workspace/chexpert_full_runs/lf_gpt225_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_49_59` |
| Training log | `/workspace/logs/chexpert_lf_gpt225_cut000_workers8_20260601T184940Z.log` |
| Concept set | gpt225 |
| Alignment model | CXR-CLIP, `cxrclip_swint_mcc` |
| Clip cutoff | `0.0` |
| Backbone | CheXpert-pretrained DenseNet121 |
| Val mean AUROC | `0.7851` |
| Val mAP | `0.4730` |
| Test mean AUROC | `0.7871` |
| Test mAP | `0.5200` |
| Concept cache train | `/workspace/chexpert_full_runs/lf_gpt225_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_49_59/concept_cache_train.pt` |
| Concept cache valid | `/workspace/chexpert_full_runs/lf_gpt225_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_49_59/concept_cache_valid.pt` |
| GLM/NEC output | `/workspace/chexpert_full_runs/lf_gpt225_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_49_59/medical_glm_nec_lf_lamauto_eps0005_k50_it80` |
| GLM runtime | `293.9` seconds |

Preferred NEC metrics from naturally sparse GLM path:

| NEC | nnz | Mean AUROC | mAP |
| ---: | ---: | ---: | ---: |
| 5 | 70 | 0.6927 | 0.3795 |
| 10 | 134 | 0.7414 | 0.4288 |
| 15 | 215 | 0.7750 | 0.4550 |
| 20 | 276 | 0.7882 | 0.4702 |
| 25 | 348 | 0.7961 | 0.4796 |
| 30 | 408 | 0.8016 | 0.4874 |
| 40 | 566 | 0.8097 | 0.4995 |
| 50 | 566 | 0.8097 | 0.4995 |

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

## Current Status And Pending Runs

Snapshot time: 2026-06-03.

### Running / Queued

| Job / pod | Status | Purpose | Notes |
| --- | --- | --- | --- |
| `chexpert-salf-queries520-cut000-from-cache-r4` | Running | SALF-CBM queries520 cut0 | CBL/SAGA completed; final evaluation was running at last check |
| `chexpert-salf-gpt225-cut000-ulimit-r2` | Running | SALF-CBM gpt225 cut0 | Prompt-grid cache generation was about 75% complete at last check |
| `simple-gpu-test` | Running | Interactive A10 pod | Used to complete VLG-CBM gpt225 natural GLM/NEC sweep |

### Completed

| Run | Status | Result artifact |
| --- | --- | --- |
| VLG-CBM queries520 threshold 0.50 | Complete | `/workspace/chexpert_full_runs/chexpert_vlg_queries520_thr050_freq001_095_simplegpu_20260530T020815Z` |
| VLG-CBM queries520 threshold 0.50 natural GLM/NEC | Complete | `/workspace/chexpert_full_runs/chexpert_vlg_queries520_thr050_freq001_095_simplegpu_20260530T020815Z/medical_glm_nec_natural_lamauto_eps0005_k50_it80_20260531T033124Z` |
| VLG-CBM queries520 threshold 0.70 | Complete | `/workspace/sgcbm_medical_runs/chexpert_vlg_queries520_nec_from_cbl_cached_a100_resume_20260528T072102Z` |
| VLG-CBM gpt225 threshold 0.70 | Complete | `/workspace/logs/chexpert_vlg_gpt225_thr070_full_simplegpu_20260531T032423Z.log` |
| VLG-CBM gpt225 threshold 0.70 natural GLM/NEC | Complete | `/workspace/chexpert_full_runs/chexpert_vlg_gpt225_thr070_full_simplegpu_20260531T032423Z/medical_glm_nec_natural_lamauto_eps0005_k50_it80_20260603T023008Z/glm_nec_sweep_metrics.json` |
| LF-CBM queries520 CXR-CLIP cut0 | Complete | `/workspace/chexpert_full_runs/lf_queries520_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_50_00` |
| LF-CBM queries520 CXR-CLIP cut0 GLM/NEC | Complete | `/workspace/chexpert_full_runs/lf_queries520_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_50_00/medical_glm_nec_lf_lamauto_eps0005_k50_it80/glm_nec_sweep_metrics.json` |
| LF-CBM gpt225 CXR-CLIP cut0 | Complete | `/workspace/chexpert_full_runs/lf_gpt225_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_49_59` |
| LF-CBM gpt225 CXR-CLIP cut0 GLM/NEC | Complete | `/workspace/chexpert_full_runs/lf_gpt225_cxrclip_cut000_workers8_job/lf_cbm_chexpert_2026_06_01_18_49_59/medical_glm_nec_lf_lamauto_eps0005_k50_it80/glm_nec_sweep_metrics.json` |
| SGCBM queries520 threshold 0.50 train + GLM/NEC | Complete | `/workspace/chexpert_full_runs/sgcbm_queries520_conv45_thr050_a100_memmap_r5_tuned_job/savlg_cbm_chexpert_2026_06_02_02_38_32/medical_glm_nec_sg_lamauto_eps002_k24_it60_a100/glm_nec_sweep_metrics.json` |
| SGCBM gpt225 threshold 0.70 train + GLM/NEC | Complete | `/workspace/chexpert_full_runs/sgcbm_gpt225_thr070_a100_memmap_r2_tuned_job/savlg_cbm_chexpert_2026_06_02_08_03_59/medical_glm_nec_sg_lamauto_eps002_k24_it60_a100/glm_nec_sweep_metrics.json` |
| queries520 SG target precompute | Complete | `/workspace/CHEX_CBM/precomputed_targets/chexpert_queries520_thr050_softbox_mh14_mw14` |
| gpt225 annotations and threshold 0.70 presence caches | Complete | `/workspace/CHEX_CBM/annotations_gpt225_train_full`, `/workspace/CHEX_CBM/annotations_gpt225_valid_full` |

### Failed / Superseded

| Job | Status | Action |
| --- | --- | --- |
| `chexpert-salf-queries520-cut000-from-cache` | Failed with `Too many open files` during concept extraction | Superseded by `chexpert-salf-queries520-cut000-ulimit-r2` |
| `chexpert-salf-queries520-cut025-from-cache` | Failed | Superseded; cutoff 0.0 is the current SALF setting |
| `chexpert-salf-queries520-train-from-shards` | Failed | Superseded by from-cache relaunch |
| `chexpert-sgcbm-gpt225-glm-nec-sweep` | OOMKilled during cache extraction | Needs relaunch as high-memory job or reuse compatible cache if produced later |
| `chexpert-sgcbm-queries520-conv45-full` / `r2` | Failed | Superseded by `r3` |
| `chexpert-lf-queries520-full` / `chexpert-lf-gpt225-full` | Failed older LF attempts | Superseded by completed `cut000-workers8` jobs |
| `chexpert-gpt225-annotations-train-full` | Failed old monolithic annotation job | Superseded by completed sharded annotation jobs |

### Needs To Run

| Priority | Run | Why |
| ---: | --- | --- |
| 1 | SALF-CBM queries520 cut0 evaluation completion | Training reached final eval; wait for final metrics and then run GLM/NEC sweep |
| 2 | SALF-CBM gpt225 cut0 full run | Prompt-grid cache generation is still running; needed for both concept-set comparison |
| 3 | GLM/NEC sweeps for completed SALF runs | Run after SALF model metrics complete |
| 4 | Results cleanup | Delete or archive stale failed jobs once artifacts are no longer needed |
