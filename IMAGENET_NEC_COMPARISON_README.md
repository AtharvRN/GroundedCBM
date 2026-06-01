# ImageNet NEC Accuracy Comparison

This note summarizes the recent ImageNet sparse-head/NEC evaluation for SG-CBM and the global-only VLG-CBM baseline.

## Runs

| Model | Training run | Epochs | Concepts | Sparse sweep |
|---|---:|---:|---:|---|
| SG-CBM | `sgcbm_imagenet_full_unified_15ep_a100_r2_20260514T014833Z_sgcbm-imagenet-full-unified-15ep-a100-r2-rhwhg` | 15 | 4309 | `lam_max=0.0007`, `tol=1e-4`, `table_device=cuda` |
| VLG-CBM | `vlg_cbm_imagenet_v1_global_only_20ep_a100_r6` | 20 | 4729 | `lam_max=0.0007`, `tol=1e-4`, `table_device=cuda` |

Both evaluations use 50,000 ImageNet validation images and VLG-style NEC truncation, so `NEC=k` keeps approximately `k * 1000` nonzero final-layer weights.

## Summary

`ACC@5` is top-1 accuracy at `NEC=5`. `AVGACC` is the mean top-1 accuracy over NEC values `{5, 10, 15, 20, 25, 30}`.

| Model | ACC@5 | AVGACC |
|---|---:|---:|
| SG-CBM | 74.13 | 74.64 |
| VLG-CBM | 74.06 | 74.62 |

## Accuracy by NEC

| NEC | SG-CBM Top-1 | SG-CBM Top-5 | VLG-CBM Top-1 | VLG-CBM Top-5 |
|---:|---:|---:|---:|---:|
| 5 | 74.13 | 91.45 | 74.06 | 91.42 |
| 10 | 74.59 | 91.76 | 74.52 | 91.67 |
| 15 | 74.65 | 91.84 | 74.68 | 91.77 |
| 20 | 74.77 | 91.93 | 74.80 | 91.87 |
| 25 | 74.83 | 92.00 | 74.81 | 91.88 |
| 30 | 74.86 | 92.03 | 74.85 | 91.93 |

## Source Artifacts

SG-CBM:

```text
/workspace/logs/sgcbm_imagenet_nec50k_eval_lam0007_tol1e4_gpu0_table_20260515T184648Z.log
/workspace/sgcbm_imagenet_runs/sgcbm_imagenet_full_unified_15ep_a100_r2_glm_nec_sweep_lam0007_tol1e4_gpu0_table_20260515T184648Z
```

VLG-CBM:

```text
/workspace/sgcbm_imagenet_runs/vlg_cbm_imagenet_v1_global_only_20ep_a100_r6/glm_path_sweep_lam0p0007_k150_tol1e4/nec_eval_val50k.json
```

