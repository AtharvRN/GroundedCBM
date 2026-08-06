# VLM-as-Judge Concept Precision Results

Date: 2026-07-26

This records the top-5 concept-presence VLM judge evaluation for CUB seed runs.

## Protocol

- Dataset subset: common 498 CUB images shared across all SG-CBM and SALF-CBM seed exports.
- Tasks: top-5 activated concept-image pairs per method/seed export, after filtering to the common image subset.
- Strict precision: `yes / total`.
- Resolved precision: `yes / (yes + no)`.
- Judge prompt: concept-presence prompt from each `judge_tasks.jsonl`.
- Raw task root on pod: `/workspace/vlm_judge_seed_runs_20260725_common498`
- Qwen2.5-VL-7B outputs: `<method_seed>/vlm_judge_qwen25vl7b.jsonl`
- Qwen2.5-VL-32B outputs: `<method_seed>/vlm_judge_qwen25vl32b.jsonl`

## Summary

| Judge | Method | Yes | No | Unsure | Total | Strict precision | Resolved precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | SALF-CBM | 6819 | 5260 | 357 | 12436 | 54.83 ± 1.63% | 56.45 ± 1.48% |
| Qwen2.5-VL-7B-Instruct | SG-CBM | 7416 | 3617 | 228 | 11261 | 65.85 ± 0.35% | 67.22 ± 0.26% |
| Qwen2.5-VL-32B-Instruct | SALF-CBM | 7483 | 4951 | 2 | 12436 | 60.17 ± 2.29% | 60.18 ± 2.29% |
| Qwen2.5-VL-32B-Instruct | SG-CBM | 8379 | 2882 | 0 | 11261 | 74.40 ± 0.69% | 74.40 ± 0.69% |

Pooled strict and resolved precision:

| Judge | Method | Strict pooled | Resolved pooled |
|---|---|---:|---:|
| Qwen2.5-VL-7B-Instruct | SALF-CBM | 6819 / 12436 = 54.83% | 6819 / 12079 = 56.45% |
| Qwen2.5-VL-7B-Instruct | SG-CBM | 7416 / 11261 = 65.86% | 7416 / 11033 = 67.22% |
| Qwen2.5-VL-32B-Instruct | SALF-CBM | 7483 / 12436 = 60.17% | 7483 / 12434 = 60.18% |
| Qwen2.5-VL-32B-Instruct | SG-CBM | 8379 / 11261 = 74.41% | 8379 / 11261 = 74.41% |

## Qwen2.5-VL-7B-Instruct

Runner failures: 0

| Run | n | yes | no | unsure | Strict precision | Resolved precision |
|---|---:|---:|---:|---:|---:|---:|
| salf_cbm_seed0 | 2487 | 1360 | 1067 | 60 | 54.68% | 56.04% |
| salf_cbm_seed1 | 2489 | 1324 | 1093 | 72 | 53.19% | 54.78% |
| salf_cbm_seed123 | 2487 | 1349 | 1041 | 97 | 54.24% | 56.44% |
| salf_cbm_seed42 | 2489 | 1356 | 1059 | 74 | 54.48% | 56.15% |
| salf_cbm_seed6885 | 2484 | 1430 | 1000 | 54 | 57.57% | 58.85% |
| sg_cbm_seed0 | 2281 | 1507 | 730 | 44 | 66.07% | 67.37% |
| sg_cbm_seed1 | 2261 | 1485 | 729 | 47 | 65.68% | 67.07% |
| sg_cbm_seed123 | 2285 | 1514 | 737 | 34 | 66.26% | 67.26% |
| sg_cbm_seed42 | 2199 | 1449 | 697 | 53 | 65.89% | 67.52% |
| sg_cbm_seed6885 | 2235 | 1461 | 724 | 50 | 65.37% | 66.86% |

## Qwen2.5-VL-32B-Instruct

Runner failures: 0

| Run | n | yes | no | unsure | Strict precision | Resolved precision |
|---|---:|---:|---:|---:|---:|---:|
| salf_cbm_seed0 | 2487 | 1505 | 982 | 0 | 60.51% | 60.51% |
| salf_cbm_seed1 | 2489 | 1471 | 1018 | 0 | 59.10% | 59.10% |
| salf_cbm_seed123 | 2487 | 1476 | 1011 | 0 | 59.35% | 59.35% |
| salf_cbm_seed42 | 2489 | 1443 | 1045 | 1 | 57.98% | 58.00% |
| salf_cbm_seed6885 | 2484 | 1588 | 895 | 1 | 63.93% | 63.95% |
| sg_cbm_seed0 | 2281 | 1706 | 575 | 0 | 74.79% | 74.79% |
| sg_cbm_seed1 | 2261 | 1668 | 593 | 0 | 73.77% | 73.77% |
| sg_cbm_seed123 | 2285 | 1721 | 564 | 0 | 75.32% | 75.32% |
| sg_cbm_seed42 | 2199 | 1637 | 562 | 0 | 74.44% | 74.44% |
| sg_cbm_seed6885 | 2235 | 1647 | 588 | 0 | 73.69% | 73.69% |

## Notes

- Qwen2.5-VL-7B produced some `unsure` responses, so resolved precision is higher than strict precision.
- Qwen2.5-VL-32B produced essentially no `unsure` responses, so strict and resolved precision are effectively identical.
- The 32B run used tensor parallelism across both A100 GPUs on `a100-gpu-test`.
- The 7B and 32B runs were written to separate JSONL files and do not overwrite each other.
