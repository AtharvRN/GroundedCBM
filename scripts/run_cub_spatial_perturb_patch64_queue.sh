#!/usr/bin/env bash
set -uo pipefail

cd /workspace/GroundedCBM

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
PATCH_SIZE="${PATCH_SIZE:-64}"
OUT_ROOT="${OUT_ROOT:-/workspace/cub_rebuttal_eval_results/cub_spatial_perturb_toppatch${PATCH_SIZE}_${TS}}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
BATCH="${BATCH:-32}"
BATCH_VLG="${BATCH_VLG:-16}"
WORKERS="${WORKERS:-4}"
RANDOM_TRIALS="${RANDOM_TRIALS:-3}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" /workspace/cub_rebuttal_eval_results
echo "${OUT_ROOT}" > /workspace/cub_rebuttal_eval_results/LATEST_CUB_SPATIAL_PERTURB_PATCH64.txt

common_args=(
  --split val
  --device cuda
  --num_workers "${WORKERS}"
  --persistent_workers
  --deletion_region top_patch
  --patch_size "${PATCH_SIZE}"
  --top_blocks 1
  --top_concepts_per_image 5
  --concept_selection top_class_contribution
  --fractions 0.05
  --random_trials "${RANDOM_TRIALS}"
  --random_mask_mode box
  --skip_insertion
)

run_job() {
  local name="$1"
  shift
  local log="${LOG_ROOT}/${name}_${TS}.log"
  echo "[queue] start ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT_ROOT}/queue_status.log"
  "$@" > "${log}" 2>&1
  local status=$?
  echo "[queue] done ${name} status=${status} $(date -u +%Y-%m-%dT%H:%M:%SZ) log=${log}" | tee -a "${OUT_ROOT}/queue_status.log"
  return 0
}

run_job sgcbm_align00_seed0_nec5_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_paper_config_seed_runs/cub_sgcbm_layer4_vlglinear_align00_seed0_20260725/savlg_cbm_cub_2026_07_25_08_20_35 \
    --model_name sgcbm --nec 5 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/sgcbm_align00_seed0_nec5_top5_patch64.json"

run_job sgcbm_align00_seed1_nec5_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_paper_config_seed_runs/cub_sgcbm_layer4_vlglinear_align00_seed1_20260725/savlg_cbm_cub_2026_07_25_08_06_15 \
    --model_name sgcbm --nec 5 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/sgcbm_align00_seed1_nec5_top5_patch64.json"

run_job sgcbm_align00_seed42_nec5_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_paper_config_seed_runs/cub_sgcbm_layer4_vlglinear_align00_seed42_20260725/savlg_cbm_cub_2026_07_25_07_48_16 \
    --model_name sgcbm --nec 5 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/sgcbm_align00_seed42_nec5_top5_patch64.json"

run_job sgcbm_align00_seed123_nec5_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_paper_config_seed_runs/cub_sgcbm_layer4_vlglinear_align00_seed123_20260725/savlg_cbm_cub_2026_07_25_07_48_16 \
    --model_name sgcbm --nec 5 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/sgcbm_align00_seed123_nec5_top5_patch64.json"

run_job sgcbm_align00_seed6885_nec5_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_paper_config_seed_runs/cub_sgcbm_layer4_vlglinear_align00_seed6885_20260725/savlg_cbm_cub_2026_07_25_07_45_09 \
    --model_name sgcbm --nec 5 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/sgcbm_align00_seed6885_nec5_top5_patch64.json"

run_job vlg_official_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/cub_release \
    --model_name vlg_cbm --mask_source gradcam --batch_size "${BATCH_VLG}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/vlg_official_gradcam_top5_patch64.json"

run_job salf_seed0_nec30_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/salf_paper_config_seed_runs/cub_salf_paper_vitb16_softmax_seed0_20260725/salf_cbm_cub_2026_07_25_09_38_20 \
    --model_name salf_cbm --nec 30 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/salf_seed0_nec30_top5_patch64.json"

run_job salf_seed1_nec30_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/salf_paper_config_seed_runs/cub_salf_paper_vitb16_softmax_seed1_20260725/salf_cbm_cub_2026_07_25_09_13_58 \
    --model_name salf_cbm --nec 30 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/salf_seed1_nec30_top5_patch64.json"

run_job salf_seed42_nec30_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/salf_paper_config_seed_runs/cub_salf_paper_vitb16_softmax_seed42_20260725/salf_cbm_cub_2026_07_25_08_48_40 \
    --model_name salf_cbm --nec 30 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/salf_seed42_nec30_top5_patch64.json"

run_job salf_seed123_nec30_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/salf_paper_config_seed_runs/cub_salf_paper_vitb16_softmax_seed123_20260725/salf_cbm_cub_2026_07_25_09_23_22 \
    --model_name salf_cbm --nec 30 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/salf_seed123_nec30_top5_patch64.json"

run_job salf_seed6885_nec30_patch64 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/salf_paper_config_seed_runs/cub_salf_paper_vitb16_softmax_seed6885_20260725/salf_cbm_cub_2026_07_25_09_13_58 \
    --model_name salf_cbm --nec 30 --batch_size "${BATCH}" \
    "${common_args[@]}" \
    --output "${OUT_ROOT}/salf_seed6885_nec30_top5_patch64.json"

python - <<'PY'
import json
import statistics as st
from pathlib import Path

out_root = Path("/workspace/cub_rebuttal_eval_results/LATEST_CUB_SPATIAL_PERTURB_PATCH64.txt").read_text().strip()
out_root = Path(out_root)
keys = [
    "class_logit_drop_selected",
    "class_logit_drop_random",
    "class_prob_drop_selected",
    "class_prob_drop_random",
    "concept_logit_drop_selected",
    "concept_logit_drop_random",
    "accuracy_after_selected_sum",
    "accuracy_after_random_sum",
]
summary = {
    "out_root": str(out_root),
    "protocol": {
        "dataset": "CUB test/val split",
        "region": "top 64x64 patch",
        "concepts_per_image": 5,
        "random_trials": 3,
        "sgcbm": "align=0, NEC=5, native concept maps",
        "salf": "NEC=30, native concept maps",
        "vlgcbm": "official checkpoint, Grad-CAM maps",
    },
    "results": {},
    "aggregates": {},
}
for path in sorted(out_root.glob("*.json")):
    payload = json.loads(path.read_text())
    frac = payload.get("fractions", {}).get("0.05", {})
    item = {
        "n": payload.get("n"),
        "model_name": payload.get("model_name"),
        "nec": payload.get("nec"),
        "accuracy_before": payload.get("accuracy_before"),
    }
    for key in keys:
        item[key] = frac.get(key)
    summary["results"][path.name] = item

for prefix, label in [("sgcbm_", "sgcbm_align00_nec5"), ("salf_", "salf_nec30")]:
    rows = [value for name, value in summary["results"].items() if name.startswith(prefix)]
    agg = {"num_runs": len(rows)}
    for key in ["accuracy_before", *keys]:
        vals = [row[key] for row in rows if row.get(key) is not None]
        if vals:
            agg[key] = {
                "mean": st.mean(vals),
                "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            }
    summary["aggregates"][label] = agg

summary_path = out_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[queue] all CUB patch64 jobs finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT_ROOT}/queue_status.log"
