#!/usr/bin/env bash
set -uo pipefail

cd /workspace/GroundedCBM

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/workspace/imagenet_spatial_perturb_runs/imagenet_spatial_perturb_nec5_toppatch32_${TS}}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
VAL_ROOT="${VAL_ROOT:-/workspace/imagenet_val_full}"
DEVKIT_DIR="${DEVKIT_DIR:-/workspace/imagenet_devkit_labels}"
LABEL_ORDER="${LABEL_ORDER:-auto}"
DEVICE="${DEVICE:-cuda}"
BATCH_SG="${BATCH_SG:-16}"
BATCH_VLG="${BATCH_VLG:-4}"
BATCH_SALF="${BATCH_SALF:-16}"
WORKERS="${WORKERS:-4}"
RANDOM_TRIALS="${RANDOM_TRIALS:-3}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
echo "${OUT_ROOT}" > /workspace/imagenet_spatial_perturb_runs/LATEST_IMAGENET_SPATIAL_PERTURB.txt

COMMON_ARGS=(
  --split val
  --imagenet_val_root "${VAL_ROOT}"
  --device "${DEVICE}"
  --num_workers "${WORKERS}"
  --persistent_workers
  --deletion_region top_patch
  --patch_size 32
  --top_blocks 1
  --top_concepts_per_image 5
  --concept_selection top_class_contribution
  --fractions 0.05
  --random_trials "${RANDOM_TRIALS}"
  --random_mask_mode box
  --skip_insertion
)
if [[ -n "${DEVKIT_DIR}" && -d "${DEVKIT_DIR}" ]]; then
  COMMON_ARGS+=(--imagenet_devkit_dir "${DEVKIT_DIR}" --imagenet_label_order "${LABEL_ORDER}")
fi

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

run_job sgcbm_seed1234_nec5 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed1234 \
    --nec_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed1234/glm_nec_sweep_lam0007_tol1e4_seed1234_20260726T173517Z \
    --model_name imagenet_sgcbm --nec 5 --batch_size "${BATCH_SG}" \
    "${COMMON_ARGS[@]}" \
    --output "${OUT_ROOT}/sgcbm_seed1234_nec5_top5_patch32.json"

run_job sgcbm_seed2024_nec5 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed2024 \
    --nec_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed2024/glm_nec_sweep_lam0007_tol1e4_seed2024_20260726T204952Z \
    --model_name imagenet_sgcbm --nec 5 --batch_size "${BATCH_SG}" \
    "${COMMON_ARGS[@]}" \
    --output "${OUT_ROOT}/sgcbm_seed2024_nec5_top5_patch32.json"

run_job sgcbm_seed6885_nec5 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed6885 \
    --nec_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed6885/glm_nec_sweep_lam0007_tol1e4_seed6885_20260726T191235Z \
    --model_name imagenet_sgcbm --nec 5 --batch_size "${BATCH_SG}" \
    "${COMMON_ARGS[@]}" \
    --output "${OUT_ROOT}/sgcbm_seed6885_nec5_top5_patch32.json"

run_job vlg_official_nec5 \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/saved_models/imagenet_vlgcbm_official \
    --model_name vlg_cbm --nec 5 --mask_source gradcam --batch_size "${BATCH_VLG}" \
    "${COMMON_ARGS[@]}" \
    --output "${OUT_ROOT}/vlg_official_nec5_gradcam_top5_patch32.json"

run_job salf_official_direct \
  python scripts/eval_spatial_perturbation.py \
    --artifact_dir /workspace/salf-cbm_models/imagenet \
    --model_name salf_imagenet_official --batch_size "${BATCH_SALF}" \
    "${COMMON_ARGS[@]}" \
    --output "${OUT_ROOT}/salf_official_direct_top5_patch32.json"

python - <<'PY'
import json
from pathlib import Path

latest = Path("/workspace/imagenet_spatial_perturb_runs/LATEST_IMAGENET_SPATIAL_PERTURB.txt")
out_root = Path(latest.read_text().strip())
summary = {
    "out_root": str(out_root),
    "protocol": {
        "dataset": "ImageNet val flat 50k",
        "region": "top 32x32 patch",
        "concepts_per_image": 5,
        "random_trials": 3,
        "sgcbm_nec": 5,
        "vlgcbm_nec": 5,
        "salf": "official direct dense W_g.pt/b_g.pt",
        "label_mode": "ImageNet devkit labels when DEVKIT_DIR is present; otherwise baseline_prediction",
    },
    "results": {},
    "missing_sg_seeds": ["only seeds 1234, 2024, and 6885 were present on a100-gpu-test-v3"],
}
for path in sorted(out_root.glob("*.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        summary["results"][path.name] = {"error": repr(exc)}
        continue
    frac = payload.get("fractions", {}).get("0.05", {})
    summary["results"][path.name] = {
        "n": payload.get("n"),
        "model_name": payload.get("model_name"),
        "nec": payload.get("nec"),
        "class_logit_drop_selected": frac.get("class_logit_drop_selected"),
        "class_logit_drop_random": frac.get("class_logit_drop_random"),
        "class_prob_drop_selected": frac.get("class_prob_drop_selected"),
        "class_prob_drop_random": frac.get("class_prob_drop_random"),
        "concept_logit_drop_selected": frac.get("concept_logit_drop_selected"),
        "concept_logit_drop_random": frac.get("concept_logit_drop_random"),
        "prediction_retention_selected": frac.get("accuracy_after_selected_sum"),
        "prediction_retention_random": frac.get("accuracy_after_random_sum"),
    }
summary_path = out_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[queue] all queued jobs finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT_ROOT}/queue_status.log"
