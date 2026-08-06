#!/usr/bin/env bash
set -uo pipefail

cd /workspace/GroundedCBM

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
PATCH_SIZE="${PATCH_SIZE:-64}"
OUT_ROOT="${OUT_ROOT:-/workspace/imagenet_spatial_perturb_runs/imagenet_spatial_perturb_nec5_toppatch${PATCH_SIZE}_${TS}}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
VAL_ROOT="${VAL_ROOT:-/workspace/imagenet_val_full}"
DEVKIT_DIR="${DEVKIT_DIR:-/workspace/imagenet_devkit_labels}"
LABEL_ORDER="${LABEL_ORDER:-auto}"
WORKERS="${WORKERS:-4}"
RANDOM_TRIALS="${RANDOM_TRIALS:-3}"
BATCH_SG="${BATCH_SG:-12}"
BATCH_VLG="${BATCH_VLG:-4}"
BATCH_SALF="${BATCH_SALF:-12}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" /workspace/imagenet_spatial_perturb_runs
echo "${OUT_ROOT}" > /workspace/imagenet_spatial_perturb_runs/LATEST_IMAGENET_SPATIAL_PERTURB_PATCH64.txt

common_args=(
  --split val
  --imagenet_val_root "${VAL_ROOT}"
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
if [[ -n "${DEVKIT_DIR}" && -d "${DEVKIT_DIR}" ]]; then
  common_args+=(--imagenet_devkit_dir "${DEVKIT_DIR}" --imagenet_label_order "${LABEL_ORDER}")
fi

run_job() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="${LOG_ROOT}/${name}_${TS}.log"
  echo "[queue] start ${name} gpu=${gpu} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT_ROOT}/queue_status.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "$@" > "${log}" 2>&1
  local status=$?
  echo "[queue] done ${name} status=${status} gpu=${gpu} $(date -u +%Y-%m-%dT%H:%M:%SZ) log=${log}" | tee -a "${OUT_ROOT}/queue_status.log"
  return 0
}

run_sg_queue() {
  run_job 0 sgcbm_seed1234_nec5_patch64 \
    python scripts/eval_spatial_perturbation.py \
      --artifact_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed1234 \
      --nec_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed1234/glm_nec_sweep_lam0007_tol1e4_seed1234_20260726T173517Z \
      --model_name imagenet_sgcbm --nec 5 --batch_size "${BATCH_SG}" \
      "${common_args[@]}" \
      --output "${OUT_ROOT}/sgcbm_seed1234_nec5_top5_patch64.json"

  run_job 0 sgcbm_seed2024_nec5_patch64 \
    python scripts/eval_spatial_perturbation.py \
      --artifact_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed2024 \
      --nec_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed2024/glm_nec_sweep_lam0007_tol1e4_seed2024_20260726T204952Z \
      --model_name imagenet_sgcbm --nec 5 --batch_size "${BATCH_SG}" \
      "${common_args[@]}" \
      --output "${OUT_ROOT}/sgcbm_seed2024_nec5_top5_patch64.json"

  run_job 0 sgcbm_seed6885_nec5_patch64 \
    python scripts/eval_spatial_perturbation.py \
      --artifact_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed6885 \
      --nec_dir /workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed6885/glm_nec_sweep_lam0007_tol1e4_seed6885_20260726T191235Z \
      --model_name imagenet_sgcbm --nec 5 --batch_size "${BATCH_SG}" \
      "${common_args[@]}" \
      --output "${OUT_ROOT}/sgcbm_seed6885_nec5_top5_patch64.json"
}

run_baseline_queue() {
  run_job 1 vlg_official_nec5_patch64 \
    python scripts/eval_spatial_perturbation.py \
      --artifact_dir /workspace/saved_models/imagenet_vlgcbm_official \
      --model_name vlg_cbm --nec 5 --mask_source gradcam --batch_size "${BATCH_VLG}" \
      "${common_args[@]}" \
      --output "${OUT_ROOT}/vlg_official_nec5_gradcam_top5_patch64.json"

  run_job 1 salf_official_direct_patch64 \
    python scripts/eval_spatial_perturbation.py \
      --artifact_dir /workspace/salf-cbm_models/imagenet \
      --model_name salf_imagenet_official --batch_size "${BATCH_SALF}" \
      "${common_args[@]}" \
      --output "${OUT_ROOT}/salf_official_direct_top5_patch64.json"
}

run_sg_queue &
sg_pid=$!
run_baseline_queue &
baseline_pid=$!
wait "${sg_pid}"
wait "${baseline_pid}"

python - <<'PY'
import json
from pathlib import Path

latest = Path("/workspace/imagenet_spatial_perturb_runs/LATEST_IMAGENET_SPATIAL_PERTURB_PATCH64.txt")
out_root = Path(latest.read_text().strip())
summary = {
    "out_root": str(out_root),
    "protocol": {
        "dataset": "ImageNet val flat 50k",
        "region": "top 64x64 patch",
        "concepts_per_image": 5,
        "random_trials": 3,
        "sgcbm_nec": 5,
        "vlgcbm_nec": 5,
        "salf": "official direct dense W_g.pt/b_g.pt",
        "label_mode": "ImageNet devkit labels when DEVKIT_DIR is present; otherwise baseline_prediction",
    },
    "results": {},
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

echo "[queue] all patch64 jobs finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT_ROOT}/queue_status.log"
