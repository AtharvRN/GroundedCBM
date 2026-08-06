#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh || true
conda activate cbm

cd /workspace/GroundedCBM

RUN_DIR="${RUN_DIR:-$(cat /workspace/partimagenetpp_runs/vlg_resnet50_global_only_latest_run_dir.txt)}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_train_90k_manifest_pvc.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_val_10k_manifest_pvc.jsonl}"
GT_BOXES="${GT_BOXES:-/workspace/partimagenetpp_eval_payload/pinpp_val_gt_boxes_generic.jsonl}"
RESULT_DIR="${RESULT_DIR:-/workspace/partimagenetpp_results}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONCEPT_OUT="${CONCEPT_OUT:-${RESULT_DIR}/vlg_resnet50_global_only_humanbox_concept_metrics_${RUN_TAG}.json}"
LOC_OUT="${LOC_OUT:-${RESULT_DIR}/vlg_resnet50_global_only_cam_gtbox_localization_${RUN_TAG}.json}"
LOG_PATH="${LOG_PATH:-/workspace/logs/partimagenetpp_vlg_resnet50_global_only_eval_${RUN_TAG}.log}"

mkdir -p "${RESULT_DIR}" "$(dirname "${LOG_PATH}")"
echo "${LOG_PATH}" > /workspace/logs/partimagenetpp_vlg_resnet50_global_only_latest_eval_log.txt

echo "[watch-eval] waiting for VLG training to finish for RUN_DIR=${RUN_DIR}"
while pgrep -af "gcbm/train_imagenet.py.*$(basename "${RUN_DIR}")" >/dev/null; do
  date -u "+[watch-eval] %Y-%m-%dT%H:%M:%SZ training still running"
  sleep 300
done

if [[ ! -s "${RUN_DIR}/concept_head_best.pt" ]]; then
  echo "[watch-eval] missing concept_head_best.pt at ${RUN_DIR}; not evaluating" >&2
  exit 2
fi
if [[ ! -s "${RUN_DIR}/concepts.txt" ]]; then
  echo "[watch-eval] missing concepts.txt at ${RUN_DIR}; not evaluating" >&2
  exit 2
fi

export PARTIMAGENETPP_TRAIN_MANIFEST="${TRAIN_MANIFEST}"
export PARTIMAGENETPP_VAL_MANIFEST="${VAL_MANIFEST}"
export PARTIMAGENETPP_GT_BOXES_JSONL="${GT_BOXES}"

echo "[watch-eval] running VLG concept metrics with human box-derived GT -> ${CONCEPT_OUT}"
python -u scripts/eval_concept_accuracy.py \
  --dataset partimagenetpp \
  --load_paths "${RUN_DIR}" \
  --model_names vlg_cbm \
  --names vlg_resnet50_global_only \
  --partimagenetpp_train_manifest "${TRAIN_MANIFEST}" \
  --partimagenetpp_val_manifest "${VAL_MANIFEST}" \
  --gt_source partimagenetpp_boxes \
  --partimagenetpp_gt_boxes_jsonl "${GT_BOXES}" \
  --output "${CONCEPT_OUT}" \
  --device cuda \
  --batch_size 256 \
  --num_workers 4 \
  --normalization model_default \
  --threshold 0.0 \
  --log_every 1000

echo "[watch-eval] finished concept_out=${CONCEPT_OUT}"

echo "[watch-eval] running VLG CAM-style GT-box localization -> ${LOC_OUT}"
python -u scripts/eval_partimagenetpp_gtbox_localization.py \
  --gcbm_path "${RUN_DIR}" \
  --model_name vlg_cbm \
  --train_manifest "${TRAIN_MANIFEST}" \
  --val_manifest "${VAL_MANIFEST}" \
  --gt_boxes_jsonl "${GT_BOXES}" \
  --output "${LOC_OUT}" \
  --device cuda \
  --batch_size 256 \
  --num_workers 4 \
  --map_normalization concept_zscore_minmax \
  --log_every 1000

echo "[watch-eval] finished loc_out=${LOC_OUT}"
