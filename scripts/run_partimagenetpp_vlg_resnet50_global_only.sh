#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh || true
conda activate cbm

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PARTIMAGENETPP_TRAIN_MANIFEST="${PARTIMAGENETPP_TRAIN_MANIFEST:-/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_train_90k_manifest_pvc.jsonl}"
export PARTIMAGENETPP_VAL_MANIFEST="${PARTIMAGENETPP_VAL_MANIFEST:-/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_val_10k_manifest_pvc.jsonl}"

cd /workspace/GroundedCBM
python -m py_compile gcbm/train_imagenet.py scripts/prepare_partimagenetpp_gcbm_manifest.py scripts/eval_concept_accuracy.py

RUN_TS="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-vlg_cbm_partimagenetpp_r50v1_global_only_matched783_${RUN_TS}}"
SAVE_DIR="${SAVE_DIR:-/workspace/partimagenetpp_runs/vlg_resnet50_global_only}"
FEATURE_DIR="${FEATURE_DIR:-/workspace/partimagenetpp_runs/vlg_resnet50_global_only_features/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-/workspace/logs/partimagenetpp_vlg_resnet50_global_only_${RUN_TS}.log}"
RESULT_PATH="${RESULT_PATH:-/workspace/partimagenetpp_results/vlg_resnet50_global_only_concept_metrics_${RUN_TS}.json}"
RAW_TRAIN="${PARTIMAGENETPP_TRAIN_MANIFEST}"
RAW_VAL="${PARTIMAGENETPP_VAL_MANIFEST}"
GCBM_TRAIN="${GCBM_TRAIN:-/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_train_90k_manifest_gcbm.jsonl}"
GCBM_VAL="${GCBM_VAL:-/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_val_10k_manifest_gcbm.jsonl}"
CONCEPT_FILE="${CONCEPT_FILE:-/workspace/partimagenetpp_runs/sgcbm_imagenet_hparams/savlg_cbm_partimagenetpp_2026_07_25_01_18_40/concepts.txt}"
VLG_BATCH_SIZE="${VLG_BATCH_SIZE:-512}"
VLG_WORKERS="${VLG_WORKERS:-12}"
VLG_PREFETCH_FACTOR="${VLG_PREFETCH_FACTOR:-4}"
VLG_FEATURE_BATCH_SIZE="${VLG_FEATURE_BATCH_SIZE:-512}"
VLG_FEATURE_WORKERS="${VLG_FEATURE_WORKERS:-8}"

mkdir -p "${SAVE_DIR}" "${FEATURE_DIR}" /workspace/logs /workspace/partimagenetpp_results
if [[ "${STAGE_PINPP_TO_SHM:-0}" == "1" ]]; then
  STAGE_ROOT="${STAGE_ROOT:-/dev/shm/partimagenetpp_vlg_${RUN_TS}}"
  TRAIN_STAGE="${STAGE_ROOT}/train"
  VAL_STAGE="${STAGE_ROOT}/val"
  RAW_TRAIN_STAGED="${STAGE_ROOT}/partimagenetpp_pinpp_train_90k_manifest_shm.jsonl"
  RAW_VAL_STAGED="${STAGE_ROOT}/partimagenetpp_pinpp_val_10k_manifest_shm.jsonl"
  GCBM_TRAIN="${STAGE_ROOT}/partimagenetpp_pinpp_train_90k_manifest_gcbm.jsonl"
  GCBM_VAL="${STAGE_ROOT}/partimagenetpp_pinpp_val_10k_manifest_gcbm.jsonl"
  mkdir -p "${TRAIN_STAGE}" "${VAL_STAGE}"
  echo "[run] staging PartImageNet++ train/val images to ${STAGE_ROOT}"
  tar -xf /workspace/partimagenetpp_eval_payload/pinpp_train_images_90k.tar -C "${TRAIN_STAGE}"
  tar -xf /workspace/partimagenetpp_eval_payload/pinpp_val_images.tar -C "${VAL_STAGE}"
  python - <<PY
import json
from pathlib import Path

def rewrite(src, dst, image_root):
    src = Path(src)
    dst = Path(dst)
    root = Path(image_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            rel = row.get("file_name")
            if not rel:
                rel = "/".join(Path(row["image"]).parts[-2:])
            row["image"] = str(root / rel)
            if "path" in row:
                row["path"] = row["image"]
            out.write(json.dumps(row, sort_keys=True) + "\\n")

rewrite("${RAW_TRAIN}", "${RAW_TRAIN_STAGED}", "${TRAIN_STAGE}")
rewrite("${RAW_VAL}", "${RAW_VAL_STAGED}", "${VAL_STAGE}")
PY
  RAW_TRAIN="${RAW_TRAIN_STAGED}"
  RAW_VAL="${RAW_VAL_STAGED}"
  export PARTIMAGENETPP_TRAIN_MANIFEST="${RAW_TRAIN}"
  export PARTIMAGENETPP_VAL_MANIFEST="${RAW_VAL}"
fi
echo "${SAVE_DIR}/${RUN_NAME}" > /workspace/partimagenetpp_runs/vlg_resnet50_global_only_latest_run_dir.txt
echo "${LOG_PATH}" > /workspace/logs/partimagenetpp_vlg_resnet50_global_only_latest_log.txt
echo "${RESULT_PATH}" > /workspace/partimagenetpp_results/vlg_resnet50_global_only_latest_result.txt

echo "[run] starting PartImageNet++ VLG-CBM run=${RUN_NAME} cuda_visible=${CUDA_VISIBLE_DEVICES}"
python scripts/prepare_partimagenetpp_gcbm_manifest.py \
  --train_in "${RAW_TRAIN}" \
  --val_in "${RAW_VAL}" \
  --train_out "${GCBM_TRAIN}" \
  --val_out "${GCBM_VAL}" \
  --summary_out /workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_gcbm_manifest_summary.json

nvidia-smi
python -u gcbm/train_imagenet.py \
  --train_root / \
  --val_root / \
  --train_manifest "${GCBM_TRAIN}" \
  --val_manifest "${GCBM_VAL}" \
  --annotation_dir /workspace/partimagenetpp_eval_payload/partimagenetpp_gdino_thr0.1_splits \
  --concept_file "${CONCEPT_FILE}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --feature_dir "${FEATURE_DIR}" \
  --epochs 15 \
  --batch_size "${VLG_BATCH_SIZE}" \
  --workers "${VLG_WORKERS}" \
  --prefetch_factor "${VLG_PREFETCH_FACTOR}" \
  --persistent_workers \
  --pin_memory \
  --resnet50_weights v1 \
  --branch_arch global_only \
  --optimizer adamw \
  --lr 0.001 \
  --scheduler cosine \
  --concept_threshold 0.15 \
  --global_pos_weight 100 \
  --loss_global_w 1.0 \
  --loss_mask_w 0.0 \
  --train_final_layer_after_cbl \
  --final_layer_type sparse \
  --feature_batch_size "${VLG_FEATURE_BATCH_SIZE}" \
  --feature_workers "${VLG_FEATURE_WORKERS}" \
  --saga_lam 0.0007 \
  --saga_n_iters 100 \
  --saga_batch_size 4096 \
  --saga_workers 0 \
  --log_every 20 \
  --save_every 5

RUN_DIR="${SAVE_DIR}/${RUN_NAME}"
PARTIMAGENETPP_TRAIN_MANIFEST="${RAW_TRAIN}" \
PARTIMAGENETPP_VAL_MANIFEST="${RAW_VAL}" \
python -u scripts/eval_concept_accuracy.py \
  --dataset partimagenetpp \
  --load_paths "${RUN_DIR}" \
  --model_names vlg_cbm \
  --names vlg_resnet50_global_only \
  --partimagenetpp_train_manifest "${RAW_TRAIN}" \
  --partimagenetpp_val_manifest "${RAW_VAL}" \
  --gt_source partimagenetpp_boxes \
  --partimagenetpp_gt_boxes_jsonl /workspace/partimagenetpp_eval_payload/pinpp_val_gt_boxes_generic.jsonl \
  --output "${RESULT_PATH}" \
  --device cuda \
  --batch_size 256 \
  --num_workers 8 \
  --normalization model_default \
  --threshold 0.0 \
  --log_every 1000

echo "[run] finished PartImageNet++ VLG-CBM run=${RUN_NAME} result=${RESULT_PATH}"
