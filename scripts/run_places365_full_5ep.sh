#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/GroundedCBM}
PLACES_ROOT=${PLACES_ROOT:-/workspace/SAVLGCBM/datasets/places365_torch}
TRAIN_IMAGE_TAR=${TRAIN_IMAGE_TAR:-${PLACES_ROOT}/train_256_places365standard.tar}
ANNOTATION_TAR=${ANNOTATION_TAR:-/workspace/places365_annotations/places365_train.tar.gz}
FULL_TRAIN_MANIFEST=${FULL_TRAIN_MANIFEST:-/workspace/places365_manifests/places365_train_manifest.jsonl}
FULL_VAL_MANIFEST=${FULL_VAL_MANIFEST:-/workspace/places365_manifests/places365_val_manifest.jsonl}
CONCEPT_FILE=${CONCEPT_FILE:-concept_files/places365_filtered.txt}
RUN_TS=${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}

LOCAL_ROOT=${LOCAL_ROOT:-/root/places365_full}
LOCAL_ANN_ROOT=${LOCAL_ANN_ROOT:-/root/places365_annotations_full}
LOCAL_TRAIN_ROOT=${LOCAL_ROOT}/data_256
LOCAL_VAL_ROOT=${LOCAL_VAL_ROOT:-${LOCAL_ROOT}/val_256}
MANIFEST_DIR=${MANIFEST_DIR:-/workspace/places365_manifests}
REMOTE_RUNS_DIR=${RUNS_DIR:-/workspace/places365_runs}
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-/root/places365_outputs}
PRECOMP=${PRECOMP:-${LOCAL_OUTPUT_ROOT}/precomputed_targets_full_thr015_${RUN_TS}}
SOURCE_PRECOMP_ROOT=${SOURCE_PRECOMP_ROOT:-}
PRECOMP_SOURCE_USED=0
RUN_ROOT=${RUN_ROOT:-${LOCAL_OUTPUT_ROOT}/sgcbm_full_5ep_${RUN_TS}}
REMOTE_PRECOMP=${REMOTE_PRECOMP:-${REMOTE_RUNS_DIR}/precomputed_targets_full_thr015_${RUN_TS}}
REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT:-${REMOTE_RUNS_DIR}/sgcbm_full_5ep_${RUN_TS}}
RUN_NAME=${RUN_NAME:-places365_sgcbm_full_5ep}

BATCH_SIZE=${BATCH_SIZE:-512}
SEED=${SEED:-6885}
EPOCHS=${EPOCHS:-5}
WORKERS=${WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PRECOMPUTE_WORKERS=${PRECOMPUTE_WORKERS:-24}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

RUN_FINAL_EVAL=${RUN_FINAL_EVAL:-1}
FINAL_LAYER_TYPE=${FINAL_LAYER_TYPE:-sparse}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-512}
FEATURE_WORKERS=${FEATURE_WORKERS:-8}
SAGA_BATCH_SIZE=${SAGA_BATCH_SIZE:-512}
SAGA_LAM=${SAGA_LAM:-0.0007}
SAGA_N_ITERS=${SAGA_N_ITERS:-100}
SAGA_STEP_SIZE=${SAGA_STEP_SIZE:-0.1}
SAGA_TABLE_DEVICE=${SAGA_TABLE_DEVICE:-cuda}
DENSE_N_ITERS=${DENSE_N_ITERS:-20}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

local_train_manifest="${MANIFEST_DIR}/places365_train_full_local_${RUN_TS}.jsonl"
local_val_manifest="${MANIFEST_DIR}/places365_val_local_${RUN_TS}.jsonl"
val_root_prefix="/root/places365_torch/val_256/val_256/"
train_root_prefix="/root/places365_torch/data_256/"

export FULL_TRAIN_MANIFEST FULL_VAL_MANIFEST local_train_manifest local_val_manifest
export LOCAL_TRAIN_ROOT LOCAL_VAL_ROOT train_root_prefix val_root_prefix

cd "${REPO_DIR}"
mkdir -p "${MANIFEST_DIR}" "${REMOTE_RUNS_DIR}" "${LOCAL_ROOT}" "${LOCAL_ANN_ROOT}" "${LOCAL_OUTPUT_ROOT}" /workspace/logs

sync_outputs() {
  status=$?
  set +e
  echo "[job] sync_outputs start status=${status} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -d "${RUN_ROOT}" ]]; then
    mkdir -p "${REMOTE_RUN_ROOT}"
    cp -a "${RUN_ROOT}/." "${REMOTE_RUN_ROOT}/"
    echo "[job] synced run outputs to ${REMOTE_RUN_ROOT}"
  fi
  if [[ -d "${PRECOMP}" && "${PRECOMP_SOURCE_USED}" != "1" ]]; then
    mkdir -p "${REMOTE_PRECOMP}"
    cp -a "${PRECOMP}/." "${REMOTE_PRECOMP}/"
    echo "[job] synced precomputed targets to ${REMOTE_PRECOMP}"
  elif [[ "${PRECOMP_SOURCE_USED}" == "1" ]]; then
    echo "[job] skipped precompute target sync because source cache was reused: ${SOURCE_PRECOMP_ROOT}"
  fi
  echo "[job] sync_outputs done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "${status}"
}
trap sync_outputs EXIT

echo "[job] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) run_ts=${RUN_TS}"
echo "[job] pod=$(hostname)"
echo "[job] local_root=${LOCAL_ROOT}"
echo "[job] local_run_root=${RUN_ROOT}/${RUN_NAME}"
echo "[job] remote_run_root=${REMOTE_RUN_ROOT}/${RUN_NAME}"
echo "[job] batch_size=${BATCH_SIZE} workers=${WORKERS} precompute_workers=${PRECOMPUTE_WORKERS}"
echo "[job] seed=${SEED} epochs=${EPOCHS} resume_checkpoint=${RESUME_CHECKPOINT:-none}"
echo "[job] final_eval=${RUN_FINAL_EVAL} final_layer_type=${FINAL_LAYER_TYPE}"
df -h /root /workspace || true

echo "[job] stage=stage_train_images start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage_images_start=$(date +%s)
if [[ -f "${LOCAL_ROOT}/.train_images_complete" ]]; then
  echo "[job] stage_train_images=skip marker=${LOCAL_ROOT}/.train_images_complete"
else
  rm -rf "${LOCAL_TRAIN_ROOT}"
  mkdir -p "${LOCAL_ROOT}"
  tar -xf "${TRAIN_IMAGE_TAR}" -C "${LOCAL_ROOT}"
  touch "${LOCAL_ROOT}/.train_images_complete"
fi
echo "[timing] stage_train_images_sec=$(( $(date +%s) - stage_images_start ))"

echo "[job] stage=stage_val_images start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage_val_start=$(date +%s)
rm -rf "${LOCAL_ROOT}/val_256"
mkdir -p "${LOCAL_ROOT}"
if [[ -f "${PLACES_ROOT}/val_256.tar" ]]; then
  tar -xf "${PLACES_ROOT}/val_256.tar" -C "${LOCAL_ROOT}"
else
  tar -C "${PLACES_ROOT}" -cf - val_256 | tar -C "${LOCAL_ROOT}" -xf -
fi
if [[ -d "${LOCAL_ROOT}/val_256/val_256" ]]; then
  LOCAL_VAL_ROOT="${LOCAL_ROOT}/val_256/val_256"
else
  LOCAL_VAL_ROOT="${LOCAL_ROOT}/val_256"
fi
export LOCAL_VAL_ROOT
echo "[timing] stage_val_images_sec=$(( $(date +%s) - stage_val_start ))"

echo "[job] stage=stage_annotations start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage_annotations_start=$(date +%s)
if [[ -f "${LOCAL_ANN_ROOT}/.annotations_complete" ]]; then
  echo "[job] stage_annotations=skip marker=${LOCAL_ANN_ROOT}/.annotations_complete"
else
  rm -rf "${LOCAL_ANN_ROOT}/places365_train"
  mkdir -p "${LOCAL_ANN_ROOT}"
  tar -xzf "${ANNOTATION_TAR}" -C "${LOCAL_ANN_ROOT}"
  touch "${LOCAL_ANN_ROOT}/.annotations_complete"
fi
echo "[timing] stage_annotations_sec=$(( $(date +%s) - stage_annotations_start ))"

echo "[job] staged_train_image_count=$(find "${LOCAL_TRAIN_ROOT}" -type f -name '*.jpg' | wc -l)"
echo "[job] staged_val_root=${LOCAL_VAL_ROOT}"
echo "[job] staged_val_image_count=$(find "${LOCAL_VAL_ROOT}" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
echo "[job] staged_annotation_count=$(find "${LOCAL_ANN_ROOT}/places365_train" -type f -name '*.json' | wc -l)"

echo "[job] stage=write_local_manifests start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - <<'PY'
import json
import os
from pathlib import Path

def rewrite_manifest(src_env, dst_env, src_prefix_env, dst_root_env):
    src = Path(os.environ[src_env])
    dst = Path(os.environ[dst_env])
    src_prefix = os.environ[src_prefix_env]
    dst_root = Path(os.environ[dst_root_env])
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for n, line in enumerate(fin, start=1):
            row = json.loads(line)
            path = str(row["path"])
            if src_prefix in path:
                rel = path.split(src_prefix, 1)[1]
            else:
                rel = path.lstrip("/")
                for marker in ("data_256/", "val_256/val_256/", "val_256/"):
                    if marker in rel:
                        rel = rel.split(marker, 1)[1]
                        break
            row["path"] = str(dst_root / rel)
            row["sample_index"] = n - 1
            row["annotation_index"] = int(row.get("annotation_index", n - 1))
            fout.write(json.dumps(row) + "\n")
    return n

train_n = rewrite_manifest("FULL_TRAIN_MANIFEST", "local_train_manifest", "train_root_prefix", "LOCAL_TRAIN_ROOT")
val_n = rewrite_manifest("FULL_VAL_MANIFEST", "local_val_manifest", "val_root_prefix", "LOCAL_VAL_ROOT")
print(json.dumps({"train_manifest": os.environ["local_train_manifest"], "train_n": train_n, "val_manifest": os.environ["local_val_manifest"], "val_n": val_n}, indent=2))
PY

echo "[job] stage=validate_val_manifest start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["local_val_manifest"])
missing = []
n = 0
with manifest.open() as handle:
    for line in handle:
        row = json.loads(line)
        n += 1
        if not Path(row["path"]).is_file():
            missing.append(row["path"])
            if len(missing) >= 10:
                break
if missing:
    raise FileNotFoundError(f"{len(missing)} sampled val paths are missing; first={missing[0]}")
if n != 36500:
    raise RuntimeError(f"Expected 36500 Places365 val rows, found {n}")
print(json.dumps({"val_manifest": str(manifest), "validated_rows": n, "val_root": os.environ["LOCAL_VAL_ROOT"]}, indent=2))
PY

echo "[job] stage=precompute_targets start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
precompute_start=$(date +%s)
if [[ -f "${PRECOMP}/concepts.txt" && -f "${PRECOMP}/train_manifest.jsonl" && -f "${PRECOMP}/train/global_targets.npy" && -f "${PRECOMP}/train/mask_targets.npy" ]]; then
  echo "[job] precompute_targets=skip existing=${PRECOMP}"
elif [[ -n "${SOURCE_PRECOMP_ROOT}" && -f "${SOURCE_PRECOMP_ROOT}/concepts.txt" && -f "${SOURCE_PRECOMP_ROOT}/train_manifest.jsonl" && -f "${SOURCE_PRECOMP_ROOT}/train/global_targets.npy" && -f "${SOURCE_PRECOMP_ROOT}/train/mask_targets.npy" ]]; then
  echo "[job] precompute_targets=copy source=${SOURCE_PRECOMP_ROOT} dest=${PRECOMP}"
  rm -rf "${PRECOMP}"
  mkdir -p "${PRECOMP}"
  cp -a "${SOURCE_PRECOMP_ROOT}/." "${PRECOMP}/"
  PRECOMP_SOURCE_USED=1
else
  rm -rf "${PRECOMP}"
  GCBM_PRECOMPUTE_WORKERS="${PRECOMPUTE_WORKERS}" \
  GCBM_PRECOMPUTE_CHUNK_SIZE=64 \
  PYTHONNOUSERSITE=1 python -u scripts/precompute_imagenet_targets.py \
    --image_root "${LOCAL_TRAIN_ROOT}" \
    --manifest "${local_train_manifest}" \
    --annotation_dir "${LOCAL_ANN_ROOT}" \
    --concept_file "${CONCEPT_FILE}" \
    --output_dir "${PRECOMP}" \
    --split train \
    --concept_threshold 0.15 \
    --spatial_target_mode soft_box \
    --mask_h 14 \
    --mask_w 14 \
    --input_size 224
fi
echo "[timing] precompute_sec=$(( $(date +%s) - precompute_start ))"

echo "[job] stage=train_cbl start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
train_start=$(date +%s)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONNOUSERSITE=1 python -u gcbm/train_imagenet.py \
  --train_root "${LOCAL_TRAIN_ROOT}" \
  --train_manifest "${PRECOMP}/train_manifest.jsonl" \
  --annotation_dir "${LOCAL_ANN_ROOT}" \
  --precomputed_target_dir "${PRECOMP}" \
  --concept_file "${PRECOMP}/concepts.txt" \
  --save_dir "${RUN_ROOT}" \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --persistent_workers \
  --pin_memory \
  --resnet50_weights v1 \
  --amp fp16 \
  --channels_last \
  --tf32 \
  --cudnn_benchmark \
  --input_size 224 \
  --mask_h 14 \
  --mask_w 14 \
  --spatial_target_mode soft_box \
  --branch_arch dual \
  --spatial_branch_mode multiscale_conv45 \
  --spatial_stage conv5 \
  --residual_alpha 0.1 \
  --residual_spatial_pooling lse \
  --global_pos_weight 100 \
  --loss_global_w 1 \
  --loss_mask_w 1 \
  --concept_threshold 0.15 \
  --optimizer adamw \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --scheduler cosine \
  --eval_every 0 \
  --save_every 1 \
  --log_every 50 \
  --val_split 0.0 \
  ${RESUME_CHECKPOINT:+--resume_checkpoint "${RESUME_CHECKPOINT}"}
echo "[timing] train_cbl_sec=$(( $(date +%s) - train_start ))"

if [[ "${RUN_FINAL_EVAL}" == "1" ]]; then
  echo "[job] stage=final_val_eval start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  final_eval_start=$(date +%s)
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTHONNOUSERSITE=1 python -u scripts/eval_places365_sgcbm_staged.py \
    --run_dir "${RUN_ROOT}/${RUN_NAME}" \
    --train_manifest "${PRECOMP}/train_manifest.jsonl" \
    --val_manifest "${local_val_manifest}" \
    --val_root "${LOCAL_VAL_ROOT}" \
    --output_dir "${RUN_ROOT}/${RUN_NAME}/official_val_${FINAL_LAYER_TYPE}_${RUN_TS}" \
    --final_layer_type "${FINAL_LAYER_TYPE}" \
    --feature_batch_size "${FEATURE_BATCH_SIZE}" \
    --feature_workers "${FEATURE_WORKERS}" \
    --feature_prefetch_factor 2 \
    --saga_batch_size "${SAGA_BATCH_SIZE}" \
    --saga_lam "${SAGA_LAM}" \
    --saga_n_iters "${SAGA_N_ITERS}" \
    --saga_step_size "${SAGA_STEP_SIZE}" \
    --saga_table_device "${SAGA_TABLE_DEVICE}" \
    --dense_n_iters "${DENSE_N_ITERS}" \
    --log_every 20
  echo "[timing] final_val_eval_sec=$(( $(date +%s) - final_eval_start ))"
fi

echo "[job] run_root=${RUN_ROOT}/${RUN_NAME}"
echo "[job] remote_run_root=${REMOTE_RUN_ROOT}/${RUN_NAME}"
du -sh "${RUN_ROOT}" || true
echo "[job] done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
