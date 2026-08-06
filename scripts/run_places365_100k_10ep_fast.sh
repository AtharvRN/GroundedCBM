#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/GroundedCBM}
PLACES_ROOT=${PLACES_ROOT:-/workspace/SAVLGCBM/datasets/places365_torch}
TRAIN_IMAGE_TAR=${TRAIN_IMAGE_TAR:-${PLACES_ROOT}/train_256_places365standard.tar}
ANNOTATION_TAR=${ANNOTATION_TAR:-/workspace/places365_annotations/places365_train.tar.gz}
FULL_MANIFEST=${FULL_MANIFEST:-/workspace/places365_manifests/places365_train_manifest.jsonl}
CONCEPT_FILE=${CONCEPT_FILE:-concept_files/places365_filtered.txt}
SUBSET_N=${SUBSET_N:-100000}
SUBSET_SEED=${SUBSET_SEED:-6885}
RUN_TS=${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}

LOCAL_ROOT=${LOCAL_ROOT:-/root/places365_100k}
LOCAL_ANN_ROOT=${LOCAL_ANN_ROOT:-/root/places365_annotations}
MANIFEST_DIR=${MANIFEST_DIR:-/workspace/places365_manifests}
RUNS_DIR=${RUNS_DIR:-/workspace/places365_runs}
PRECOMP=${PRECOMP:-${RUNS_DIR}/precomputed_targets_100k_thr015_${RUN_TS}}
RUN_ROOT=${RUN_ROOT:-${RUNS_DIR}/sgcbm_100k_10ep_${RUN_TS}}

BATCH_SIZE=${BATCH_SIZE:-512}
WORKERS=${WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
PRECOMPUTE_WORKERS=${PRECOMPUTE_WORKERS:-12}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

subset_manifest="${MANIFEST_DIR}/places365_train_${SUBSET_N}_local_manifest.jsonl"
subset_files="${MANIFEST_DIR}/places365_train_${SUBSET_N}_files.txt"
subset_archive_files="${MANIFEST_DIR}/places365_train_${SUBSET_N}_archive_files.txt"
subset_annotations="${MANIFEST_DIR}/places365_train_${SUBSET_N}_annotation_files.txt"
export FULL_MANIFEST SUBSET_N SUBSET_SEED LOCAL_ROOT subset_manifest subset_files subset_annotations

cd "${REPO_DIR}"
mkdir -p "${MANIFEST_DIR}" "${RUNS_DIR}" "${LOCAL_ROOT}/data_256" "${LOCAL_ANN_ROOT}"

echo "[job] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) run_ts=${RUN_TS}"
echo "[job] pod=$(hostname)"
echo "[job] subset_n=${SUBSET_N} subset_seed=${SUBSET_SEED} batch_size=${BATCH_SIZE} workers=${WORKERS} precompute_workers=${PRECOMPUTE_WORKERS}"

echo "[job] stage=prepare_subset start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - <<'PY'
import collections
import json
import os
import random
from pathlib import Path

src = Path(os.environ["FULL_MANIFEST"])
out_manifest = Path(os.environ["subset_manifest"])
out_files = Path(os.environ["subset_files"])
out_annotations = Path(os.environ["subset_annotations"])
subset_n = int(os.environ["SUBSET_N"])
subset_seed = int(os.environ["SUBSET_SEED"])
root_prefix = "/root/places365_torch/data_256/"
local_prefix = str(Path(os.environ["LOCAL_ROOT"]) / "data_256") + "/"

rows_by_class = collections.defaultdict(list)
with src.open() as fin:
    for line in fin:
        row = json.loads(line)
        rows_by_class[int(row["class_id"])].append(row)

classes = sorted(rows_by_class)
if not classes:
    raise RuntimeError(f"no classes found in manifest: {src}")

rng = random.Random(subset_seed)
per_class = subset_n // len(classes)
remainder = subset_n % len(classes)
selected = []
for pos, class_id in enumerate(classes):
    rows = list(rows_by_class[class_id])
    rng.shuffle(rows)
    want = per_class + (1 if pos < remainder else 0)
    selected.extend(rows[:want])

if len(selected) < subset_n:
    selected_ids = {id(row) for row in selected}
    extras = [row for class_id in classes for row in rows_by_class[class_id] if id(row) not in selected_ids]
    rng.shuffle(extras)
    selected.extend(extras[: subset_n - len(selected)])

selected = selected[:subset_n]
rng.shuffle(selected)
class_counts = collections.Counter(int(row["class_id"]) for row in selected)
if subset_n >= len(classes) and len(class_counts) != len(classes):
    raise RuntimeError(
        f"stratified subset only covered {len(class_counts)}/{len(classes)} classes; "
        f"subset_n={subset_n}"
    )

with src.open() as fin, out_manifest.open("w") as mf, out_files.open("w") as ff, out_annotations.open("w") as af:
    for count, row in enumerate(selected, start=1):
        path = str(row["path"])
        rel = path.split(root_prefix, 1)[1] if root_prefix in path else path.lstrip("/")
        row["path"] = local_prefix + rel
        row["sample_index"] = count - 1
        row["annotation_index"] = int(row.get("annotation_index", count - 1))
        mf.write(json.dumps(row) + "\n")
        ff.write(rel + "\n")
        af.write(f"places365_train/{row['annotation_index']}.json\n")
print(json.dumps({
    "manifest": str(out_manifest),
    "n": len(selected),
    "num_classes": len(class_counts),
    "min_class_count": min(class_counts.values()),
    "max_class_count": max(class_counts.values()),
    "files": str(out_files),
    "annotations": str(out_annotations),
}, indent=2))
PY

echo "[job] stage=stage_images start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage_images_start=$(date +%s)
existing_images=0
if [[ -d "${LOCAL_ROOT}/data_256" ]]; then
  existing_images=$(find "${LOCAL_ROOT}/data_256" -type f -name '*.jpg' | wc -l)
fi
if [[ "${existing_images}" -ge "${SUBSET_N}" ]]; then
  echo "[job] stage_images=skip existing_image_count=${existing_images}"
elif [[ -f "${TRAIN_IMAGE_TAR}" ]]; then
  rm -rf "${LOCAL_ROOT}/data_256"
  mkdir -p "${LOCAL_ROOT}"
  sed 's#^#data_256/#' "${subset_files}" > "${subset_archive_files}"
  tar -xf "${TRAIN_IMAGE_TAR}" -C "${LOCAL_ROOT}" -T "${subset_archive_files}"
else
  rm -rf "${LOCAL_ROOT}/data_256"
  mkdir -p "${LOCAL_ROOT}/data_256"
  tar -C "${PLACES_ROOT}/data_256" -cf - -T "${subset_files}" | tar -C "${LOCAL_ROOT}/data_256" -xf -
fi
echo "[timing] stage_images_sec=$(( $(date +%s) - stage_images_start ))"

echo "[job] stage=stage_annotations start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rm -rf "${LOCAL_ANN_ROOT}/places365_train"
mkdir -p "${LOCAL_ANN_ROOT}"
stage_annotations_start=$(date +%s)
tar -xzf "${ANNOTATION_TAR}" -C "${LOCAL_ANN_ROOT}" -T "${subset_annotations}"
echo "[timing] stage_annotations_sec=$(( $(date +%s) - stage_annotations_start ))"

echo "[job] staged_image_count=$(find "${LOCAL_ROOT}/data_256" -type f -name '*.jpg' | wc -l)"
echo "[job] staged_annotation_count=$(find "${LOCAL_ANN_ROOT}/places365_train" -type f -name '*.json' | wc -l)"

echo "[job] stage=precompute_targets start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rm -rf "${PRECOMP}"
precompute_start=$(date +%s)
GCBM_PRECOMPUTE_WORKERS="${PRECOMPUTE_WORKERS}" \
GCBM_PRECOMPUTE_CHUNK_SIZE=64 \
PYTHONNOUSERSITE=1 python -u scripts/precompute_imagenet_targets.py \
  --image_root "${LOCAL_ROOT}/data_256" \
  --manifest "${subset_manifest}" \
  --annotation_dir "${LOCAL_ANN_ROOT}" \
  --concept_file "${CONCEPT_FILE}" \
  --output_dir "${PRECOMP}" \
  --split train \
  --concept_threshold 0.15 \
  --spatial_target_mode soft_box \
  --mask_h 14 \
  --mask_w 14 \
  --input_size 224
echo "[timing] precompute_sec=$(( $(date +%s) - precompute_start ))"

echo "[job] stage=train start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
train_start=$(date +%s)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONNOUSERSITE=1 python -u gcbm/train_imagenet.py \
  --train_root "${LOCAL_ROOT}/data_256" \
  --train_manifest "${PRECOMP}/train_manifest.jsonl" \
  --annotation_dir "${LOCAL_ANN_ROOT}" \
  --precomputed_target_dir "${PRECOMP}" \
  --concept_file "${PRECOMP}/concepts.txt" \
  --save_dir "${RUN_ROOT}" \
  --run_name places365_sgcbm_100k_10ep_fast \
  --epochs 10 \
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
  --log_every 25 \
  --val_split 0.0
echo "[timing] train_sec=$(( $(date +%s) - train_start ))"
echo "[job] done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
