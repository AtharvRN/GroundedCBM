#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/GroundedCBM}
PLACES_ROOT=${PLACES_ROOT:-/workspace/SAVLGCBM/datasets/places365_torch}
TRAIN_IMAGE_TAR=${TRAIN_IMAGE_TAR:-${PLACES_ROOT}/train_256_places365standard.tar}
ANNOTATION_TAR=${ANNOTATION_TAR:-/workspace/places365_annotations/places365_train.tar.gz}
FULL_TRAIN_MANIFEST=${FULL_TRAIN_MANIFEST:-/workspace/places365_manifests/places365_train_manifest.jsonl}
FULL_VAL_MANIFEST=${FULL_VAL_MANIFEST:-/workspace/places365_manifests/places365_val_manifest.jsonl}
CONCEPT_FILE=${CONCEPT_FILE:-concept_files/places365_filtered.txt}
SOURCE_PRECOMP_ROOT=${SOURCE_PRECOMP_ROOT:-/workspace/places365_runs/precomputed_targets_full_thr015_20260725T045040Z}
SUBSET_N=${SUBSET_N:-200000}
SUBSET_SEED=${SUBSET_SEED:-6885}
EPOCHS=${EPOCHS:-5}
RUN_TS=${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}

LOCAL_ROOT=${LOCAL_ROOT:-/root/places365_${SUBSET_N}}
LOCAL_ANN_ROOT=${LOCAL_ANN_ROOT:-/root/places365_annotations_${SUBSET_N}}
LOCAL_TRAIN_ROOT=${LOCAL_ROOT}/data_256
LOCAL_VAL_ROOT=${LOCAL_VAL_ROOT:-${LOCAL_ROOT}/val_256}
MANIFEST_DIR=${MANIFEST_DIR:-/workspace/places365_manifests}
REMOTE_RUNS_DIR=${RUNS_DIR:-/workspace/places365_runs}
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-/root/places365_outputs}
PRECOMP=${PRECOMP:-${LOCAL_OUTPUT_ROOT}/precomputed_targets_${SUBSET_N}_thr015_${RUN_TS}}
RUN_ROOT=${RUN_ROOT:-${LOCAL_OUTPUT_ROOT}/sgcbm_${SUBSET_N}_${EPOCHS}ep_${RUN_TS}}
REMOTE_PRECOMP=${REMOTE_PRECOMP:-${REMOTE_RUNS_DIR}/precomputed_targets_${SUBSET_N}_thr015_${RUN_TS}}
REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT:-${REMOTE_RUNS_DIR}/sgcbm_${SUBSET_N}_${EPOCHS}ep_${RUN_TS}}
RUN_NAME=${RUN_NAME:-places365_sgcbm_${SUBSET_N}_${EPOCHS}ep}

BATCH_SIZE=${BATCH_SIZE:-512}
WORKERS=${WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PRECOMPUTE_WORKERS=${PRECOMPUTE_WORKERS:-16}
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

subset_manifest="${MANIFEST_DIR}/places365_train_${SUBSET_N}_local_${RUN_TS}.jsonl"
subset_files="${MANIFEST_DIR}/places365_train_${SUBSET_N}_files_${RUN_TS}.txt"
subset_archive_files="${MANIFEST_DIR}/places365_train_${SUBSET_N}_archive_files_${RUN_TS}.txt"
subset_annotations="${MANIFEST_DIR}/places365_train_${SUBSET_N}_annotation_files_${RUN_TS}.txt"
local_val_manifest="${MANIFEST_DIR}/places365_val_local_${RUN_TS}.jsonl"
val_root_prefix="/root/places365_torch/val_256/val_256/"
train_root_prefix="/root/places365_torch/data_256/"

export FULL_TRAIN_MANIFEST FULL_VAL_MANIFEST SUBSET_N SUBSET_SEED
export LOCAL_TRAIN_ROOT LOCAL_VAL_ROOT train_root_prefix val_root_prefix
export subset_manifest subset_files subset_annotations local_val_manifest

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
  if [[ -d "${PRECOMP}" ]]; then
    mkdir -p "${REMOTE_PRECOMP}"
    cp -a "${PRECOMP}/." "${REMOTE_PRECOMP}/"
    echo "[job] synced precomputed targets to ${REMOTE_PRECOMP}"
  fi
  echo "[job] sync_outputs done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "${status}"
}
trap sync_outputs EXIT

echo "[job] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) run_ts=${RUN_TS}"
echo "[job] pod=$(hostname)"
echo "[job] subset_n=${SUBSET_N} epochs=${EPOCHS} batch_size=${BATCH_SIZE} workers=${WORKERS} precompute_workers=${PRECOMPUTE_WORKERS}"
echo "[job] source_precomp_root=${SOURCE_PRECOMP_ROOT}"
echo "[job] local_run_root=${RUN_ROOT}/${RUN_NAME}"
echo "[job] remote_run_root=${REMOTE_RUN_ROOT}/${RUN_NAME}"
df -h /root /workspace || true

echo "[job] stage=prepare_subset start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - <<'PY'
import collections
import json
import os
import random
from pathlib import Path

src = Path(os.environ["FULL_TRAIN_MANIFEST"])
out_manifest = Path(os.environ["subset_manifest"])
out_files = Path(os.environ["subset_files"])
out_annotations = Path(os.environ["subset_annotations"])
subset_n = int(os.environ["SUBSET_N"])
subset_seed = int(os.environ["SUBSET_SEED"])
root_prefix = os.environ["train_root_prefix"]
local_prefix = str(Path(os.environ["LOCAL_TRAIN_ROOT"])) + "/"

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
    selected_keys = {(row["path"], row.get("annotation_index")) for row in selected}
    extras = [
        row
        for class_id in classes
        for row in rows_by_class[class_id]
        if (row["path"], row.get("annotation_index")) not in selected_keys
    ]
    rng.shuffle(extras)
    selected.extend(extras[: subset_n - len(selected)])

selected = selected[:subset_n]
rng.shuffle(selected)
class_counts = collections.Counter(int(row["class_id"]) for row in selected)
if subset_n >= len(classes) and len(class_counts) != len(classes):
    raise RuntimeError(f"subset covered {len(class_counts)}/{len(classes)} classes")

with out_manifest.open("w") as mf, out_files.open("w") as ff, out_annotations.open("w") as af:
    for count, row in enumerate(selected, start=1):
        path = str(row["path"])
        rel = path.split(root_prefix, 1)[1] if root_prefix in path else path.lstrip("/")
        if "data_256/" in rel:
            rel = rel.split("data_256/", 1)[1]
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
}, indent=2))
PY

echo "[job] stage=stage_train_images start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stage_images_start=$(date +%s)
rm -rf "${LOCAL_TRAIN_ROOT}"
mkdir -p "${LOCAL_ROOT}"
sed 's#^#data_256/#' "${subset_files}" > "${subset_archive_files}"
tar -xf "${TRAIN_IMAGE_TAR}" -C "${LOCAL_ROOT}" -T "${subset_archive_files}"
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
if [[ -d "${SOURCE_PRECOMP_ROOT}" && -f "${SOURCE_PRECOMP_ROOT}/train/global_targets.npy" ]]; then
  echo "[job] stage_annotations=skip reason=using_source_precompute"
else
  rm -rf "${LOCAL_ANN_ROOT}/places365_train"
  mkdir -p "${LOCAL_ANN_ROOT}"
  tar -xzf "${ANNOTATION_TAR}" -C "${LOCAL_ANN_ROOT}" -T "${subset_annotations}"
fi
echo "[timing] stage_annotations_sec=$(( $(date +%s) - stage_annotations_start ))"

echo "[job] staged_train_image_count=$(find "${LOCAL_TRAIN_ROOT}" -type f -name '*.jpg' | wc -l)"
echo "[job] staged_val_root=${LOCAL_VAL_ROOT}"
echo "[job] staged_val_image_count=$(find "${LOCAL_VAL_ROOT}" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
if [[ -d "${LOCAL_ANN_ROOT}/places365_train" ]]; then
  echo "[job] staged_annotation_count=$(find "${LOCAL_ANN_ROOT}/places365_train" -type f -name '*.json' | wc -l)"
else
  echo "[job] staged_annotation_count=0"
fi

echo "[job] stage=write_val_manifest start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["FULL_VAL_MANIFEST"])
dst = Path(os.environ["local_val_manifest"])
src_prefix = os.environ["val_root_prefix"]
dst_root = Path(os.environ["LOCAL_VAL_ROOT"])
with src.open() as fin, dst.open("w") as fout:
    for n, line in enumerate(fin, start=1):
        row = json.loads(line)
        path = str(row["path"])
        if src_prefix in path:
            rel = path.split(src_prefix, 1)[1]
        else:
            rel = path.lstrip("/")
            if "val_256/val_256/" in rel:
                rel = rel.split("val_256/val_256/", 1)[1]
        row["path"] = str(dst_root / rel)
        row["sample_index"] = n - 1
        fout.write(json.dumps(row) + "\n")
print(json.dumps({"val_manifest": str(dst), "val_n": n}, indent=2))
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
rm -rf "${PRECOMP}"
if [[ -d "${SOURCE_PRECOMP_ROOT}" && -f "${SOURCE_PRECOMP_ROOT}/train/global_targets.npy" ]]; then
  SOURCE_PRECOMP_ROOT="${SOURCE_PRECOMP_ROOT}" PRECOMP="${PRECOMP}" subset_manifest="${subset_manifest}" \
  PYTHONNOUSERSITE=1 python - <<'PY'
import json
import os
import shutil
from pathlib import Path

import numpy as np

source = Path(os.environ["SOURCE_PRECOMP_ROOT"])
out = Path(os.environ["PRECOMP"])
manifest = Path(os.environ["subset_manifest"])
out_train = out / "train"
out_train.mkdir(parents=True, exist_ok=True)

rows = []
indices = []
with manifest.open() as fin:
    for line in fin:
        row = json.loads(line)
        rows.append(row)
        indices.append(int(row["annotation_index"]))
indices_np = np.asarray(indices, dtype=np.int64)

src_train = source / "train"
src_offsets = np.load(src_train / "offsets.npy", mmap_mode="r")
starts = src_offsets[indices_np]
ends = src_offsets[indices_np + 1]
lengths = ends - starts
out_offsets = np.empty(len(indices) + 1, dtype=np.int64)
out_offsets[0] = 0
np.cumsum(lengths, out=out_offsets[1:])
total_entries = int(out_offsets[-1])

src_global = np.load(src_train / "global_targets.npy", mmap_mode="r")
src_concept_ids = np.load(src_train / "concept_ids.npy", mmap_mode="r")
src_masks = np.load(src_train / "mask_targets.npy", mmap_mode="r")

global_out = np.lib.format.open_memmap(
    out_train / "global_targets.npy",
    mode="w+",
    dtype=src_global.dtype,
    shape=(len(indices), src_global.shape[1]),
)
chunk = 8192
for lo in range(0, len(indices), chunk):
    hi = min(len(indices), lo + chunk)
    global_out[lo:hi] = src_global[indices_np[lo:hi]]
global_out.flush()

concept_out = np.lib.format.open_memmap(
    out_train / "concept_ids.npy",
    mode="w+",
    dtype=src_concept_ids.dtype,
    shape=(total_entries,),
)
mask_out = np.lib.format.open_memmap(
    out_train / "mask_targets.npy",
    mode="w+",
    dtype=src_masks.dtype,
    shape=(total_entries,) + tuple(src_masks.shape[1:]),
)
for i, (start, end) in enumerate(zip(starts, ends)):
    dst_start = out_offsets[i]
    dst_end = out_offsets[i + 1]
    if dst_end == dst_start:
        continue
    concept_out[dst_start:dst_end] = src_concept_ids[start:end]
    mask_out[dst_start:dst_end] = src_masks[start:end]
concept_out.flush()
mask_out.flush()
np.save(out_train / "offsets.npy", out_offsets)

for name in ("concepts.txt",):
    shutil.copy2(source / name, out / name)
with manifest.open() as fin, (out / "train_manifest.jsonl").open("w") as fout:
    shutil.copyfileobj(fin, fout)

metadata = json.loads((src_train / "metadata.json").read_text()) if (src_train / "metadata.json").is_file() else {}
metadata.update(
    {
        "split": "train",
        "n_examples": len(indices),
        "n_concepts": int(src_global.shape[1]),
        "total_entries": total_entries,
        "global_targets_path": str(out_train / "global_targets.npy"),
        "offsets_path": str(out_train / "offsets.npy"),
        "concept_ids_path": str(out_train / "concept_ids.npy"),
        "mask_targets_path": str(out_train / "mask_targets.npy"),
        "training_manifest_path": str(out / "train_manifest.jsonl"),
        "source_precomputed_root": str(source),
        "source_manifest_path": str(manifest),
    }
)
(out_train / "metadata.json").write_text(json.dumps(metadata, indent=2))
summary = {
    "mode": "subset_from_precomputed_targets",
    "source_precomputed_root": str(source),
    "manifest": str(manifest),
    "output_root": str(out),
    "n_examples": len(indices),
    "n_concepts": int(src_global.shape[1]),
    "total_entries": total_entries,
    "mask_h": int(src_masks.shape[1]),
    "mask_w": int(src_masks.shape[2]),
}
(out / "precompute_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
PY
else
  GCBM_PRECOMPUTE_WORKERS="${PRECOMPUTE_WORKERS}" \
  GCBM_PRECOMPUTE_CHUNK_SIZE=64 \
  PYTHONNOUSERSITE=1 python -u scripts/precompute_imagenet_targets.py \
    --image_root "${LOCAL_TRAIN_ROOT}" \
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
  --val_split 0.0
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
