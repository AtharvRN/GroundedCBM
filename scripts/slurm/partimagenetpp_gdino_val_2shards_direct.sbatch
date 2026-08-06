#!/bin/bash
#SBATCH --job-name=partimgpp-gdino-val
#SBATCH --account=bfah-delta-gpu
#SBATCH --qos=bfah-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/u/anair13/partimagenetpp_logs/%x-%j.out
#SBATCH --error=/u/anair13/partimagenetpp_logs/%x-%j.err

set -euo pipefail

GDINO=${GDINO:-/u/anair13/GroundingDINO}
SCRIPT=${SCRIPT:-/u/anair13/partimagenetpp_manifest_build/gdino_annotate_part_manifest.py}
MANIFEST=${MANIFEST:-/u/anair13/partimagenetpp_val_manifest.jsonl}
CONFIG=${CONFIG:-${GDINO}/groundingdino/config/GroundingDINO_SwinT_OGC.py}
CHECKPOINT=${CHECKPOINT:-${GDINO}/weights/groundingdino_swint_ogc.pth}
BOX_THRESHOLD=${BOX_THRESHOLD:-0.1}
CHUNK_SIZE=${CHUNK_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
OVERWRITE=${OVERWRITE:-1}
FINAL_JSONL_ROOT=${FINAL_JSONL_ROOT:-/u/anair13/partimagenetpp_gdino_val_thr${BOX_THRESHOLD}_jsonl}
OUT_JSONL_ROOT=${OUT_JSONL_ROOT:-/tmp/${USER}/partimagenetpp_gdino_val_${SLURM_JOB_ID}_thr${BOX_THRESHOLD}}
NUM_SHARDS=2

mkdir -p /u/anair13/partimagenetpp_logs "${OUT_JSONL_ROOT}" "${FINAL_JSONL_ROOT}"

cd "${GDINO}"
source gdino_uv/bin/activate

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "manifest=${MANIFEST}"
echo "script=${SCRIPT}"
echo "groundingdino=${GDINO}"
echo "checkpoint=${CHECKPOINT}"
echo "box_threshold=${BOX_THRESHOLD}"
echo "chunk_size=${CHUNK_SIZE}"
echo "batch_size=${BATCH_SIZE}"
echo "overwrite=${OVERWRITE}"
echo "out_jsonl_root=${OUT_JSONL_ROOT}"
echo "final_jsonl_root=${FINAL_JSONL_ROOT}"
echo "num_shards=${NUM_SHARDS}"
nvidia-smi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

for SHARD in 0 1; do
  LOG="/u/anair13/partimagenetpp_logs/gdino_partimagenetpp_val_${SLURM_JOB_ID}_shard${SHARD}.log"
  OUT_JSONL="${OUT_JSONL_ROOT}/shard${SHARD}.jsonl"
  echo "launching shard=${SHARD} gpu=${SHARD} log=${LOG} out_jsonl=${OUT_JSONL}"
  CUDA_VISIBLE_DEVICES="${SHARD}" python -u "${SCRIPT}" \
    --manifest "${MANIFEST}" \
    --groundingdino_root "${GDINO}" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --box_threshold "${BOX_THRESHOLD}" \
    --chunk_size "${CHUNK_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --device cuda \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${SHARD}" \
    --progress_every "${PROGRESS_EVERY}" \
    --output_jsonl "${OUT_JSONL}" \
    "${OVERWRITE_ARGS[@]}" \
    > "${LOG}" 2>&1 &
done

wait
for SHARD in 0 1; do
  cp "${OUT_JSONL_ROOT}/shard${SHARD}.jsonl" "${FINAL_JSONL_ROOT}/shard${SHARD}.jsonl"
  gzip -c "${FINAL_JSONL_ROOT}/shard${SHARD}.jsonl" > "${FINAL_JSONL_ROOT}/shard${SHARD}.jsonl.gz"
done
wc -l "${FINAL_JSONL_ROOT}"/shard*.jsonl
echo "all val shards complete"
