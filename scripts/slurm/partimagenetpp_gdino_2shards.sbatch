#!/bin/bash
#SBATCH --job-name=partimgpp-gdino-2shards
#SBATCH --account=bfah-delta-gpu
#SBATCH --qos=bfah-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts/logs/%x-%j.out
#SBATCH --error=/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts/logs/%x-%j.err

set -euo pipefail

EXP=${EXP:-/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts}
GDINO=${GDINO:-/u/anair13/GroundingDINO}
MANIFEST=${MANIFEST:-${EXP}/manifest.jsonl}
CONFIG=${CONFIG:-${GDINO}/groundingdino/config/GroundingDINO_SwinT_OGC.py}
CHECKPOINT=${CHECKPOINT:-${GDINO}/weights/groundingdino_swint_ogc.pth}
BOX_THRESHOLD=${BOX_THRESHOLD:-0.1}
CHUNK_SIZE=${CHUNK_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
OVERWRITE=${OVERWRITE:-0}
FINAL_JSONL_ROOT=${FINAL_JSONL_ROOT:-${EXP}/gdino_annotations_thr${BOX_THRESHOLD}_jsonl}
OUT_JSONL_ROOT=${OUT_JSONL_ROOT:-/tmp/${USER}/partimagenetpp_gdino_${SLURM_JOB_ID}_thr${BOX_THRESHOLD}}
NUM_SHARDS=2

mkdir -p "${EXP}/logs"
mkdir -p "${OUT_JSONL_ROOT}"
mkdir -p "${FINAL_JSONL_ROOT}"

cd "${GDINO}"
source gdino_uv/bin/activate

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "manifest=${MANIFEST}"
echo "exp=${EXP}"
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

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

for SHARD in 0 1; do
  LOG="${EXP}/logs/gdino_partimagenetpp_${SLURM_JOB_ID}_shard${SHARD}.log"
  OUT_JSONL="${OUT_JSONL_ROOT}/shard${SHARD}.jsonl"
  echo "launching shard=${SHARD} gpu=${SHARD} log=${LOG} out_jsonl=${OUT_JSONL}"
  CUDA_VISIBLE_DEVICES="${SHARD}" python -u "${EXP}/scripts/gdino_annotate_part_manifest.py" \
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
done
echo "all shards complete"
