#!/bin/bash
#SBATCH --job-name=partimgpp-salf-cache
#SBATCH --account=bfah-delta-gpu
#SBATCH --qos=bfah-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts/logs/%x-%j.out
#SBATCH --error=/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts/logs/%x-%j.err

set -euo pipefail

REPO_SRC=${REPO_SRC:-/projects/bfah/anair13/cbm_data/partimagenetpp_salf_repo/GroundedCBM}
REPO_RUN=${REPO_RUN:-/tmp/${USER}/groundedcbm_partimagenetpp_salf_cache_${SLURM_JOB_ID}}
EXP=${EXP:-/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts}
MANIFEST=${MANIFEST:-${EXP}/manifest.jsonl}
CONCEPT_SET=${CONCEPT_SET:-concept_files/partimagenetpp_generic_parts.txt}
ACTIVATION_DIR=${ACTIVATION_DIR:-${EXP}/salf_cache}

GRID_H=${GRID_H:-7}
GRID_W=${GRID_W:-7}
PROMPT_RADIUS=${PROMPT_RADIUS:-32}
SPATIAL_BATCH_SIZE=${SPATIAL_BATCH_SIZE:-32}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-8}
VAL_SPLIT=${VAL_SPLIT:-0.1}

export PARTIMAGENETPP_MANIFEST="${MANIFEST}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

mkdir -p "${REPO_RUN}" "${ACTIVATION_DIR}" "${EXP}/logs"
rsync -a --exclude '__pycache__' --exclude 'saved_activations' --exclude 'artifacts' "${REPO_SRC}/" "${REPO_RUN}/"

cd "${REPO_RUN}"
source /u/anair13/GroundingDINO/gdino_uv/bin/activate

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo_run=${REPO_RUN}"
echo "manifest=${PARTIMAGENETPP_MANIFEST}"
echo "activation_dir=${ACTIVATION_DIR}"
echo "grid=${GRID_H}x${GRID_W}"
echo "prompt_radius=${PROMPT_RADIUS}"
echo "spatial_batch_size=${SPATIAL_BATCH_SIZE}"
echo "prompt_batch_size=${PROMPT_BATCH_SIZE}"
nvidia-smi

for SPLIT in train val; do
  echo "[salf-cache] starting split=${SPLIT}"
  for SHARD in 0 1; do
    (
      export CUDA_VISIBLE_DEVICES="${SHARD}"
      python -u scripts/precompute_salf_prompt_grid.py \
        --dataset partimagenetpp \
        --concept_set "${CONCEPT_SET}" \
        --backbone clip_RN50 \
        --lf_clip_name clip_RN50 \
        --activation_dir "${ACTIVATION_DIR}" \
        --device cuda \
        --seed 42 \
        --val_split "${VAL_SPLIT}" \
        --grid_h "${GRID_H}" \
        --grid_w "${GRID_W}" \
        --prompt_radius "${PROMPT_RADIUS}" \
        --spatial_batch_size "${SPATIAL_BATCH_SIZE}" \
        --prompt_batch_size "${PROMPT_BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --split "${SPLIT}" \
        --num_shards 2 \
        --shard_id "${SHARD}"
    ) &
  done
  wait
  python -u scripts/precompute_salf_prompt_grid.py \
    --dataset partimagenetpp \
    --concept_set "${CONCEPT_SET}" \
    --backbone clip_RN50 \
    --lf_clip_name clip_RN50 \
    --activation_dir "${ACTIVATION_DIR}" \
    --device cpu \
    --seed 42 \
    --val_split "${VAL_SPLIT}" \
    --grid_h "${GRID_H}" \
    --grid_w "${GRID_W}" \
    --prompt_radius "${PROMPT_RADIUS}" \
    --split "${SPLIT}" \
    --num_shards 2 \
    --merge_only
  echo "[salf-cache] completed split=${SPLIT}"
done

echo "[salf-cache] done"
