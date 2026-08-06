#!/bin/bash
#SBATCH --job-name=partimgpp-salf-smoke
#SBATCH --account=bfah-delta-gpu
#SBATCH --qos=bfah-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts/logs/%x-%j.out
#SBATCH --error=/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts/logs/%x-%j.err

set -euo pipefail

REPO_SRC=${REPO_SRC:-/projects/bfah/anair13/cbm_data/partimagenetpp_salf_repo/GroundedCBM}
REPO_RUN=${REPO_RUN:-/tmp/${USER}/groundedcbm_partimagenetpp_salf_${SLURM_JOB_ID}}
EXP=${EXP:-/projects/bfah/anair13/cbm_data/partimagenetpp_generic_parts}
MANIFEST=${MANIFEST:-${EXP}/manifest.jsonl}
SAVE_DIR=${SAVE_DIR:-${EXP}/salf_smoke_runs}
ACTIVATION_DIR=${ACTIVATION_DIR:-/tmp/${USER}/partimagenetpp_salf_cache_${SLURM_JOB_ID}}

export PARTIMAGENETPP_MANIFEST="${MANIFEST}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

mkdir -p "${REPO_RUN}" "${SAVE_DIR}" "${ACTIVATION_DIR}"
rsync -a --exclude '__pycache__' --exclude 'saved_activations' --exclude 'artifacts' "${REPO_SRC}/" "${REPO_RUN}/"

cd "${REPO_RUN}"
source /u/anair13/GroundingDINO/gdino_uv/bin/activate

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo_run=${REPO_RUN}"
echo "manifest=${PARTIMAGENETPP_MANIFEST}"
echo "save_dir=${SAVE_DIR}"
echo "activation_dir=${ACTIVATION_DIR}"
nvidia-smi

CONFIG_PATH="${REPO_RUN}/partimagenetpp_salf_smoke_config.json"
cat > "${CONFIG_PATH}" <<JSON
{
  "model_name": "salf_cbm",
  "dataset": "partimagenetpp",
  "concept_set": "concept_files/partimagenetpp_generic_parts.txt",
  "backbone": "clip_RN50",
  "lf_clip_name": "clip_RN50",
  "save_dir": "${SAVE_DIR}",
  "activation_dir": "${ACTIVATION_DIR}",
  "device": "cuda",
  "num_workers": 4,
  "spatial_num_workers": 4,
  "max_train_images": 20,
  "max_test_images": 10,
  "val_split": 0.1,
  "grid_h": 7,
  "grid_w": 7,
  "prompt_radius": 3,
  "spatial_batch_size": 2,
  "prompt_batch_size": 64,
  "clip_cutoff": null,
  "cbl_epochs": 1,
  "cbl_batch_size": 4,
  "dense": true,
  "saga_n_iters": 5,
  "skip_train_val_eval": true,
  "skip_test_eval": true
}
JSON

cat "${CONFIG_PATH}"
python -u train_cbm.py --config "${CONFIG_PATH}"
