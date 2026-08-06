#!/usr/bin/env bash
set -euo pipefail
python train_cbm.py --config autoresearch/trials/cub_sgcbm_acc_nec_000001/config.json
RUN_DIR=$(ls -td artifacts/autoresearch/cub_sgcbm/cub_sgcbm_acc_nec_000001/savlg_cbm_cub_* | head -1)
echo "RUN_DIR=${RUN_DIR}"
python scripts/train_sparse_nec.py --dataset cub --load_path "${RUN_DIR}" --lam 0.001 --n_iters 4000 --saga_batch_size 512 --cbl_batch_size 32
python scripts/autoresearch.py record --memory autoresearch/memory/cub_sgcbm_trials.jsonl --trial_id cub_sgcbm_acc_nec_000001 --run_dir "${RUN_DIR}" --nec_json "${RUN_DIR}/nec_metrics.json"
