#!/usr/bin/env bash
set -euo pipefail

cd /workspace/GroundedCBM

RUN_TS="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_ROOT="/tmp/partimagenetpp_salf_adequacy_v3_${RUN_TS}"
STAGED_TRAIN_ROOT="${STAGE_ROOT}/train_images"
STAGED_VAL_ROOT="${STAGE_ROOT}/val_images"
STAGED_TRAIN_MANIFEST="${STAGE_ROOT}/partimagenetpp_pinpp_train_90k_manifest_local.jsonl"
STAGED_VAL_MANIFEST="${STAGE_ROOT}/partimagenetpp_pinpp_val_10k_manifest_local.jsonl"
SAVE_ROOT="/workspace/partimagenetpp_runs/salf_scratch_resnet50_clipvitb16_adequacy_v3"
RESULT_DIR="/workspace/partimagenetpp_results"
NEC_ROOT="/workspace/partimagenetpp_runs/salf_scratch_resnet50_clipvitb16_adequacy_v3_nec"
GT_BOXES="/workspace/partimagenetpp_eval_payload/pinpp_val_gt_boxes_generic_nonidentity.jsonl"
GT_SEGMENTS="/workspace/partimagenetpp_eval_payload/pinpp_val_gt_segments_generic_nonidentity.jsonl"

python -m py_compile \
  data/utils.py \
  methods/salf.py \
  model/cbm.py \
  train_cbm.py \
  evaluations/sparse_utils.py \
  scripts/eval_concept_accuracy.py \
  scripts/eval_partimagenetpp_gtbox_localization.py \
  scripts/run_partimagenetpp_salf_sparse_nec_path.py
python -m json.tool configs/partimagenetpp_salf_scratch_resnet50_clipvitb16_adequacy.json >/dev/null

mkdir -p "${STAGED_TRAIN_ROOT}" "${STAGED_VAL_ROOT}" "${SAVE_ROOT}" "${RESULT_DIR}" "${NEC_ROOT}"
echo "[salf-adequacy] staging PartImageNet++ archives under ${STAGE_ROOT}"
tar -xf /workspace/partimagenetpp_eval_payload/pinpp_train_images_90k.tar -C "${STAGED_TRAIN_ROOT}"
tar -xf /workspace/partimagenetpp_eval_payload/pinpp_val_images.tar -C "${STAGED_VAL_ROOT}"

python - <<PY
import json
from pathlib import Path

def rewrite(source, destination, image_root):
    source, destination, image_root = Path(source), Path(destination), Path(image_root)
    count = 0
    with source.open("r", encoding="utf-8") as input_file, destination.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            relative = row.get("file_name") or "/".join(Path(row["image"]).parts[-2:])
            row["image"] = str(image_root / relative)
            if "path" in row:
                row["path"] = row["image"]
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    print({"destination": str(destination), "rows": count}, flush=True)

rewrite("/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_train_90k_manifest_pvc.jsonl", "${STAGED_TRAIN_MANIFEST}", "${STAGED_TRAIN_ROOT}")
rewrite("/workspace/partimagenetpp_eval_payload/partimagenetpp_pinpp_val_10k_manifest_pvc.jsonl", "${STAGED_VAL_MANIFEST}", "${STAGED_VAL_ROOT}")
PY

export PARTIMAGENETPP_TRAIN_MANIFEST="${STAGED_TRAIN_MANIFEST}"
export PARTIMAGENETPP_VAL_MANIFEST="${STAGED_VAL_MANIFEST}"
CACHE_DIR="/workspace/partimagenetpp_runs/salf_scratch_resnet50_clipvitb16_activations"
test -s "${CACHE_DIR}/partimagenetpp_train_salf_resnet50_partimagenetpp_scratch_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_5431310002d47ebf_P.pt"
test -s "${CACHE_DIR}/partimagenetpp_val_salf_resnet50_partimagenetpp_scratch_clip_clip_ViT_B_16_prompt_grid_gh7_gw7_r32_5431310002d47ebf_P.pt"
echo "[salf-adequacy] reusing verified CLIP ViT-B/16 prompt-grid cache"
nvidia-smi

python -u train_cbm.py \
  --config configs/partimagenetpp_salf_scratch_resnet50_clipvitb16_adequacy.json \
  --save_dir "${SAVE_ROOT}" \
  --num_workers 0

RUN_DIR="$(python - <<'PY'
from pathlib import Path

root = Path("/workspace/partimagenetpp_runs/salf_scratch_resnet50_clipvitb16_adequacy_v3")
for run in sorted(root.glob("salf_cbm_partimagenetpp_*"), key=lambda item: item.stat().st_mtime, reverse=True):
    required = ("args.txt", "concept_layer.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt")
    if all((run / name).is_file() for name in required):
        print(run)
        break
PY
)"
test -n "${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" > /workspace/partimagenetpp_runs/salf_scratch_resnet50_clipvitb16_adequacy_v3_latest_run_dir.txt

CONCEPT_OUT="${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_humanboxes_concept_${RUN_TS}.json"
BOX_OUT="${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_humanboxes_${RUN_TS}.json"
SEGMENT_OUT="${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_humansegments_${RUN_TS}.json"
NEC_OUT="${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_nec_${RUN_TS}.json"
NEC_DIR="${NEC_ROOT}/$(basename "${RUN_DIR}")"

python -u scripts/eval_concept_accuracy.py \
  --dataset partimagenetpp --load_paths "${RUN_DIR}" --model_names salf_cbm --names salf_adequacy_v3 \
  --partimagenetpp_train_manifest "${PARTIMAGENETPP_TRAIN_MANIFEST}" \
  --partimagenetpp_val_manifest "${PARTIMAGENETPP_VAL_MANIFEST}" \
  --gt_source partimagenetpp_boxes --partimagenetpp_gt_boxes_jsonl "${GT_BOXES}" \
  --output "${CONCEPT_OUT}" --device cuda --batch_size 256 --num_workers 4 \
  --normalization model_default --threshold 0.0 --log_every 1000

python -u scripts/eval_partimagenetpp_gtbox_localization.py \
  --gcbm_path "${RUN_DIR}" --model_name salf_cbm \
  --train_manifest "${PARTIMAGENETPP_TRAIN_MANIFEST}" --val_manifest "${PARTIMAGENETPP_VAL_MANIFEST}" \
  --gt_boxes_jsonl "${GT_BOXES}" --output "${BOX_OUT}" --device cuda --batch_size 128 --num_workers 4 \
  --evaluation_map_size 224 --threshold_mode mean --activation_thresholds 0.5 --log_every 1000

python -u scripts/eval_partimagenetpp_gtbox_localization.py \
  --gcbm_path "${RUN_DIR}" --model_name salf_cbm \
  --train_manifest "${PARTIMAGENETPP_TRAIN_MANIFEST}" --val_manifest "${PARTIMAGENETPP_VAL_MANIFEST}" \
  --gt_segments_jsonl "${GT_SEGMENTS}" --output "${SEGMENT_OUT}" --device cuda --batch_size 128 --num_workers 4 \
  --evaluation_map_size 224 --threshold_mode mean --activation_thresholds 0.5 --log_every 1000

python -u scripts/run_partimagenetpp_salf_sparse_nec_path.py \
  --source_run_dir "${RUN_DIR}" --output_dir "${NEC_DIR}" \
  --train_manifest "${PARTIMAGENETPP_TRAIN_MANIFEST}" --val_manifest "${PARTIMAGENETPP_VAL_MANIFEST}" \
  --result_json "${NEC_OUT}" --lam_max 0.1 --n_iters 100 --max_glm_steps 150 \
  --cbl_batch_size 512 --saga_batch_size 4096 --num_workers 8

printf '%s\n' "${CONCEPT_OUT}" > "${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_latest_concept_result.txt"
printf '%s\n' "${BOX_OUT}" > "${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_latest_box_result.txt"
printf '%s\n' "${SEGMENT_OUT}" > "${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_latest_segment_result.txt"
printf '%s\n' "${NEC_OUT}" > "${RESULT_DIR}/salf_scratch_resnet50_clipvitb16_adequacy_v3_latest_nec_result.txt"
echo "[salf-adequacy] completed run=${RUN_DIR}"
