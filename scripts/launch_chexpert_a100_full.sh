#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <pod> <gpu_index> <vlg|sgcbm|lf|salf>" >&2
  exit 1
fi

POD="$1"
GPU="$2"
METHOD="$3"

case "$METHOD" in
  vlg)
    CONFIG="configs/chexpert_vlg.yaml"
    RUN_PREFIX="chexpert_vlg_full"
    ;;
  sgcbm)
    CONFIG="configs/chexpert_sgcbm.yaml"
    RUN_PREFIX="chexpert_sgcbm_full"
    ;;
  lf)
    CONFIG="configs/chexpert_lf.yaml"
    RUN_PREFIX="chexpert_lf_full"
    ;;
  salf)
    CONFIG="configs/chexpert_salf.yaml"
    RUN_PREFIX="chexpert_salf_full"
    ;;
  *)
    echo "unsupported method: $METHOD" >&2
    exit 1
    ;;
esac

kubectl exec "$POD" -- sh -lc "
set -e
REPO=/workspace/GroundedCBM-MedicalCBM-dev
PY=/opt/conda/envs/cbm/bin/python
PIP=/opt/conda/envs/cbm/bin/pip
LOG_DIR=/workspace/logs
mkdir -p \"\$LOG_DIR\" /workspace/chexpert_full_runs
cd \"\$REPO\"

\$PY - <<'PY'
missing = []
for mod in ('transformers', 'tokenizers', 'sentencepiece', 'open_clip'):
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
if missing:
    raise SystemExit('missing:' + ','.join(missing))
PY

ts=\$(date -u +%Y%m%dT%H%M%SZ)
log=\"\$LOG_DIR/${RUN_PREFIX}_\${ts}.log\"
nohup env CUDA_VISIBLE_DEVICES=${GPU} HF_HUB_DISABLE_PROGRESS_BARS=1 \\
  \$PY -u train_cbm.py --config ${CONFIG} > \"\$log\" 2>&1 &
pid=\$!
echo POD=${POD}
echo GPU=${GPU}
echo METHOD=${METHOD}
echo CONFIG=${CONFIG}
echo PID=\$pid
echo LOG=\$log
"
