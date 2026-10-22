#!/bin/bash
# Backup lane on GPU 3 only: single-GPU training for the 2 remaining Lane A jobs.
set -u
cd /home/2025user/zhou/medsam

LOG_DIR=outputs/retrain_logs
mkdir -p "$LOG_DIR"

CONDA_ENV=/home/2025user/zhou/anaconda3/envs/medsam
export PATH=$CONDA_ENV/bin:$PATH
PYTHON=$CONDA_ENV/bin/python

# Single-GPU jobs (sequential on GPU 3)
MODES=(stream_global_only attn_crop_off)

for m in "${MODES[@]}"; do
  echo "[$(date '+%F %T')] GPU3 start: $m"
  # Before starting, remove the stub so training can save real results
  if [ "$m" = "tms_edl" ]; then
    tm_path="outputs/tms_edl/test_metrics.json"
  else
    tm_path="outputs/ablations/ablation_${m}/test_metrics.json"
  fi
  # Keep stub present so Lane A also skips during training.
  # Trainer will overwrite test_metrics.json when it finishes eval.
  CUDA_VISIBLE_DEVICES=3 \
    $PYTHON scripts/train_ddp.py \
        --config config/single_gpu.yaml --mode "$m" \
        > "$LOG_DIR/${m}_gpu3.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] GPU3 FAIL on $m (rc=$rc) — continuing"
  else
    echo "[$(date '+%F %T')] GPU3 done: $m"
  fi
done
echo "[$(date '+%F %T')] GPU3 backup lane COMPLETE"
