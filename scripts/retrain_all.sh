#!/bin/bash
# Retrain all 10 TMS-EDL variants with updated config (80 ep, patience 15).
# Two lanes, each 2-GPU DDP, running concurrently.
# Lane A = GPU 0+1 (5 jobs)
# Lane B = GPU 2+3 (5 jobs)
set -u
cd /home/2025user/zhou/medsam

LOG_DIR=outputs/retrain_logs
mkdir -p "$LOG_DIR"

# Use medsam conda env
CONDA_ENV=/home/2025user/zhou/anaconda3/envs/medsam
export PATH=$CONDA_ENV/bin:$PATH
TORCHRUN=$CONDA_ENV/bin/torchrun

MODES_A=(tms_edl fusion_average fusion_gating stream_global_only attn_crop_off)
MODES_B=(fusion_dempster_shafer stream_local_only kl_disabled kl_no_annealing loss_edl_mse)

metrics_path() {
  # Return the path to test_metrics.json for a given mode
  local m=$1
  if [ "$m" = "tms_edl" ]; then
    echo "outputs/tms_edl/test_metrics.json"
  else
    echo "outputs/ablations/ablation_${m}/test_metrics.json"
  fi
}

run_lane() {
  local gpus=$1; local port=$2; local lane=$3; shift 3
  for m in "$@"; do
    local metrics=$(metrics_path "$m")
    if [ -f "$metrics" ]; then
      echo "[$(date '+%F %T')] Lane $lane skip (already done): $m"
      continue
    fi
    echo "[$(date '+%F %T')] Lane $lane start: $m"
    # Retry up to 2 times if job fails (OOM, crash, etc.)
    local attempts=0
    local rc=1
    while [ $attempts -lt 2 ] && [ $rc -ne 0 ]; do
      attempts=$((attempts + 1))
      CUDA_VISIBLE_DEVICES=$gpus \
        $TORCHRUN --nproc_per_node=2 --master_port=$port scripts/train_ddp.py \
            --config config/default.yaml --mode "$m" \
            >> "$LOG_DIR/${m}.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ] && [ $attempts -lt 2 ]; then
        echo "[$(date '+%F %T')] Lane $lane FAIL on $m (rc=$rc, attempt $attempts), retrying with resume..."
        sleep 5
      fi
    done
    if [ $rc -ne 0 ]; then
      echo "[$(date '+%F %T')] Lane $lane FAIL on $m after $attempts attempts (rc=$rc) — continuing"
    else
      echo "[$(date '+%F %T')] Lane $lane done: $m"
    fi
  done
  echo "[$(date '+%F %T')] Lane $lane COMPLETE"
}

run_lane "0,1" 30500 A "${MODES_A[@]}" > "$LOG_DIR/laneA.log" 2>&1 &
LANE_A_PID=$!
run_lane "2,3" 30501 B "${MODES_B[@]}" > "$LOG_DIR/laneB.log" 2>&1 &
LANE_B_PID=$!

echo "Lane A PID: $LANE_A_PID  |  Lane B PID: $LANE_B_PID"
echo "Logs: $LOG_DIR/"

wait $LANE_A_PID $LANE_B_PID
echo "[$(date '+%F %T')] ALL LANES COMPLETE"
