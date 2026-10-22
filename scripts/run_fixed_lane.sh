#!/usr/bin/env bash
# Core comparison after the D7/D8/D9 fixes.  Two lanes share one work list and
# coordinate with an atomic mkdir claim, so GPU0 and GPU1 never duplicate a run.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
LANE=$1                      # A or B
export CUDA_VISIBLE_DEVICES=$2
CFG=config/revision_fixed.yaml
OUT=outputs_fixed
LOGDIR=$OUT/logs
QUEUE=revision/results/queue_fixed_status.tsv
mkdir -p "$LOGDIR" revision/results; touch "$QUEUE"

run () {
  local mode=$1 seed=$2 dir
  if [ "$mode" = "tms_edl" ]; then dir="$OUT/tms_edl/seed$seed"
  else dir="$OUT/ablations/ablation_$mode/seed$seed"; fi
  [ -f "$dir/eval_revision.json" ] && { echo "[skip] $mode s$seed done"; return; }
  mkdir -p "$(dirname "$dir")"
  mkdir "$dir.claim" 2>/dev/null || { echo "[skip] $mode s$seed claimed"; return; }

  echo -e "$(date -Is)\t$LANE\t$mode\tseed$seed\tSTART" >> "$QUEUE"
  echo "=== $(date -Is) [$LANE] TRAIN $mode seed$seed ==="
  $PY scripts/train_ddp.py --config "$CFG" --mode "$mode" --seed "$seed" \
      --output_dir "$OUT" > "$LOGDIR/${mode}_seed${seed}.train.log" 2>&1
  if [ $? -ne 0 ] || [ ! -f "$dir/best_model.pth" ]; then
    echo -e "$(date -Is)\t$LANE\t$mode\tseed$seed\tTRAIN_FAIL" >> "$QUEUE"; return
  fi
  echo "=== $(date -Is) [$LANE] EVAL $mode seed$seed ==="
  $PY scripts/eval_revision.py --run_dir "$dir" --mode "$mode" --config "$CFG" \
      --device cuda:0 > "$LOGDIR/${mode}_seed${seed}.eval.log" 2>&1
  [ $? -eq 0 ] && echo -e "$(date -Is)\t$LANE\t$mode\tseed$seed\tDONE" >> "$QUEUE" \
               || echo -e "$(date -Is)\t$LANE\t$mode\tseed$seed\tEVAL_FAIL" >> "$QUEUE"
  find "$dir" -name 'checkpoint_epoch*.pth' -delete
}

if [ "$LANE" = "A" ]; then
  for s in 42 43 44; do for m in tms_edl stream_global_only attn_crop_off fusion_discounted; do run "$m" "$s"; done; done
else
  for s in 44 43 42; do for m in fusion_discounted attn_crop_off stream_global_only tms_edl; do run "$m" "$s"; done; done
fi
echo "=== $(date -Is) LANE-$LANE COMPLETE ==="
