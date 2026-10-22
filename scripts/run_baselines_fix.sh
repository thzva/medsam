#!/usr/bin/env bash
# Supplementary queue: the four deterministic baselines that crashed with
#   "binary_cross_entropy ... unsafe to autocast"
# Waits for the main queue to finish so both share GPU0 serially.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
export CUDA_VISIBLE_DEVICES=0
CFG=config/revision_final.yaml
OUT=outputs_final
LOGDIR=$OUT/logs
QUEUE=revision/results/queue_final_status.tsv
mkdir -p "$LOGDIR"

echo "waiting for the main queue to finish..."
while pgrep -f "bash scripts/run_final_queue.sh" > /dev/null; do sleep 180; done
echo "main queue done at $(date -Is); running the four repaired baselines"

for mode in resnet50_ce densenet121_ce vit_b16_ce resnet50_mc_dropout ; do
  dir="$OUT/baselines/$mode/seed42"
  if [ -f "$dir/eval_revision.json" ]; then echo "[skip] $mode"; continue; fi
  echo -e "$(date -Is)\t$mode\tseed42\tSTART(retry)" >> "$QUEUE"
  echo "=== $(date -Is)  TRAIN $mode seed42 (retry) ==="
  $PY scripts/train_ddp.py --config "$CFG" --mode "$mode" --seed 42 \
      --output_dir "$OUT" > "$LOGDIR/${mode}_seed42.train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$dir/best_model.pth" ]; then
    echo -e "$(date -Is)\t$mode\tseed42\tTRAIN_FAIL($rc)" >> "$QUEUE"; continue
  fi
  $PY scripts/eval_revision.py --run_dir "$dir" --mode "$mode" --config "$CFG" \
      --device cuda:0 > "$LOGDIR/${mode}_seed42.eval.log" 2>&1
  [ $? -eq 0 ] && echo -e "$(date -Is)\t$mode\tseed42\tDONE" >> "$QUEUE" \
                || echo -e "$(date -Is)\t$mode\tseed42\tEVAL_FAIL" >> "$QUEUE"
  find "$dir" -name 'checkpoint_epoch*.pth' -delete
done
echo "=== $(date -Is)  BASELINE RETRY COMPLETE ==="
