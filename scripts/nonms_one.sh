#!/usr/bin/env bash
# One mil_no_nms seed, pinned to a given GPU.  Atomic claim keeps the lanes
# from colliding with the sequential runner that is already going.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
seed=$1; export CUDA_VISIBLE_DEVICES=$2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=config/revision_mil.yaml; OUT=outputs_mil
Q=revision/results/queue_mil_status.tsv
d="$OUT/ablations/ablation_mil_no_nms/seed$seed"
[ -f "$d/eval_revision.json" ] && { echo "[skip] seed$seed done"; exit 0; }
mkdir -p "$(dirname "$d")" "$OUT/logs"
mkdir "$d.claim" 2>/dev/null || { echo "[skip] seed$seed claimed"; exit 0; }
echo "=== $(date -Is) [gpu$2] TRAIN mil_no_nms seed$seed ==="
echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tSTART(gpu$2,par)" >> "$Q"
$PY scripts/train_ddp.py --config "$CFG" --mode mil_no_nms --seed "$seed" \
    --output_dir "$OUT" > "$OUT/logs/mil_no_nms_seed${seed}.train.log" 2>&1
if [ -f "$d/best_model.pth" ]; then
  $PY scripts/eval_revision.py --run_dir "$d" --mode mil_no_nms --config "$CFG" \
      --device cuda:0 > "$OUT/logs/mil_no_nms_seed${seed}.eval.log" 2>&1 \
    && echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tDONE" >> "$Q" \
    || echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tEVAL_FAIL" >> "$Q"
  find "$d" -name 'checkpoint_epoch*.pth' -delete
else
  echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tTRAIN_FAIL" >> "$Q"
fi
rm -rf "$d.claim"
