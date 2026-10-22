#!/usr/bin/env bash
# One LR-sweep point.  Only backbone_lr varies; augmentation, epochs and
# patience are held at the tuned settings so the sweep isolates the LR.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
CFG=$1; OUT=$2; export CUDA_VISIBLE_DEVICES=$3
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
Q=revision/results/queue_lrsweep_status.tsv
d="$OUT/ablations/ablation_stream_global_only/seed42"
mkdir -p "$OUT/logs" revision/results; touch "$Q"
[ -f "$d/eval_revision.json" ] && exit 0
mkdir -p "$(dirname "$d")"; mkdir "$d.claim" 2>/dev/null || exit 0
echo -e "$(date -Is)\t$(basename $CFG)\tSTART(gpu$3)" >> "$Q"
$PY scripts/train_ddp.py --config "$CFG" --mode stream_global_only --seed 42 \
    --output_dir "$OUT" > "$OUT/logs/sweep.train.log" 2>&1
if [ -f "$d/best_model.pth" ]; then
  $PY scripts/eval_revision.py --run_dir "$d" --mode stream_global_only --config "$CFG" \
      --device cuda:0 > "$OUT/logs/sweep.eval.log" 2>&1
  echo -e "$(date -Is)\t$(basename $CFG)\tDONE" >> "$Q"
else echo -e "$(date -Is)\t$(basename $CFG)\tTRAIN_FAIL" >> "$Q"; fi
rm -rf "$d.claim"
