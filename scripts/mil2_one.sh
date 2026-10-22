#!/usr/bin/env bash
# One (mode, seed) of the MIL re-run, pinned to a GPU.  Atomic claim so several
# of these can be launched at once without colliding.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
mode=$1; seed=$2; export CUDA_VISIBLE_DEVICES=$3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=${CFG:-config/revision_tuned.yaml}; OUT=${OUT:-outputs_tuned}
Q=revision/results/queue_${OUT}_status.tsv
mkdir -p "$OUT/logs" revision/results; touch "$Q"
case "$mode" in tms_edl) d="$OUT/tms_edl/seed$seed";; *) d="$OUT/ablations/ablation_$mode/seed$seed";; esac
[ -f "$d/eval_revision.json" ] && { echo "[skip] $mode s$seed"; exit 0; }
mkdir -p "$(dirname "$d")"
mkdir "$d.claim" 2>/dev/null || { echo "[skip] $mode s$seed claimed"; exit 0; }
echo -e "$(date -Is)\t$mode\tseed$seed\tSTART(gpu$3)" >> "$Q"
echo "=== $(date -Is) [gpu$3] TRAIN $mode seed$seed ==="
$PY scripts/train_ddp.py --config "$CFG" --mode "$mode" --seed "$seed" \
    --output_dir "$OUT" > "$OUT/logs/${mode}_seed${seed}.train.log" 2>&1
if [ -f "$d/best_model.pth" ]; then
  $PY scripts/eval_revision.py --run_dir "$d" --mode "$mode" --config "$CFG" \
      --device cuda:0 > "$OUT/logs/${mode}_seed${seed}.eval.log" 2>&1 \
    && echo -e "$(date -Is)\t$mode\tseed$seed\tDONE" >> "$Q" \
    || echo -e "$(date -Is)\t$mode\tseed$seed\tEVAL_FAIL" >> "$Q"
  find "$d" -name 'checkpoint_epoch*.pth' -delete
else echo -e "$(date -Is)\t$mode\tseed$seed\tTRAIN_FAIL" >> "$Q"; fi
rm -rf "$d.claim"
