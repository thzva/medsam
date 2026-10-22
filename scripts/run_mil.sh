#!/usr/bin/env bash
# Re-run only the configurations whose architecture changed with the real MIL
# attention head + NMS patch selection.  stream_global_only (single stream) and
# attn_crop_off (no cropping) are architecturally untouched, so their existing
# results in outputs_fixed remain the correct comparison.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=config/revision_mil.yaml
OUT=outputs_mil
Q=revision/results/queue_mil_status.tsv
NEED_MB=26000
mkdir -p "$OUT/logs" revision/results; touch "$Q"

pick_gpu () {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null |
  while IFS=, read -r i tot used; do
    [ $(( tot - used )) -ge "$NEED_MB" ] && { echo "$i"; return; }
  done
}
dir_of () { case "$1" in tms_edl) echo "$OUT/tms_edl/seed$2";; *) echo "$OUT/ablations/ablation_$1/seed$2";; esac; }

for seed in 42 43 44; do
 for mode in tms_edl fusion_discounted; do
  d=$(dir_of "$mode" "$seed")
  [ -f "$d/eval_revision.json" ] && { echo "[skip] $mode s$seed"; continue; }
  while :; do
    g=$(pick_gpu); load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    [ -n "$g" ] && [ "${load:-99}" -lt 45 ] && break
    sleep 180
  done
  export CUDA_VISIBLE_DEVICES=$g
  echo "=== $(date -Is) [gpu$g] TRAIN $mode seed$seed ==="
  echo -e "$(date -Is)\t$mode\tseed$seed\tSTART(gpu$g)" >> "$Q"
  $PY scripts/train_ddp.py --config "$CFG" --mode "$mode" --seed "$seed" \
      --output_dir "$OUT" > "$OUT/logs/${mode}_seed${seed}.train.log" 2>&1
  if [ -f "$d/best_model.pth" ]; then
    $PY scripts/eval_revision.py --run_dir "$d" --mode "$mode" --config "$CFG" \
        --device cuda:0 > "$OUT/logs/${mode}_seed${seed}.eval.log" 2>&1 \
      && echo -e "$(date -Is)\t$mode\tseed$seed\tDONE" >> "$Q" \
      || echo -e "$(date -Is)\t$mode\tseed$seed\tEVAL_FAIL" >> "$Q"
    find "$d" -name 'checkpoint_epoch*.pth' -delete
  else
    echo -e "$(date -Is)\t$mode\tseed$seed\tTRAIN_FAIL" >> "$Q"
  fi
 done
done
echo "=== $(date -Is) MIL RERUN COMPLETE ==="
