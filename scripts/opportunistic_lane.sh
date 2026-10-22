#!/usr/bin/env bash
# Opportunistic second lane: waits until a GPU has enough FREE memory and a
# tolerable load, then takes one unclaimed run.  Never competes for a card that
# is already tight -- the point is to use spare capacity, not to create the
# contention that stalled the previous attempt.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=config/revision_fixed.yaml
OUT=outputs_fixed
Q=revision/results/queue_fixed_status.tsv
NEED_MB=26000        # our peak is ~17 GB; keep a real margin
MAX_LOAD=45

dir_of () { case "$1" in tms_edl) echo "$OUT/tms_edl/seed$2";; *) echo "$OUT/ablations/ablation_$1/seed$2";; esac; }

pick_gpu () {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null |
  while IFS=, read -r i tot used; do
    free=$(( tot - used ))
    [ "$free" -ge "$NEED_MB" ] && { echo "$i"; return; }
  done
}

WORK="tms_edl:43 attn_crop_off:43 fusion_discounted:43 stream_global_only:44 tms_edl:44 attn_crop_off:44"

for item in $WORK; do
  mode=${item%%:*}; seed=${item##*:}
  d=$(dir_of "$mode" "$seed")
  [ -f "$d/eval_revision.json" ] && continue
  # wait for capacity
  while :; do
    [ -f "$d/eval_revision.json" ] && break 
    load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    g=$(pick_gpu)
    if [ -n "$g" ] && [ "${load:-99}" -lt "$MAX_LOAD" ]; then break; fi
    sleep 180
  done
  [ -f "$d/eval_revision.json" ] && continue
  mkdir -p "$(dirname "$d")"
  mkdir "$d.claim" 2>/dev/null || { echo "[skip] $mode s$seed claimed elsewhere"; continue; }
  export CUDA_VISIBLE_DEVICES=$g
  echo "=== $(date -Is) [OPP gpu$g] $mode seed$seed ==="
  echo -e "$(date -Is)\tOPP\t$mode\tseed$seed\tSTART(gpu$g)" >> "$Q"
  if [ ! -f "$d/best_model.pth" ]; then
    $PY scripts/train_ddp.py --config "$CFG" --mode "$mode" --seed "$seed" \
        --output_dir "$OUT" > "$OUT/logs/${mode}_seed${seed}.train.log" 2>&1
  fi
  if [ -f "$d/best_model.pth" ]; then
    $PY scripts/eval_revision.py --run_dir "$d" --mode "$mode" --config "$CFG" \
        --device cuda:0 > "$OUT/logs/${mode}_seed${seed}.eval.log" 2>&1 \
      && echo -e "$(date -Is)\tOPP\t$mode\tseed$seed\tDONE" >> "$Q" \
      || echo -e "$(date -Is)\tOPP\t$mode\tseed$seed\tEVAL_FAIL" >> "$Q"
    find "$d" -name 'checkpoint_epoch*.pth' -delete
  else
    echo -e "$(date -Is)\tOPP\t$mode\tseed$seed\tTRAIN_FAIL" >> "$Q"
  fi
  rm -rf "$d.claim"
done
echo "=== $(date -Is) OPPORTUNISTIC LANE COMPLETE ==="
