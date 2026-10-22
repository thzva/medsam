#!/usr/bin/env bash
# Resume on GPU1 only, one job at a time.
# Phase 1: evaluate runs whose training already finished (cheap, no retraining).
# Phase 2: train + evaluate whatever is still missing.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=config/revision_fixed.yaml
OUT=outputs_fixed
LOGDIR=$OUT/logs
Q=revision/results/queue_fixed_status.tsv
mkdir -p "$LOGDIR"

dir_of () {
  case "$1" in
    tms_edl) echo "$OUT/tms_edl/seed$2" ;;
    *)       echo "$OUT/ablations/ablation_$1/seed$2" ;;
  esac
}

do_eval () {                       # mode seed
  local d; d=$(dir_of "$1" "$2")
  [ -f "$d/eval_revision.json" ] && { echo "[skip] $1 s$2 evaluated"; return; }
  [ -f "$d/best_model.pth" ] || { echo "[skip] $1 s$2 no checkpoint"; return; }
  echo -e "$(date -Is)\tR\t$1\tseed$2\tEVAL_START" >> "$Q"
  echo "=== $(date -Is) EVAL $1 seed$2 ==="
  $PY scripts/eval_revision.py --run_dir "$d" --mode "$1" --config "$CFG" \
      --device cuda:0 > "$LOGDIR/$1_seed$2.eval.log" 2>&1 \
    && echo -e "$(date -Is)\tR\t$1\tseed$2\tDONE" >> "$Q" \
    || echo -e "$(date -Is)\tR\t$1\tseed$2\tEVAL_FAIL" >> "$Q"
}

do_train () {                      # mode seed
  local d; d=$(dir_of "$1" "$2")
  [ -f "$d/eval_revision.json" ] && { echo "[skip] $1 s$2 done"; return; }
  if [ ! -f "$d/best_model.pth" ]; then
    echo -e "$(date -Is)\tR\t$1\tseed$2\tTRAIN_START" >> "$Q"
    echo "=== $(date -Is) TRAIN $1 seed$2 ==="
    $PY scripts/train_ddp.py --config "$CFG" --mode "$1" --seed "$2" \
        --output_dir "$OUT" > "$LOGDIR/$1_seed$2.train.log" 2>&1
    [ -f "$d/best_model.pth" ] || { echo -e "$(date -Is)\tR\t$1\tseed$2\tTRAIN_FAIL" >> "$Q"; return; }
  fi
  do_eval "$1" "$2"
  find "$d" -name 'checkpoint_epoch*.pth' -delete
}

echo "### phase 1: evaluate already-trained runs ###"
for pair in "tms_edl 44" "fusion_discounted 42" "attn_crop_off 44" "attn_crop_off 42" "stream_global_only 43"; do
  do_eval $pair
done

echo "### phase 2: train the remainder ###"
for pair in "stream_global_only 42" "stream_global_only 44" "tms_edl 43" "attn_crop_off 43" "fusion_discounted 43"; do
  do_train $pair
done
echo "=== $(date -Is) RESUME COMPLETE ==="
