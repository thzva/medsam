#!/usr/bin/env bash
# TVC round 2: uniform-MIL and discounting rows under the current code.
# TVC revision experiments. Several lanes share one ordered job list and claim
# jobs atomically (mkdir), so lanes on GPU1/GPU3 never duplicate a run.
# usage: tvc_lane.sh <lane_name> <gpu> <light|heavy|any>
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
LANE=$1; GPU=$2; KIND=$3
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=config/revision_tvc.yaml; OUT=outputs_tvc
Q=revision/results/queue_tvc2_status.tsv
mkdir -p "$OUT/logs" revision/results; touch "$Q"
ABL=" mil_uniform fusion_discounted fusion_yager fusion_cautious fusion_average fusion_gating evid_dirichlet stream_local_only stream_global_only "

# priority order; "L" = single stream (~15-20 GB), "H" = dual stream (~35 GB)
JOBS="
H mil_uniform 42
H fusion_discounted 42
H mil_uniform 43
H fusion_discounted 43
H mil_uniform 44
H fusion_discounted 44
"
need_mb () { [ "$1" = H ] && echo 38000 || echo 22000; }
free_mb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' '; }

run () {
  local k=$1 mode=$2 seed=$3 d
  if [ "$mode" = tms_edl ]; then d="$OUT/tms_edl/seed$seed"
  elif [[ "$ABL" == *" $mode "* ]]; then d="$OUT/ablations/ablation_$mode/seed$seed"
  else d="$OUT/baselines/$mode/seed$seed"; fi
  [ -f "$d/eval_revision.json" ] && return 0
  mkdir -p "$(dirname "$d")"
  # wait until this GPU has room (the vLLM tenant on GPU3 may grow)
  local tries=0
  while [ "$(free_mb)" -lt "$(need_mb $k)" ]; do
    tries=$((tries+1)); [ $tries -gt 120 ] && return 0; sleep 60; done
  mkdir "$d.claim" 2>/dev/null || return 0
  echo -e "$(date -Is)\t$LANE(gpu$GPU)\t$mode\tseed$seed\tSTART" >> "$Q"
  $PY scripts/train_ddp.py --config "$CFG" --mode "$mode" --seed "$seed" \
      --output_dir "$OUT" > "$OUT/logs/${mode}_seed${seed}.train.log" 2>&1
  if [ -f "$d/best_model.pth" ]; then
    $PY scripts/eval_revision.py --run_dir "$d" --mode "$mode" --config "$CFG" \
        --device cuda:0 > "$OUT/logs/${mode}_seed${seed}.eval.log" 2>&1 \
      && echo -e "$(date -Is)\t$LANE(gpu$GPU)\t$mode\tseed$seed\tDONE" >> "$Q" \
      || echo -e "$(date -Is)\t$LANE(gpu$GPU)\t$mode\tseed$seed\tEVAL_FAIL" >> "$Q"
    find "$d" -name 'checkpoint_epoch*.pth' -delete; rm -f "$d/last.pth"
  else echo -e "$(date -Is)\t$LANE(gpu$GPU)\t$mode\tseed$seed\tTRAIN_FAIL" >> "$Q"; fi
  rm -rf "$d.claim"
}

for pass in 1 2; do   # second pass picks up anything skipped for lack of memory
  while read -r k mode seed; do
    [ -z "$k" ] && continue
    [ "$KIND" = light ] && [ "$k" = H ] && continue
    [ "$KIND" = heavy ] && [ "$k" = L ] && continue
    run "$k" "$mode" "$seed"
  done <<< "$JOBS"
done
echo -e "$(date -Is)\t$LANE(gpu$GPU)\t-\t-\tLANE_COMPLETE" >> "$Q"
