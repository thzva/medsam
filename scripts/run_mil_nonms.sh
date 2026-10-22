#!/usr/bin/env bash
# Disentangling ablation: MIL head with plain top-k (nms_radius=0).
# tms_edl in outputs_mil already gives MIL+NMS; outputs_fixed gives neither.
# The three together separate the MIL head's effect from spatial diversification.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=config/revision_mil.yaml; OUT=outputs_mil
Q=revision/results/queue_mil_status.tsv
NEED_MB=26000
echo "waiting for the MIL rerun to finish..."
while pgrep -f "bash scripts/run_mil.sh" >/dev/null; do sleep 300; done
pick_gpu () {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null |
  while IFS=, read -r i tot used; do [ $(( tot - used )) -ge "$NEED_MB" ] && { echo "$i"; return; }; done
}
for seed in 42 43 44; do
  d="$OUT/ablations/ablation_mil_no_nms/seed$seed"
  [ -f "$d/eval_revision.json" ] && continue
  while :; do g=$(pick_gpu); l=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    [ -n "$g" ] && [ "${l:-99}" -lt 45 ] && break; sleep 180; done
  export CUDA_VISIBLE_DEVICES=$g
  echo "=== $(date -Is) [gpu$g] TRAIN mil_no_nms seed$seed ==="
  echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tSTART(gpu$g)" >> "$Q"
  $PY scripts/train_ddp.py --config "$CFG" --mode mil_no_nms --seed "$seed" \
      --output_dir "$OUT" > "$OUT/logs/mil_no_nms_seed${seed}.train.log" 2>&1
  if [ -f "$d/best_model.pth" ]; then
    $PY scripts/eval_revision.py --run_dir "$d" --mode mil_no_nms --config "$CFG" \
        --device cuda:0 > "$OUT/logs/mil_no_nms_seed${seed}.eval.log" 2>&1 \
      && echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tDONE" >> "$Q" \
      || echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tEVAL_FAIL" >> "$Q"
    find "$d" -name 'checkpoint_epoch*.pth' -delete
  else echo -e "$(date -Is)\tmil_no_nms\tseed$seed\tTRAIN_FAIL" >> "$Q"; fi
done
echo "=== $(date -Is) MIL_NO_NMS COMPLETE ==="
