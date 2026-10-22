#!/usr/bin/env bash
# Re-evaluate every completed run with the corrected near-OOD pipeline
# (resampling aligned with the clean path) and the added diagnostics
# (severity-0 control, within-test conformal).  Inference only, no retraining.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
export CUDA_VISIBLE_DEVICES=0
CFG=config/revision_final.yaml
LOG=revision/results/reeval.log

echo "waiting for the training queues to finish..."
while pgrep -f "bash scripts/run_final_queue.sh" >/dev/null \
   || pgrep -f "bash scripts/run_baselines_fix.sh" >/dev/null; do sleep 180; done
echo "training done at $(date -Is); re-evaluating all runs"

for ck in $(find outputs_final -name best_model.pth | sort); do
  dir=$(dirname "$ck")
  mode=$(basename "$(dirname "$dir")")
  case "$dir" in *"/tms_edl/"*) mode="tms_edl";; esac
  mode=${mode#ablation_}
  echo "=== $(date -Is)  RE-EVAL $mode  $dir ==="
  $PY scripts/eval_revision.py --run_dir "$dir" --mode "$mode" --config "$CFG" \
      --device cuda:0 >> "$LOG" 2>&1 \
    && echo "    ok" || echo "    FAILED $dir"
done
echo "=== $(date -Is)  RE-EVAL COMPLETE ==="
