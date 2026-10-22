#!/usr/bin/env bash
cd /home/2025user/zhou/medsam
Q=revision/results/queue_final_status.tsv
echo "=== $(date -Is) ==="
n_done=$(find outputs_final -name eval_revision.json 2>/dev/null | wc -l)
n_fail=$(grep -c "FAIL" $Q 2>/dev/null || echo 0)
echo "evals: $n_done / 31    recorded failures: $n_fail"
# what is actually executing right now
cur=$(ps -eo cmd --no-headers | grep -E '[t]rain_ddp.py|[e]val_revision.py' | head -1 \
      | grep -oE '\-\-mode [a-z0-9_]+|--run_dir [^ ]+' | tr '\n' ' ')
phase="idle"
ps -eo cmd --no-headers | grep -q '[t]rain_ddp.py' && phase="TRAIN"
ps -eo cmd --no-headers | grep -q '[e]val_revision.py' && phase="EVAL"
echo "phase: $phase  $cur"
echo "queues: main=$(ps -eo cmd --no-headers|grep -c '[r]un_final_queue.sh') retry=$(ps -eo cmd --no-headers|grep -c '[r]un_baselines_fix.sh') watcher=$(ps -eo cmd --no-headers|grep -c '[w]atch_revision.sh')"
echo "gpu0: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 0)"
echo "disk: $(df -h /home/2025user/zhou | tail -1 | awk '{print $4" free ("$5" used)"}')   outputs_final: $(du -sh outputs_final 2>/dev/null | cut -f1)"
echo "--- last 3 queue events ---"; tail -3 $Q
if [ "$n_fail" -gt 0 ]; then echo "--- failures ---"; grep "FAIL" $Q; fi
