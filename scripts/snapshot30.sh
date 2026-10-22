#!/usr/bin/env bash
# Status snapshot every 30 minutes, appended to a single log.
set -u
cd /home/2025user/zhou/medsam
S=revision/results/snapshots.log
while true; do
  {
    echo "===== $(date -Is)  load=$(cut -d' ' -f1-3 /proc/loadavg) ====="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null |
      while IFS=, read -r i u m t; do printf "  gpu%s util=%s%% free=%s MiB\n" "$i" "$u" "$((t-m))"; done
    for f in outputs_tuned/logs/*.train.log outputs_lrsweep/*/logs/sweep.train.log; do
      [ -e "$f" ] || continue
      case "$f" in outputs_lrsweep/*) n="sweep_$(echo "$f" | cut -d/ -f2)";; *) n=$(basename "$f" .train.log);; esac
      last=$(tr '\r' '\n' < "$f" | grep -a "val_auc=" | tail -1 |
             grep -oE "Epoch [0-9]+/[0-9]+.*val_auc=[0-9.]+ \([^)]*\)" | sed 's/ | train_loss=[0-9.]*//;s/ | val_loss=[0-9.]*//')
      best=$(tr '\r' '\n' < "$f" | grep -oE "val_auc=[0-9.]+" | cut -d= -f2 | sort -rn | head -1)
      printf "  %-30s best=%-8s %s\n" "$n" "${best:-–}" "${last:-starting}"
    done
    echo "  done: tuned=$(find outputs_tuned -name eval_revision.json 2>/dev/null|wc -l) sweep=$(find outputs_lrsweep -name eval_revision.json 2>/dev/null|wc -l)"
  } >> "$S" 2>&1
  sleep 1800
done
