#!/usr/bin/env bash
# Regenerate tables, figures and the response PDF whenever a new eval lands.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
LAST=""
while true; do
  CUR=$(find outputs_revision outputs_final -name eval_revision.json -printf '%p %T@\n' 2>/dev/null | sort | md5sum)
  if [ "$CUR" != "$LAST" ]; then
    LAST="$CUR"
    echo "=== $(date -Is) regenerating ==="
    $PY scripts/make_tables.py
    $PY scripts/make_figures.py
    ( cd revision && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 \
                 && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 )
    echo "    PDF rebuilt: revision/response_to_reviewers.pdf"
  fi
  # stop once the queue is done and nothing changed for a full cycle
  # Exit only when EVERY stage is done: main queue, baseline retries and
  # the re-evaluation pass.  Keying on the main queue alone made the
  # watcher quit while later stages were still producing results.
  if grep -q "RE-EVAL COMPLETE" revision/results/queue_reeval.log 2>/dev/null \
     && ! pgrep -f "bash scripts/run_baselines_fix.sh" >/dev/null \
     && ! pgrep -f "bash scripts/reeval_all.sh" >/dev/null; then
     sleep 120
     $PY scripts/make_tables.py; $PY scripts/make_figures.py
     ( cd revision && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 \
                   && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 )
     echo "=== $(date -Is) queue complete, watcher exiting ==="; exit 0
  fi
  sleep 300
done
