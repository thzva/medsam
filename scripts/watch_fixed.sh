#!/usr/bin/env bash
# Regenerate tables, figures and the response PDF as runs land.
# Exits only when every lane and the sweeper have finished.
set -u
cd /home/2025user/zhou/medsam
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
LAST=""
while true; do
  CUR=$(find outputs_fixed outputs_final -name eval_revision.json -printf '%p %T@\n' 2>/dev/null | sort | md5sum)
  if [ "$CUR" != "$LAST" ]; then
    LAST="$CUR"
    echo "=== $(date -Is) regenerating ($(find outputs_fixed -name eval_revision.json 2>/dev/null | wc -l)/12 fixed runs) ==="
    $PY scripts/make_tables.py
    $PY scripts/make_figures.py
    ( cd revision && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 \
                  && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 )
    echo "    PDF rebuilt"
  fi
  if ! pgrep -f "bash scripts/run_fixed_lane" >/dev/null \
     && ! pgrep -f "bash scripts/run_fixed_sweeper.sh" >/dev/null; then
    sleep 120
    $PY scripts/make_tables.py; $PY scripts/make_figures.py
    ( cd revision && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 \
                  && pdflatex -interaction=nonstopmode response_to_reviewers.tex >/dev/null 2>&1 )
    echo "=== $(date -Is) ALL LANES COMPLETE, watcher exiting ==="; exit 0
  fi
  sleep 240
done
