#!/usr/bin/env bash
# Sweeper lane: picks up anything the two main lanes failed or skipped,
# once they are done.  Shares the same atomic mkdir claim.
set -u
cd /home/2025user/zhou/medsam
while pgrep -f "bash scripts/run_fixed_lane.sh" >/dev/null; do sleep 300; done
exec bash scripts/run_fixed_lane.sh A 0
