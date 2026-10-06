#!/bin/sh
# Re-run the open-weight study until it exits cleanly. Every step resumes from
# what it has already written, so a relaunch after a sleep, a lost GPU context
# or a guard kill only repeats the unfinished chunk. Gives up after 8 tries.
cd "$(dirname "$0")/.."
FROM=${1:-corpus}
for try in 1 2 3 4 5 6 7 8; do
  echo "=== attempt $try from $FROM $(date)" >> logs/openweight_run.log
  PYTHONUNBUFFERED=1 COMPLLM_PREFILL_BUDGET=1024 COMPLLM_VRAM_HEADROOM=1.1e9 .venv/Scripts/python tools/guard.py \
    --min-free-gb 1.2 -- python src/revision/run_openweight.py --from "$FROM" \
    >> logs/openweight_run.log 2>&1 && { echo "=== study complete $(date)" >> logs/openweight_run.log; exit 0; }
  sleep 30
done
exit 1
