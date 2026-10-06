#!/bin/sh
# All-API frontier corpus: every model answers all 120 prompts, five models in
# parallel, each in its own process and part file.
cd "$(dirname "$0")/.."
for m in GPT Claude Gemini Grok DeepSeek; do
  COMPLLM_CONCURRENCY=8 PYTHONUNBUFFERED=1 .venv/Scripts/python src/revision/collect_frontier.py \
    --api-v1 --expand --models $m > logs/collect_$m.log 2>&1 &
done
wait
echo "=== collection done $(date)" >> logs/collect_done.log
