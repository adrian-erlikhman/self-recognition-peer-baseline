#!/bin/sh
# Judge the all-API frontier corpus: one process per judge, each with its own
# output file (<cond>_api120_<Judge>.jsonl), combined afterwards.
cd "$(dirname "$0")/.."
COND=$1
for j in GPT Claude Gemini Grok DeepSeek; do
  COMPLLM_CONCURRENCY=8 PYTHONUNBUFFERED=1 .venv/Scripts/python src/revision/run_condition.py \
    --panel frontier --condition $COND --corpus data/responses_api120.csv \
    --tag api120_$j --judges $j > logs/judge_${COND}_$j.log 2>&1 &
done
wait
cat results/revision/frontier/${COND}_api120_*.jsonl > results/revision/frontier/${COND}_api120.jsonl
echo "=== $COND judging done $(date)" >> logs/judge_done.log
