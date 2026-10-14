#!/bin/bash
# Smoke test for the PARALLEL LoCoMo path (--locomo-concurrency).
# Independent out-dir (locomo_smoke) so it can never touch the PILOT/full
# run state in locomo_v1. Only 2 conversations x 2 QA/category => 4 questions,
# but BOTH memory stores are built CONCURRENTLY, which is exactly the code
# path the full run will exercise (10-way). Uses the same 36K TPM budget the
# full run will use (new settings default), while the account is otherwise
# idle (PILOT stopped first).
cd /home/apulis-dev/code/SimpleMem
source /home/apulis-dev/code/envs/simplemem/bin/activate || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p eval_output
setsid nohup python eval/run_simplemem_eval.py \
  --skip-cognitive --resume --limit-conv 2 --qa-per-category 2 \
  --locomo-concurrency 10 \
  --judge-model Qwen/Qwen3-8B --judge-concurrency 4 --run-model Qwen/Qwen3-8B \
  --out-dir eval_output/locomo_smoke --run-tag locomo_smoke --strict \
  > /home/apulis-dev/code/SimpleMem/eval_output/locomo_smoke.log 2>&1 < /dev/null &
echo "smoke launched pid $!"
