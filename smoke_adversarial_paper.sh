#!/bin/bash
# Smoke test for the PAPER adversarial protocol (--adversarial-mode paper).
# Independent out-dir (locomo_smoke_paper) so it can never touch the state in
# locomo_v1. 2 conversations x 2 QA/category = 20 questions (4 of them
# category-5 "paper" ones), both memory stores built CONCURRENTLY - the same
# code path the full run exercises (10-way). Validates on the REAL System:
#   hybrid_retriever.retrieve(question, enable_reflection=False),
#   answer_generator._format_contexts, llm_client.chat_completion(
#   temperature=0.5, max_retries=3), extract_json, the protocol/ground_truth
#   fields on the rows, and the adversarial-paper judge rubric.
# Same 36K TPM budget as the full run.
cd /home/apulis-dev/code/SimpleMem
source /home/apulis-dev/code/envs/simplemem/bin/activate || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p eval_output
setsid nohup python eval/run_simplemem_eval.py \
  --skip-cognitive --resume --limit-conv 2 --qa-per-category 2 \
  --adversarial-mode paper \
  --locomo-concurrency 10 \
  --judge-model Qwen/Qwen3-8B --judge-concurrency 4 --run-model Qwen/Qwen3-8B \
  --out-dir eval_output/locomo_smoke_paper --run-tag locomo_smoke_paper --strict \
  > /home/apulis-dev/code/SimpleMem/eval_output/locomo_smoke_paper.log 2>&1 < /dev/null &
echo "smoke (paper adversarial) launched pid $!"
