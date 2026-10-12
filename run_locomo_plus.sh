#!/bin/bash
# Detached Launcher: SimpleMem on LoCoMo-Plus (Cognitive category only, 401 samples).
# judge = run-model = Qwen/Qwen3-8B (SiliconFlow API). The original LoCoMo 5-category QA
# is skipped (--skip-locomo) so this is a focused LoCoMo-Plus experiment.
# Runs detached (setsid) so it survives the SSH session.
#
# Zero-429 posture (SiliconFlow free tier: 1000 RPM / 50K TPM, account level):
#   - PRIMARY defense: client-side token-bucket throttle (RateLimiter in
#     simplemem/core/utils/llm_client.py) caps OFFERED load at 300 RPM / 30K TPM
#     (= 30% / 60% of the caps) across ALL LLM paths (memory builder, answer
#     generator, judge) via a process-wide singleton. The sliding window never
#     sees an excess, so no request is ever rejected. Wall clock ~12h
#     (20.2M tokens / 30K TPM floor ~11h); correctness first.
#   - cognitive-concurrency 20 is now just parallelism: the throttle, not the
#     worker count, paces the offered load (48 workers without throttle were
#     measured offering 61K TPM = 123% of the cap -> 99% rejected).
#   - SAFETY NET: up to 10 attempts per call with jittered exponential backoff
#     (LLM_* / JUDGE_* settings in simplemem/core/settings.py).
#   - ACCOUNTING: every real API attempt is counted; summary.json reports
#     n_llm_calls / n_429 / rate_limiter. --strict exits with code 1 if ANY
#     sample errors OR any 429 occurs, so a throttled or half-failed run can
#     never be mistaken for a clean result.
#   - Fresh out-dir (locomo_plus_v3), no --resume: previous runs' outputs stay
#     untouched for comparison, and no degraded record can leak into metrics.
cd /home/apulis-dev/code/SimpleMem
source /home/apulis-dev/code/envs/simplemem/bin/activate || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p eval_output
setsid nohup python eval/run_simplemem_eval.py \
  --skip-locomo \
  --cognitive-concurrency 20 \
  --judge-model Qwen/Qwen3-8B --judge-concurrency 8 --run-model Qwen/Qwen3-8B \
  --out-dir eval_output/locomo_plus_v3 --run-tag locomo_plus_v3 --strict \
  > /home/apulis-dev/code/SimpleMem/eval_output/locomo_plus_v3_run.log 2>&1 < /dev/null &
echo "launched LoCoMo-Plus (Cognitive) eval, pid $!"
