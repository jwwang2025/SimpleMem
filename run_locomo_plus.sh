#!/bin/bash
# Detached Launcher: SimpleMem on LoCoMo-Plus (Cognitive category only, 401 samples).
# judge = run-model = Qwen/Qwen3-8B (SiliconFlow API). The original LoCoMo 5-category QA
# is skipped (--skip-locomo) so this is a focused LoCoMo-Plus experiment.
# Runs detached (setsid) so it survives the SSH session.
#
# Zero-429 posture (SiliconFlow free tier: 1000 RPM / 50K TPM, account level):
#   - PRIMARY defense: client-side token-bucket throttle (RateLimiter in
#     simplemem/core/utils/llm_client.py) caps OFFERED load at 300 RPM / 24K TPM
#     (= 30% / 48% of the caps) across ALL LLM paths (memory builder, answer
#     generator, judge) via a process-wide singleton. Estimates are calibrated
#     against real response usage (EWMA) because Qwen3's actual output ran
#     ~1.5x the static estimate - without calibration the true offered TPM
#     exceeded the cap and drew steady-state 429s (observed 00:04-00:25).
#   - cognitive-concurrency 20 is just parallelism; the throttle paces the
#     offered load (48 workers without throttle offered 61K TPM = 123% of cap).
#   - SAFETY NET: up to 10 attempts per call with jittered exponential backoff.
#   - ACCOUNTING: summary.json reports n_llm_calls / n_429 / rate_limiter;
#     --strict exits 1 on ANY sample error OR ANY 429.
#   - v3 restart (00:30): first 2h ran with the un-calibrated 30K budget and
#     drew 252 x 429 (all retried away, no hard failures, 38 samples done and
#     kept). This relaunch resumes those 38 samples via --resume; the previous
#     run log is kept as locomo_plus_v3_run.log.
cd /home/apulis-dev/code/SimpleMem
source /home/apulis-dev/code/envs/simplemem/bin/activate || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p eval_output
setsid nohup python eval/run_simplemem_eval.py \
  --skip-locomo --resume \
  --cognitive-concurrency 20 \
  --judge-model Qwen/Qwen3-8B --judge-concurrency 8 --run-model Qwen/Qwen3-8B \
  --out-dir eval_output/locomo_plus_v3 --run-tag locomo_plus_v3 --strict \
  > /home/apulis-dev/code/SimpleMem/eval_output/locomo_plus_v3_run2.log 2>&1 < /dev/null &
echo "launched LoCoMo-Plus (Cognitive) eval (resume), pid $!"
