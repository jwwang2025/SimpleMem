#!/bin/bash
# Detached Launcher: SimpleMem on LoCoMo-Plus (Cognitive category only, 401 samples).
# judge = run-model = Qwen/Qwen3-8B (SiliconFlow API). The original LoCoMo 5-category QA
# is skipped (--skip-locomo) so this is a focused LoCoMo-Plus experiment.
# Runs detached (setsid) so it survives the SSH session.
#
# Rate-limit posture (SiliconFlow free tier: 1000 RPM / 50K TPM, account level):
#   - cognitive-concurrency 28 -> offered TPM ~34K (about 70% of the 50K cap),
#     leaving headroom for bursts; expected 429 rate < 5% and no hard failures.
#     (48 was measured to offer ~61K = 123% of the cap -> 99% of requests 429'd
#     and 93% of extraction windows silently produced zero entries.)
#   - Every LLM call now retries up to 10 times with jittered exponential
#     backoff (LLM_* / JUDGE_* settings in simplemem/core/settings.py).
#   - --strict: exit code 1 if any sample ends in error, so a throttled or
#     half-failed run can never be mistaken for a clean result.
#   - Fresh out-dir (locomo_plus_v2), no --resume: the previous run's output
#     stays untouched for comparison, and no silently-degraded record from the
#     old run can leak into the new metrics.
cd /home/apulis-dev/code/SimpleMem
source /home/apulis-dev/code/envs/simplemem/bin/activate || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p eval_output
setsid nohup python eval/run_simplemem_eval.py \
  --skip-locomo \
  --cognitive-concurrency 28 \
  --judge-model Qwen/Qwen3-8B --judge-concurrency 8 --run-model Qwen/Qwen3-8B \
  --out-dir eval_output/locomo_plus_v2 --run-tag locomo_plus_v2 --strict \
  > /home/apulis-dev/code/SimpleMem/eval_output/locomo_plus_v2_run.log 2>&1 < /dev/null &
echo "launched LoCoMo-Plus (Cognitive) eval, pid $!"
