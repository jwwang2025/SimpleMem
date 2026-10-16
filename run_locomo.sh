#!/bin/bash
# Detached Launcher: SimpleMem on LoCoMo-10 (original 5-category QA).
# 10 conversations / 1,986 questions:
#   1 single-hop(282)  2 multi-hop(321)  3 temporal(96)
#   4 open-domain(841) 5 adversarial(446) - category 5 runs the PAPER
#   protocol (--adversarial-mode paper, the default): reference is the fixed
#   string "Not mentioned in the conversation", retrieval with reflection
#   disabled, forced binary-choice generation (refusal vs the dataset's
#   adversarial_answer) at temperature 0.5, generic Relevance & Accuracy
#   judge rubric. Mirrors the SimpleMem paper's reference implementation
#   (test_locomo10.py) so our category-5 numbers are protocol-compatible.
#   (--adversarial-mode open restores the legacy open-ended + Skeptical
#   Judge behaviour; the two are NOT comparable to each other.)
# LoCoMo-Plus cognitive is skipped (--skip-cognitive) so this experiment is
# fully isolated from the LoCoMo-Plus run:
#   - separate out-dir => separate predictions.jsonl / judged.jsonl /
#     summary.json and a separate --resume state
#   - memory stores live at eval_output/locomo_v1/db/conv_N (the Plus run
#     used eval_output/locomo_plus_v3/db/cognitive_N) - no shared files
#   - LoCoMo memory stores are KEPT after each conversation (unlike the
#     cognitive path which force-rmtree's them), so a re-run with --resume
#     still has to rebuild unless you pass --keep-cognitive-db semantics;
#     simply let it rebuild - builds are ~13 extraction windows per conv.
#
# Zero-429 posture (identical mechanism to the Plus run; SiliconFlow free tier
# 1000 RPM / 50K TPM, ACCOUNT level):
#   - client-side token-bucket throttle caps OFFERED load at 300 RPM / 36K TPM
#     across ALL LLM paths via a process-wide singleton (debt-based accounting
#     + EWMA calibration against real usage). 36K = 72% of the account cap;
#     the Plus batch ran 24K with zero 429, and the limiter now tracks the
#     budget almost exactly, so 36K keeps ~28% headroom. The LoCoMo batch is
#     10 CONCURRENT conversations (--locomo-concurrency 10): all workers share
#     the one limiter, so the whole process stays inside budget no matter how
#     many conversations are in flight (~3.5 QA/min).
#   - up to 10 attempts per call with jittered exponential backoff
#   - --strict exits 1 on ANY sample error OR ANY 429
#   IMPORTANT: the account cap is shared across processes. Do NOT run this
#   concurrently with another eval unless BOTH budgets are lowered so the sum
#   stays under ~45K (e.g. 16K + 24K). LLM_TPM_BUDGET can be overridden per
#   process via the environment, e.g. LLM_TPM_BUDGET=16000 bash run_locomo.sh
#
# Usage:
#   bash run_locomo.sh                 # full run: 10 convs x all 1,986 QA (~10-12h
#                                      #   at 36K TPM, 10-way conversation parallelism)
#   PILOT=1 bash run_locomo.sh         # quick validation: 2 convs x 20/cat (~1h)
#   (tweak PILOT_CONV / PILOT_QA below for other pilot sizes)
cd /home/apulis-dev/code/SimpleMem
source /home/apulis-dev/code/envs/simplemem/bin/activate || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p eval_output

EXTRA_ARGS=""
if [ "${PILOT:-0}" = "1" ]; then
  # --limit-conv N        : only the first N conversations
  # --qa-per-category N   : sample N questions per category per conversation
  #                         (seed = 42 + conv_index, fixed => --resume consistent)
  EXTRA_ARGS="--limit-conv ${PILOT_CONV:-2} --qa-per-category ${PILOT_QA:-20}"
  echo "PILOT mode: ${EXTRA_ARGS}"
fi

setsid nohup python eval/run_simplemem_eval.py \
  --skip-cognitive --resume ${EXTRA_ARGS} \
  --adversarial-mode paper \
  --locomo-concurrency ${LOCOMO_CONCURRENCY:-10} \
  --judge-model Qwen/Qwen3-8B --judge-concurrency 8 --run-model Qwen/Qwen3-8B \
  --out-dir eval_output/locomo_v1 --run-tag locomo_v1 --strict \
  > /home/apulis-dev/code/SimpleMem/eval_output/locomo_v1_run1.log 2>&1 < /dev/null &
echo "launched LoCoMo-10 eval (resume, ${LOCOMO_CONCURRENCY:-10}-way conversation parallelism), pid $!"
