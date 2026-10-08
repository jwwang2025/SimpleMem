#!/bin/bash
# Detached Launcher: SimpleMem on LoCoMo-Plus (Cognitive category only, 401 samples).
# judge = run-model = Qwen3-8B (local :8000). The original LoCoMo 5-category QA
# is skipped (--skip-locomo) so this is a focused LoCoMo-Plus experiment.
# Runs detached (setsid) so it survives the SSH session.
cd /home/apulis-dev/code/SimpleMem
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p eval_output
setsid nohup python eval/run_simplemem_eval.py \
  --skip-locomo \
  --judge-model Qwen3-8B --judge-concurrency 8 --run-model Qwen3-8B \
  --out-dir eval_output/locomo_plus --run-tag locomo_plus --resume \
  > /home/apulis-dev/code/SimpleMem/eval_output/locomo_plus_run.log 2>&1 < /dev/null &
echo "launched LoCoMo-Plus (Cognitive) eval, pid $!"
