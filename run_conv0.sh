#!/bin/bash
# Detached Launcher: SimpleMem LoCoMo eval on conversation 0 (all its QA),
# judge = Qwen3-8B (local :8000). Skips the 401 cognitive samples.
# Runs detached (setsid) so it survives the SSH session.
cd /home/apulis-dev/code/SimpleMem
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen || exit 1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p eval_output
setsid nohup python eval/run_simplemem_eval.py \
  --limit-conv 1 --skip-cognitive \
  --judge-model Qwen3-8B --judge-concurrency 8 --run-model Qwen3-8B \
  --out-dir eval_output/conv0 --run-tag conv0 --resume \
  > /home/apulis-dev/code/SimpleMem/eval_output/conv0_run.log 2>&1 < /dev/null &
echo "launched conv0 eval, pid $!"
