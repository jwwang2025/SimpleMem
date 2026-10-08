#!/bin/bash
set -e
cd /home/apulis-dev/code/SimpleMem
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen
python eval/run_simplemem_eval.py \
  --limit-conv 1 --qa-per-category 1 --cognitive-limit 1 \
  --judge-model Qwen3-8B --judge-concurrency 4 --run-model Qwen3-8B \
  --out-dir eval_output/smoke --run-tag smoke
