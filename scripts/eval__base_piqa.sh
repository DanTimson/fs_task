#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-0.5B"
OUT_PREFIX="logs/qwen25_0p5b_base_piqa"

mkdir -p logs

lm_eval \
  --model hf \
  --model_args pretrained="$MODEL_ID",dtype=bfloat16 \
  --tasks piqa \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size auto \
  --output_path "$OUT_PREFIX"