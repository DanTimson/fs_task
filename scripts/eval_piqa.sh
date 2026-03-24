#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <merged_model_dir> <output_prefix>"
  exit 1
fi

MODEL_DIR="$1"
OUT_PREFIX="$2"

mkdir -p "$(dirname "$OUT_PREFIX")"

lm_eval \
  --model hf \
  --model_args pretrained="$MODEL_DIR",dtype=bfloat16 \
  --tasks piqa \
  --device cuda:0 \
  --batch_size auto \
  --output_path "$OUT_PREFIX"