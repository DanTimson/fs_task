#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
export PYTHONUNBUFFERED=1

python -m src.train --config configs/adamw_lora_small.yaml 2>&1 | tee logs/adamw_small_train_stdout.log