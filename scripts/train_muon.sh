#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
export PYTHONUNBUFFERED=1

python -m src.train --config configs/muon_lora.yaml 2>&1 | tee logs/muon_train_stdout.log