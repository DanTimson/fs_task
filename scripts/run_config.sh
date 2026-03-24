#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <config>"
  exit 1
fi

mkdir -p logs
export PYTHONUNBUFFERED=1

CONFIG="$1"
NAME="$(basename "$CONFIG" .yaml)"

python -m src.train --config "$CONFIG" 2>&1 | tee "logs/${NAME}.log"