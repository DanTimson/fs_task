#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi

CONFIG="$1"

export PYTHONUNBUFFERED=1

python -m src.train --config "$CONFIG"