#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "lm-evaluation-harness" ]; then
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
fi

cd lm-evaluation-harness
pip install -e ".[hf]"