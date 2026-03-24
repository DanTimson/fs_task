#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

python --version | tee logs/python_version.txt
pip freeze | tee logs/pip_freeze.txt

python - <<'PY' | tee logs/torch_env.txt
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device_name:", torch.cuda.get_device_name(0))
PY