#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   TORCH_BACKEND=cuda bash setup_env.sh
#   TORCH_BACKEND=rocm bash setup_env.sh
#   TORCH_BACKEND=cpu  bash setup_env.sh
#
# Optional:
#   DOWNLOAD_ASSETS=1 bash setup_env.sh

TORCH_BACKEND="${TORCH_BACKEND:-}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"

echo "> Environment info"
uname -a || true
python3 --version || true
env | sort

if [ -z "${TORCH_BACKEND}" ]; then
  echo "TORCH_BACKEND is not set."
  echo "Use one of: cuda, rocm, cpu"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "> Syncing project dependencies with uv"
uv sync

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "> Installing PyTorch backend: ${TORCH_BACKEND}"
case "$TORCH_BACKEND" in
  cuda)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ;;
  rocm)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    ;;
  cpu)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ;;
  *)
    echo "Unknown TORCH_BACKEND: ${TORCH_BACKEND}"
    exit 1
    ;;
esac

echo "> Verifying torch"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device_name:", torch.cuda.get_device_name(0))
PY

if [ "$DOWNLOAD_ASSETS" = "1" ]; then
  echo "> Downloading HF assets"
  hf download Qwen/Qwen2.5-0.5B --local-dir ./models/Qwen2.5-0.5B
  hf download Elriggs/openwebtext-100k --repo-type dataset --local-dir ./datasets/openwebtext-100k
fi

echo "> Done"