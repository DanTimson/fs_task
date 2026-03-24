#!/usr/bin/env bash
set -euo pipefail

python - <<'PY' > /tmp/best_eval_runs.txt
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("configs/best_eval.yaml").read_text())

for name, spec in cfg.items():
    print(f"{name}|{spec['adapter_dir']}|{spec['merged_dir']}|{spec['piqa_prefix']}")
PY

while IFS='|' read -r name adapter_dir merged_dir piqa_prefix; do
  echo "=== Exporting merged model for $name ==="
  python scripts/export_merged_model.py \
    --adapter_dir "$adapter_dir" \
    --output_dir "$merged_dir" \
    --dtype bfloat16

  echo "=== Evaluating PIQA for $name ==="
  bash scripts/eval_piqa.sh "$merged_dir" "$piqa_prefix"
done < /tmp/best_eval_runs.txt