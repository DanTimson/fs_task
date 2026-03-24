#!/usr/bin/env bash
set -euo pipefail

./scripts/run_config.sh configs/adamw_lora_lr1e4.yaml
./scripts/run_config.sh configs/adamw_lora_lr3e4.yaml
./scripts/run_config.sh configs/adamw_lora_lr1e3.yaml
./scripts/run_config.sh configs/adamw_lora_lr3e3.yaml

./scripts/run_config.sh configs/muon_lora_lr1e3.yaml
./scripts/run_config.sh configs/muon_lora_lr3e3.yaml
./scripts/run_config.sh configs/muon_lora_lr3e4.yaml
./scripts/run_config.sh configs/muon_lora_lr1e2.yaml