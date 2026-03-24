#!/usr/bin/env bash
set -euo pipefail

./scripts/run_config.sh configs/adamwlr1e4.yaml
./scripts/run_config.sh configs/adamwlr3e4.yaml
./scripts/run_config.sh configs/adamwlr1e3.yaml

./scripts/run_config.sh configs/muon_lr1e3.yaml
./scripts/run_config.sh configs/muon_lr3e3.yaml
./scripts/run_config.sh configs/muon_lr1e2.yaml