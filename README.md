# LLM Optimizer Comparison for LoRA Fine-Tuning

This repository contains a reproducible comparison of **AdamW** and **Muon** for **LoRA fine-tuning** of **Qwen2.5-0.5B** on a subset of **openwebtext-100k**, with downstream evaluation on **PIQA**.

The project was built for a technical task with the following core requirements:

- model: **Qwen/Qwen2.5-0.5B**
- training dataset: at least **1% of openwebtext-100k**
- evaluation dataset: **PIQA**
- comparison axes:
  - training time
  - GPU memory usage
  - training convergence
  - final model quality

## Summary

The base pipeline in this repository:

1. fine-tunes Qwen2.5-0.5B with LoRA using AdamW and Muon
2. logs per-step loss, learning rate, timing, and memory
3. aggregates run metrics into compact CSV tables
4. selects the best runs
5. evaluates:
   - the unfine-tuned base model
   - the best AdamW fine-tuned model
   - the best Muon fine-tuned model
## Setup

### 1. Install `uv`

This project uses `uv` for dependency synchronization.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell if needed so that `uv` is available in `PATH`.

### 2. Create and sync the environment

The setup script supports explicit backend selection:

- `cuda`
- `rocm`
- `cpu`

Examples:

```bash
TORCH_BACKEND=cuda bash setup_env.sh
TORCH_BACKEND=rocm bash setup_env.sh
TORCH_BACKEND=cpu bash setup_env.sh
```

To also predownload the model and dataset from Hugging Face:

```bash
DOWNLOAD_ASSETS=1 TORCH_BACKEND=cuda bash setup_env.sh
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

## Reproducibility

### Step 1. Run training experiments

To launch the LR sweep:

```bash
bash scripts/run_lr_experiment.sh
```

To run a single config directly:

```bash
bash scripts/run_config.sh configs/<config_name>.yaml
```

Example:

```bash
bash scripts/run_config.sh configs/adamw_lora_lr1e3.yaml
bash scripts/run_config.sh configs/muon_lora_lr3e3.yaml
```

### Step 2. Aggregate training results

```bash
python scripts/aggregate_results.py
```

This produces compact result files in `logs/`, typically including:

- `run_metrics.csv`
- `best_runs.csv`

Depending on the current script version, additional aggregate files may also be produced.

## PIQA evaluation

### Step 3. Install evaluation dependencies

```bash
bash scripts/install_eval.sh
```

### Step 4. Evaluate the unfine-tuned base model

```bash
bash scripts/eval_base_piqa.sh
```

### Step 5. Evaluate the selected fine-tuned runs

The selected checkpoints are defined in:

```text
configs/best_eval.yaml
```

Update that file so it points to the chosen best AdamW and Muon runs, then run:

```bash
bash scripts/eval_best_runs.sh
```

This script:

1. merges LoRA adapters into the base model
2. evaluates the merged checkpoints on PIQA

### Step 6. Collect PIQA results

```bash
python scripts/collect_piqa_results.py
```

This produces:

- `logs/piqa_summary.csv`

## Typical full workflow

From a clean environment, the base pipeline can be reproduced with:

```bash
source .venv/bin/activate
bash scripts/run_lr_experiment.sh
python scripts/aggregate_results.py
bash scripts/install_eval.sh
bash scripts/eval_base_piqa.sh
bash scripts/eval_best_runs.sh
python scripts/collect_piqa_results.py
```

## Generated artifacts

Generated files are typically written to:

- `outputs/` — checkpoints and merged models
- `logs/` — logs, CSV summaries, evaluation JSONs

These directories are treated as reproducible generated artifacts and may be excluded from Git tracking.

## Main experiment design

The base comparison is performed under matched settings:

- same model
- same LoRA configuration
- same number of epochs
- same number of samples
- same sequence length
- same batch size / effective batch size
- same random seed

Only the optimizer and optimizer-specific learning rate differ.

## Report

Report layout is:

```text
report/
  report.tex
  report.pdf
  figures/
  tables/
```

## Acknowledgments

The Muon implementation in this repository is adapted from the upstream Muon code referenced in the source headers.
