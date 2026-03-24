import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset as load_dataset_hf


def setup_logging(log_dir: str, experiment_name: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config: Dict[str, Any]):
    model_name = config["model"]["name"]
    torch_dtype = getattr(torch, config["model"]["torch_dtype"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=None
    )

    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_dataset(config: Dict[str, Any]) -> List[str]:
    dataset_path = config["data"]["dataset_path"]
    text_column = config["data"]["text_column"]
    sample_count = config["data"].get("sample_count", None)

    dataset = load_dataset_hf("parquet", data_files=dataset_path, split="train")

    if sample_count is not None:
        total = len(dataset)
        if sample_count < total:
            dataset = dataset.select(range(sample_count))
            logging.info(
                f"Using only first {sample_count} samples (total available: {total})"
            )
        else:
            logging.warning(
                f"sample_count={sample_count} exceeds dataset size ({total}), using all samples."
            )

    texts = dataset[text_column]
    return list(texts)


def collate_fn(batch: List[str], tokenizer, max_seq_len: int):
    tokenized = tokenizer(
        batch,
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return tokenized["input_ids"], tokenized["attention_mask"]


def save_metrics(metrics: Dict[str, Any], output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def get_optimizer(model, config):
    optimizer_name = config["training"]["optimizer"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "muon":
        from src.muon import SingleDeviceMuonWithAuxAdam

        muon_params = []
        adamw_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        param_groups = []
        if muon_params:
            param_groups.append(
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": 0.95,
                }
            )
        if adamw_params:
            param_groups.append(
                {
                    "params": adamw_params,
                    "use_muon": False,
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                }
            )

        if not param_groups:
            raise ValueError("No trainable parameters found for MuonWithAuxAdam")

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        return optimizer
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
