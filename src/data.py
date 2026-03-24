from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import torch
from datasets import load_dataset


def load_model_and_tokenizer(cfg, device: torch.device):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[cfg["model"]["dtype"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch_dtype,
        device_map=None,
    )

    lora_cfg = cfg["lora"]
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.to(device)

    return model, tokenizer

def build_train_subset(cfg: dict, seed: int):
    ds = load_dataset(cfg["data"]["dataset_name"], split="train")
    ds = ds.shuffle(seed=seed)

    frac = cfg["data"]["train_fraction"]
    n_frac = max(1, int(len(ds) * frac))
    n = min(n_frac, cfg["data"].get("max_samples", n_frac))

    ds = ds.select(range(n))
    return ds


def tokenize_dataset(ds, tokenizer, text_column: str, max_seq_len: int):
    def _tok(example):
        return tokenizer(
            example[text_column],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_attention_mask=True,
        )

    tokenized = ds.map(_tok, remove_columns=ds.column_names)
    return tokenized