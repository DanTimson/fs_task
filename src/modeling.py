import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def load_model_and_tokenizer(cfg: dict, device: torch.device):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[cfg["model"]["dtype"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        dtype=dtype,
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