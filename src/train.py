import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.collate import collate_batch
from src.data import build_train_subset, tokenize_dataset
from src.modeling import load_model_and_tokenizer
from src.optim.adamw import build_adamw
from src.optim.muon import build_muon


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(cfg: dict):
    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["experiment"]["log_dir"], exist_ok=True)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train(cfg: dict):
    ensure_dirs(cfg)
    save_json(cfg, os.path.join(cfg["experiment"]["output_dir"], "config_used.json"))
    set_seed(cfg["experiment"]["seed"])
    device = pick_device()

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model_and_tokenizer(cfg, device)

    train_ds = build_train_subset(cfg, seed=cfg["experiment"]["seed"])
    train_ds = tokenize_dataset(
        train_ds,
        tokenizer=tokenizer,
        text_column=cfg["data"]["text_column"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )

    dataloader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["micro_batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=lambda b: collate_batch(b, tokenizer),
        pin_memory=False,
    )

    optimizer_name = cfg["optimizer"]["name"].lower()

    if optimizer_name == "adamw":
        optimizer = build_adamw(model, cfg)
    elif optimizer_name == "muon":
        optimizer = build_muon(model, cfg)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    grad_accum = cfg["data"]["grad_accum_steps"]
    total_update_steps = math.ceil(len(dataloader) / grad_accum) * cfg["training"]["epochs"]
    warmup_steps = int(total_update_steps * cfg["training"]["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    log_path = os.path.join(cfg["experiment"]["output_dir"], "train_log.jsonl")
    summary_path = os.path.join(cfg["experiment"]["output_dir"], "summary.json")

    use_autocast = cfg["model"]["dtype"] in ("float16", "bfloat16")
    autocast_dtype = torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float16

    model.train()
    optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    global_step = 0
    optimizer_step = 0
    running_loss = 0.0
    start_time = time.time()

    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(cfg["training"]["epochs"]):
            for batch in dataloader:
                step_start = time.time()

                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=autocast_dtype,
                    enabled=use_autocast and torch.cuda.is_available(),
                ):
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum

                loss.backward()
                running_loss += loss.item() * grad_accum
                global_step += 1

                did_step = (global_step % grad_accum == 0)

                if did_step:
                    if cfg["training"]["max_grad_norm"] is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            cfg["training"]["max_grad_norm"],
                        )

                    opt_start = time.time()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_time = time.time() - opt_start
                    optimizer_step += 1
                else:
                    opt_time = 0.0

                step_time = time.time() - step_start

                if torch.cuda.is_available():
                    mem_alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
                    mem_peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                else:
                    mem_alloc_gb = 0.0
                    mem_reserved_gb = 0.0
                    mem_peak_gb = 0.0

                if global_step % cfg["training"]["log_every"] == 0:
                    row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "did_optimizer_step": did_step,
                        "loss": running_loss / cfg["training"]["log_every"],
                        "lr": scheduler.get_last_lr()[0],
                        "step_time_sec": step_time,
                        "optimizer_time_sec": opt_time,
                        "mem_alloc_gb": mem_alloc_gb,
                        "mem_reserved_gb": mem_reserved_gb,
                        "mem_peak_gb": mem_peak_gb,
                        "elapsed_sec": time.time() - start_time,
                    }
                    print(row)
                    log_file.write(json.dumps(row) + "\n")
                    log_file.flush()
                    running_loss = 0.0

    total_time = time.time() - start_time

    model.save_pretrained(cfg["experiment"]["output_dir"])
    tokenizer.save_pretrained(cfg["experiment"]["output_dir"])

    summary = {
        "total_time_sec": total_time,
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if torch.cuda.is_available()
            else 0.0
        ),
        "optimizer_steps": optimizer_step,
        "device": str(device),
    }
    save_json(summary, summary_path)

    print("Training complete.")
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()