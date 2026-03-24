#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=dtype_map[args.dtype],
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()