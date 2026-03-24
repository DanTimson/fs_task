#!/usr/bin/env python3
from pathlib import Path
import json
import csv

OUT_DIR = Path("outputs")
rows = []

for summary_path in sorted(OUT_DIR.glob("*/summary.json")):
    with open(summary_path, "r", encoding="utf-8") as f:
        rows.append(json.load(f))

rows.sort(key=lambda x: x["run_name"])

fieldnames = [
    "run_name",
    "optimizer_name",
    "learning_rate",
    "epochs",
    "num_train_samples",
    "max_seq_len",
    "micro_batch_size",
    "grad_accum_steps",
    "effective_batch_size",
    "total_time_sec",
    "peak_mem_gb",
    "optimizer_steps",
    "seed",
    "device",
]

with open("logs/summary_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})

print("Wrote logs/summary_table.csv")