#!/usr/bin/env python3
from pathlib import Path
import csv
import json


OUT_DIR = Path("outputs")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_last_train_log_row(run_dir: Path) -> dict:
    log_path = run_dir / "train_log.jsonl"
    if not log_path.exists():
        return {}

    last = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)

    return last or {}


def main():
    rows = []

    for summary_path in sorted(OUT_DIR.glob("*/summary.json")):
        run_dir = summary_path.parent

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        last_log = load_last_train_log_row(run_dir)

        row = {
            "run_name": summary.get("run_name"),
            "optimizer_name": summary.get("optimizer_name"),
            "learning_rate": summary.get("learning_rate"),
            "epochs": summary.get("epochs"),
            "num_train_samples": summary.get("num_train_samples"),
            "max_seq_len": summary.get("max_seq_len"),
            "micro_batch_size": summary.get("micro_batch_size"),
            "grad_accum_steps": summary.get("grad_accum_steps"),
            "effective_batch_size": summary.get("effective_batch_size"),
            "total_time_sec": summary.get("total_time_sec"),
            "peak_mem_gb": summary.get("peak_mem_gb"),
            "optimizer_steps": summary.get("optimizer_steps"),
            "seed": summary.get("seed"),
            "device": summary.get("device"),
            "final_logged_loss": last_log.get("loss"),
            "final_logged_lr": last_log.get("lr"),
            "final_logged_mem_peak_gb": last_log.get("mem_peak_gb"),
        }
        rows.append(row)

    rows.sort(key=lambda x: (x.get("optimizer_name") or "", x.get("run_name") or ""))

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
        "final_logged_loss",
        "final_logged_lr",
        "final_logged_mem_peak_gb",
    ]

    out_csv = LOG_DIR / "summary_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()