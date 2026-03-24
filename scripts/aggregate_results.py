#!/usr/bin/env python3
from pathlib import Path
import csv
import json
from statistics import mean


OUT_DIR = Path("outputs")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def safe_mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def workload_key(summary: dict):
    return (
        summary.get("epochs"),
        summary.get("num_train_samples"),
        summary.get("max_seq_len"),
        summary.get("micro_batch_size"),
        summary.get("grad_accum_steps"),
        summary.get("seed"),
    )


def main():
    all_step_rows = []
    run_rows = []

    for summary_path in sorted(OUT_DIR.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        train_rows = read_jsonl(run_dir / "train_log.jsonl")

        run_name = summary.get("run_name") or run_dir.name

        # collect all step rows
        for row in train_rows:
            all_step_rows.append({
                "run_name": run_name,
                "optimizer_name": summary.get("optimizer_name"),
                "learning_rate": summary.get("learning_rate"),
                "epochs": summary.get("epochs"),
                "num_train_samples": summary.get("num_train_samples"),
                "max_seq_len": summary.get("max_seq_len"),
                "micro_batch_size": summary.get("micro_batch_size"),
                "grad_accum_steps": summary.get("grad_accum_steps"),
                "effective_batch_size": summary.get("effective_batch_size"),
                **row,
            })

        losses = [r.get("loss") for r in train_rows if r.get("loss") is not None]
        lrs = [r.get("lr") for r in train_rows if r.get("lr") is not None]
        step_times = [r.get("step_time_sec") for r in train_rows if r.get("step_time_sec") is not None]
        opt_times = [r.get("optimizer_time_sec") for r in train_rows if r.get("optimizer_time_sec") is not None]
        peak_mems = [r.get("mem_peak_gb") for r in train_rows if r.get("mem_peak_gb") is not None]

        last_k = 10
        tail_losses = losses[-last_k:] if losses else []

        final_logged_loss = losses[-1] if losses else None
        min_logged_loss = min(losses) if losses else None
        avg_last_10_loss = safe_mean(tail_losses)
        final_logged_lr = lrs[-1] if lrs else None
        max_logged_mem_peak_gb = max(peak_mems) if peak_mems else None
        avg_step_time_sec = safe_mean(step_times)
        avg_optimizer_time_sec = safe_mean(opt_times)
        num_log_rows = len(train_rows)

        run_rows.append({
            "run_name": run_name,
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
            "num_log_rows": num_log_rows,
            "final_logged_loss": final_logged_loss,
            "min_logged_loss": min_logged_loss,
            "avg_last_10_loss": avg_last_10_loss,
            "final_logged_lr": final_logged_lr,
            "max_logged_mem_peak_gb": max_logged_mem_peak_gb,
            "avg_step_time_sec": avg_step_time_sec,
            "avg_optimizer_time_sec": avg_optimizer_time_sec,
        })

    # write all step logs
    if all_step_rows:
        step_fields = list(all_step_rows[0].keys())
        with open(LOG_DIR / "all_train_logs.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=step_fields)
            writer.writeheader()
            writer.writerows(all_step_rows)

    # write run metrics
    if run_rows:
        run_rows.sort(key=lambda x: (x["optimizer_name"] or "", x["run_name"] or ""))
        run_fields = list(run_rows[0].keys())
        with open(LOG_DIR / "run_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=run_fields)
            writer.writeheader()
            writer.writerows(run_rows)

        # best per optimizer per workload
        grouped = {}
        for row in run_rows:
            key = (
                row.get("optimizer_name"),
                row.get("epochs"),
                row.get("num_train_samples"),
                row.get("max_seq_len"),
                row.get("micro_batch_size"),
                row.get("grad_accum_steps"),
                row.get("seed"),
            )
            grouped.setdefault(key, []).append(row)

        best_rows = []
        for _, rows in grouped.items():
            rows = sorted(
                rows,
                key=lambda r: (
                    float("inf") if r.get("avg_last_10_loss") is None else r["avg_last_10_loss"],
                    float("inf") if r.get("final_logged_loss") is None else r["final_logged_loss"],
                    float("inf") if r.get("total_time_sec") is None else r["total_time_sec"],
                    float("inf") if r.get("peak_mem_gb") is None else r["peak_mem_gb"],
                )
            )
            best_rows.append(rows[0])

        best_rows.sort(key=lambda x: (x["optimizer_name"] or "", x["run_name"] or ""))
        with open(LOG_DIR / "best_runs.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=run_fields)
            writer.writeheader()
            writer.writerows(best_rows)

    print("Wrote:")
    print(f"  {LOG_DIR / 'all_train_logs.csv'}")
    print(f"  {LOG_DIR / 'run_metrics.csv'}")
    print(f"  {LOG_DIR / 'best_runs.csv'}")


if __name__ == "__main__":
    main()