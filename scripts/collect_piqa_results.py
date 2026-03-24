#!/usr/bin/env python3
from pathlib import Path
import csv
import json
import yaml


def resolve_latest_json(prefix: str) -> Path:
    prefix_path = Path(prefix)

    if prefix_path.is_file():
        return prefix_path

    direct_matches = sorted(prefix_path.parent.glob(prefix_path.name + "*.json"))
    if direct_matches:
        return max(direct_matches, key=lambda p: p.stat().st_mtime)

    if prefix_path.exists() and prefix_path.is_dir():
        nested_matches = sorted(prefix_path.rglob("results_*.json"))
        if nested_matches:
            return max(nested_matches, key=lambda p: p.stat().st_mtime)

    parent = prefix_path.parent if prefix_path.parent != Path("") else Path(".")
    candidate_dir = parent / prefix_path.name
    if candidate_dir.exists() and candidate_dir.is_dir():
        nested_matches = sorted(candidate_dir.rglob("results_*.json"))
        if nested_matches:
            return max(nested_matches, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"No JSON results found for prefix: {prefix}")


def main():
    cfg = yaml.safe_load(Path("configs/best_eval.yaml").read_text())
    rows = []

    for name, spec in cfg.items():
        json_path = resolve_latest_json(spec["piqa_prefix"])

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", {}).get("piqa", {})
        row = {
            "run_label": name,
            "model_kind": "base" if name == "base" else "finetuned",
            "piqa_acc": results.get("acc,none"),
            "piqa_acc_stderr": results.get("acc_stderr,none"),
            "piqa_norm_acc": results.get("acc_norm,none"),
            "piqa_norm_acc_stderr": results.get("acc_norm_stderr,none"),
            "source_json": str(json_path),
        }
        rows.append(row)

    out_path = Path("logs/piqa_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()