"""
Summarize HH GPT judge results into a CSV for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = "test/gpt_judge_HH/config_evaluation.yaml"
DEFAULT_SUMMARY = "test/gpt_judge_HH/results/summary.json"


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_summary_paths(
    configs: list[str] | None,
    summaries: list[str] | None,
) -> list[tuple[Path, str]]:
    resolved: list[tuple[Path, str]] = []

    if summaries:
        for summary in summaries:
            summary_path = Path(summary)
            resolved.append((summary_path, summary_path.stem))
        return resolved

    config_paths = configs or [DEFAULT_CONFIG]
    for config_path in config_paths:
        config = _load_config(config_path)
        output_cfg = config.get("output", {})
        summary_path = Path(output_cfg.get("summary_file", DEFAULT_SUMMARY))
        run_id = Path(config_path).stem
        resolved.append((summary_path, run_id))
    return resolved


def _build_row(summary: dict[str, Any], summary_path: Path, run_id: str) -> dict[str, Any]:
    counts = summary.get("counts") or {}
    win_rates = summary.get("win_rates") or {}
    total = summary.get("total")
    if total is None:
        total = sum(counts.values()) if counts else 0
    if not win_rates and total:
        win_rates = {key: counts.get(key, 0) / total for key in counts}

    tie_count = counts.get("TIE", counts.get("tie", 0))
    tie_rate = win_rates.get("TIE", win_rates.get("tie", 0.0))

    return {
        "run_id": run_id,
        "summary_file": str(summary_path),
        "judge_model": summary.get("model", ""),
        "total": total,
        "sft_wins": counts.get("sft", 0),
        "og_dpo_wins": counts.get("og_dpo", 0),
        "dpo_wins": counts.get("dpo", 0),
        "ties": tie_count,
        "sft_win_rate": win_rates.get("sft", 0.0),
        "og_dpo_win_rate": win_rates.get("og_dpo", 0.0),
        "dpo_win_rate": win_rates.get("dpo", 0.0),
        "tie_rate": tie_rate,
    }


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "summary_file",
        "judge_model",
        "total",
        "sft_wins",
        "og_dpo_wins",
        "dpo_wins",
        "ties",
        "sft_win_rate",
        "og_dpo_win_rate",
        "dpo_win_rate",
        "tie_rate",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize HH judge results into a CSV."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Path to evaluation config YAML (can be passed multiple times).",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=None,
        help="Path to a summary JSON file (can be passed multiple times).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output CSV path.",
    )
    args = parser.parse_args()

    summary_paths = _resolve_summary_paths(args.config, args.summary)
    if not summary_paths:
        raise ValueError("No summary paths found.")

    rows: list[dict[str, Any]] = []
    for summary_path, run_id in summary_paths:
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        summary = _load_json(summary_path)
        rows.append(_build_row(summary, summary_path, run_id))

    output_path = (
        Path(args.output)
        if args.output
        else summary_paths[0][0].parent / "win_rates_summary.csv"
    )
    _write_csv(rows, output_path)
    print(f"Saved summary CSV to {output_path}")


if __name__ == "__main__":
    main()
