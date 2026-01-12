"""
Evaluate a model with AlpacaEval 2.0.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model with AlpacaEval 2.0")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test/alpacaeval",
        help="Output directory for outputs/results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Leaderboard name override",
    )
    parser.add_argument(
        "--annotators_config",
        type=str,
        default="weighted_alpaca_eval_gpt4_turbo",
        help="AlpacaEval annotators config",
    )
    parser.add_argument(
        "--reference_outputs",
        type=str,
        default="gpt4_turbo",
        help="Reference outputs name",
    )
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation step")
    parser.add_argument("--skip_eval", action="store_true", help="Skip AlpacaEval step")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip analysis step")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        choices=["cpu", "cuda", "mps"],
        help="Device for generation",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max_instances", type=int, default=None, help="Limit number of prompts")
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="tatsu-lab/alpaca_eval",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Local dataset file path (JSON/JSONL/Parquet)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    outputs_dir = output_dir / "outputs"
    results_dir = output_dir / "results"
    outputs_file = outputs_dir / "model_outputs.json"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    generate_script = script_dir / "generate_outputs.py"
    analyze_script = script_dir / "analyze_results.py"

    leaderboard_name = args.name or args.model_name.split("/")[-1]

    if not args.skip_generation:
        cmd = [
            sys.executable,
            str(generate_script),
            "--model_name",
            args.model_name,
            "--output_file",
            str(outputs_file),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--dataset_repo",
            args.dataset_repo,
        ]
        if args.max_instances is not None:
            cmd.extend(["--max_instances", str(args.max_instances)])
        if args.data_file:
            cmd.extend(["--data_file", args.data_file])
        _run(cmd)
    elif not outputs_file.exists():
        raise FileNotFoundError(
            f"Outputs not found at {outputs_file}. Run without --skip_generation."
        )

    if not args.skip_eval:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set. Skipping AlpacaEval.")
            return

        cmd = [
            "alpaca_eval",
            "--model_outputs",
            str(outputs_file),
            "--annotators_config",
            args.annotators_config,
            "--reference_outputs",
            args.reference_outputs,
            "--output_path",
            str(results_dir),
            "--name",
            leaderboard_name,
        ]
        if args.max_instances is not None:
            cmd.extend(["--max_instances", str(args.max_instances)])
        _run(cmd)

    if not args.skip_analysis:
        cmd = [
            sys.executable,
            str(analyze_script),
            "--results_dir",
            str(results_dir),
            "--model_name",
            leaderboard_name,
        ]
        _run(cmd)


if __name__ == "__main__":
    main()
