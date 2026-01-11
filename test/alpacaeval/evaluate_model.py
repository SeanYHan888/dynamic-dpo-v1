"""
Generate AlpacaEval outputs, run judging, and analyze results.
"""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with AlpacaEval")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="test/alpacaeval")
    parser.add_argument("--skip_generation", action="store_true")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    outputs_file = os.path.join(args.output_dir, "outputs", "model_outputs.json")
    results_dir = os.path.join(args.output_dir, "results")

    if not args.skip_generation or not os.path.exists(outputs_file):
        subprocess.run(
            [
                "python",
                "test/alpacaeval/generate_outputs.py",
                "--model_name",
                args.model_name,
                "--output_file",
                outputs_file,
            ],
            check=True,
        )

    subprocess.run(
        [
            "alpaca_eval",
            "--model_outputs",
            outputs_file,
            "--annotators_config",
            "weighted_alpaca_eval_gpt4_turbo",
            "--output_path",
            results_dir,
            "--name",
            args.model_name.split("/")[-1],
        ],
        check=True,
    )

    subprocess.run(
        [
            "python",
            "test/alpacaeval/analyze_results.py",
            "--results_dir",
            results_dir,
            "--model_name",
            args.model_name.split("/")[-1],
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
