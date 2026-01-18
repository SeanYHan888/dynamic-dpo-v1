"""
Evaluate a model with AlpacaEval 2.0.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _tokenize(value: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", value.lower()) if part]


def _matches_model(query: str, candidate: str) -> bool:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return False
    candidate_lower = candidate.lower()
    return all(token in candidate_lower for token in query_tokens)


def _sanitize_generators(outputs_df):
    for column in ("generator", "generator_1", "generator_2"):
        if column in outputs_df.columns:
            outputs_df[column] = outputs_df[column].apply(
                lambda value: re.sub(r"[\\\\/]+", "_", str(value))
            )
    return outputs_df


def _load_and_sanitize_outputs(path: str | Path):
    from alpaca_eval import utils as alpaca_utils

    outputs_df = alpaca_utils.load_or_convert_to_dataframe(path)
    return _sanitize_generators(outputs_df)


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def _write_csv_rows(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_leaderboard_rows(
    source_path: Path,
    target_path: Path,
    existing_rows: list[dict[str, str]],
    existing_fieldnames: list[str],
) -> None:
    if not source_path.exists():
        return

    new_rows, new_fieldnames = _read_csv_rows(source_path)
    if not new_rows:
        return

    fieldnames = list(existing_fieldnames)
    for name in new_fieldnames:
        if name not in fieldnames:
            fieldnames.append(name)
    if not fieldnames:
        fieldnames = list(new_fieldnames)

    if source_path == target_path:
        if not existing_rows:
            return
        merged_rows = existing_rows + new_rows
        _write_csv_rows(target_path, merged_rows, fieldnames)
        return

    if not target_path.exists():
        _write_csv_rows(target_path, new_rows, fieldnames)
        return

    if set(new_fieldnames) - set(existing_fieldnames):
        merged_rows = existing_rows + new_rows
        _write_csv_rows(target_path, merged_rows, fieldnames)
        return

    with target_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(new_rows)


def _resolve_leaderboard_paths(
    results_dir: Path, leaderboard_name: str
) -> tuple[Path, Path]:
    named_path = results_dir / leaderboard_name / "leaderboard.csv"
    direct_path = results_dir / "leaderboard.csv"
    return named_path, direct_path


def _candidate_result_paths(results_dir: Path, run_name: str, filename: str) -> list[Path]:
    # AlpacaEval has historically written results in either:
    # - output_path/<name>/<filename>
    # - output_path/<filename>
    # - output_path/results/<name>/<filename>
    # - output_path/results/<filename>
    return [
        results_dir / run_name / filename,
        results_dir / filename,
        results_dir / "results" / run_name / filename,
        results_dir / "results" / filename,
    ]


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_model_rows(rows: list[dict[str, str]], model_name: str) -> list[dict[str, str]]:
    if not rows:
        return []

    def _row_label(row: dict[str, str]) -> str:
        for key in ("name", "model", "generator", "model_name", ""):
            value = row.get(key)
            if value:
                return str(value)
        for value in row.values():
            if value:
                return str(value)
        return ""

    exact = [row for row in rows if _row_label(row) == model_name]
    if exact:
        return exact
    return [row for row in rows if _matches_model(model_name, _row_label(row))]


def _find_latest_csv(
    results_dir: Path, filename: str, model_name: str | None = None
) -> Path | None:
    candidates = [path for path in results_dir.rglob(filename) if path.is_file()]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if model_name is None:
        return candidates[0]

    for path in candidates:
        try:
            rows, _ = _read_csv_rows(path)
        except Exception:
            continue
        if _extract_model_rows(rows, model_name):
            return path
    return candidates[0]


def _find_latest_json(results_dir: Path, filename: str) -> Path | None:
    candidates = [path for path in results_dir.rglob(filename) if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _write_leaderboard_history(
    source_path: Path,
    history_path: Path,
    run_name: str,
    evaluated_at: str,
) -> None:
    if not source_path.exists():
        return

    rows, fieldnames = _read_csv_rows(source_path)
    rows = _extract_model_rows(rows, run_name)
    if not rows:
        return

    for row in rows:
        row.setdefault("run_id", evaluated_at)
        row.setdefault("evaluated_at", evaluated_at)

    existing_rows: list[dict[str, str]] = []
    existing_fieldnames: list[str] = []
    if history_path.exists():
        existing_rows, existing_fieldnames = _read_csv_rows(history_path)

    fieldnames_out = list(existing_fieldnames)
    for name in fieldnames:
        if name not in fieldnames_out:
            fieldnames_out.append(name)
    for name in ("run_id", "evaluated_at"):
        if name not in fieldnames_out:
            fieldnames_out.append(name)
    if not fieldnames_out:
        fieldnames_out = list(fieldnames)

    merged_rows = existing_rows + rows
    history_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(history_path, merged_rows, fieldnames_out)


def _select_reference_file(repo_id: str, alias: str) -> str:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates = [
        file
        for file in files
        if file.lower().endswith((".json", ".jsonl", ".parquet", ".csv", ".tsv"))
    ]
    if not candidates:
        raise RuntimeError("No reference output files found in the dataset repo.")

    tokens = _tokenize(alias)
    if tokens:
        matched = [
            file
            for file in candidates
            if all(token in file.lower() for token in tokens)
        ]
        if matched:
            return sorted(matched)[0]

    for file in candidates:
        lowered = file.lower()
        if "gpt4" in lowered and "baseline" in lowered:
            return file

    return sorted(candidates)[0]


def _resolve_reference_outputs(reference_outputs: str, reference_repo: str) -> str:
    path = Path(reference_outputs)
    if path.suffix in {".json", ".jsonl", ".parquet", ".csv", ".tsv"}:
        if path.exists():
            return str(path)
        return hf_hub_download(
            repo_id=reference_repo, repo_type="dataset", filename=path.name
        )

    alias_map = {
        "gpt4_turbo": "alpaca_eval_gpt4_baseline.json",
        "gpt4": "alpaca_eval_gpt4_baseline.json",
        "gpt4_baseline": "alpaca_eval_gpt4_baseline.json",
        "alpaca_eval_gpt4_baseline": "alpaca_eval_gpt4_baseline.json",
    }
    alias_key = reference_outputs.lower().strip()
    if alias_key in alias_map:
        filename = alias_map[alias_key]
    else:
        filename = _select_reference_file(reference_repo, reference_outputs)

    return hf_hub_download(
        repo_id=reference_repo, repo_type="dataset", filename=filename
    )


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
        "--model_outputs",
        type=str,
        default=None,
        help="Override path to model outputs JSON/JSONL (uses output_dir if unset).",
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
        help="Reference outputs name or file path",
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
    parser.add_argument(
        "--reference_repo",
        type=str,
        default="tatsu-lab/alpaca_eval",
        help="HuggingFace dataset repo ID for reference outputs",
    )
    parser.add_argument(
        "--leaderboard_history",
        type=str,
        default=None,
        help=(
            "Optional CSV path to append your model's rows after each run "
            "(defaults to <output_dir>/results/leaderboard_history.csv)."
        ),
    )
    parser.add_argument(
        "--write_history_from_existing",
        action="store_true",
        help=(
            "When used with --skip_eval, scan output_dir for an existing leaderboard.csv "
            "and append matching rows to the history CSV."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    outputs_dir = output_dir / "outputs"
    results_dir = output_dir / "results"
    outputs_file = (
        Path(args.model_outputs) if args.model_outputs else outputs_dir / "model_outputs.json"
    )
    outputs_file.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    generate_script = script_dir / "generate_outputs.py"
    analyze_script = script_dir / "analyze_results.py"

    leaderboard_name = args.name or args.model_name.split("/")[-1]
    evaluated_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    history_path = (
        Path(args.leaderboard_history)
        if args.leaderboard_history
        else results_dir / "leaderboard_history.csv"
    )

    if not args.skip_generation:
        cmd = [
            sys.executable,
            str(generate_script),
            "--model_name",
            args.model_name,
            "--generator_name",
            leaderboard_name,
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

        from alpaca_eval import main as alpaca_main

        reference_outputs = _resolve_reference_outputs(
            args.reference_outputs, args.reference_repo
        )
        model_outputs_df = _load_and_sanitize_outputs(outputs_file)
        reference_outputs_df = _load_and_sanitize_outputs(reference_outputs)
        weights_dir = outputs_dir / "weights"

        alpaca_main.evaluate(
            model_outputs=model_outputs_df,
            reference_outputs=reference_outputs_df,
            annotators_config=args.annotators_config,
            output_path=str(results_dir),
            name=leaderboard_name,
            max_instances=args.max_instances,
            metric_kwargs={"save_weights_dir": str(weights_dir)},
        )
        candidate_leaderboards = _candidate_result_paths(
            results_dir, leaderboard_name, "leaderboard.csv"
        )
        updated_leaderboard_path = _first_existing_path(candidate_leaderboards)
        if updated_leaderboard_path is None:
            updated_leaderboard_path = _find_latest_csv(
                results_dir, "leaderboard.csv", model_name=leaderboard_name
            )
        if updated_leaderboard_path is None:
            print(
                "Warning: AlpacaEval did not create a leaderboard.csv in any expected location:"
            )
            for candidate in candidate_leaderboards:
                print("-", candidate)
        else:
            _write_leaderboard_history(
                updated_leaderboard_path,
                history_path,
                leaderboard_name,
                evaluated_at=evaluated_at,
            )
    elif args.write_history_from_existing:
        existing_leaderboard = _find_latest_csv(
            results_dir, "leaderboard.csv", model_name=leaderboard_name
        )
        if existing_leaderboard is None:
            print(
                "No leaderboard.csv found under output_dir; cannot write history "
                f"for name={leaderboard_name}."
            )
        else:
            _write_leaderboard_history(
                existing_leaderboard,
                history_path,
                leaderboard_name,
                evaluated_at=evaluated_at,
            )

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
