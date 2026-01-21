from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

try:
    from .rm_scorer import (
        RMConfig,
        RewardModelScorer,
        normalize_model_output,
        sha256_text,
    )
except ImportError:  # Allows running as a script without `-m`.
    from rm_scorer import RMConfig, RewardModelScorer, normalize_model_output, sha256_text


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class OutputRow:
    instruction: str
    output: str
    generator: str | None


def load_outputs(path: str | Path) -> tuple[dict[str, OutputRow], list[str]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {file_path}")

    by_instruction: dict[str, OutputRow] = {}
    order: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction")
        output = item.get("output")
        if not instruction or output is None:
            continue
        if instruction in by_instruction:
            continue
        generator = item.get("generator")
        by_instruction[instruction] = OutputRow(
            instruction=str(instruction),
            output=str(output),
            generator=str(generator) if generator is not None else None,
        )
        order.append(instruction)
    return by_instruction, order


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def load_score_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        instruction = row.get("instruction")
        if isinstance(instruction, str) and instruction:
            cache[instruction] = row
    return cache


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_pair_stats(
    path: Path,
    *,
    allowed_instructions: set[str] | None = None,
) -> tuple[set[str], int, int, int, list[float], list[float], list[float]]:
    done: set[str] = set()
    wins_left = 0
    wins_right = 0
    ties = 0
    deltas: list[float] = []
    scores_left: list[float] = []
    scores_right: list[float] = []

    for row in iter_jsonl(path):
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            continue
        if allowed_instructions is not None and instruction not in allowed_instructions:
            continue

        done.add(instruction)

        winner = row.get("winner")
        if winner == "A":
            wins_left += 1
        elif winner == "B":
            wins_right += 1
        elif winner == "TIE":
            ties += 1

        scores = row.get("scores")
        if isinstance(scores, dict):
            score_a = scores.get("A")
            score_b = scores.get("B")
            if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)):
                score_a_f = float(score_a)
                score_b_f = float(score_b)
                scores_left.append(score_a_f)
                scores_right.append(score_b_f)
                deltas.append(score_a_f - score_b_f)

    return done, wins_left, wins_right, ties, deltas, scores_left, scores_right


def should_tie(
    score_a: float,
    score_b: float,
    *,
    tie_if_max_below: float | None,
    tie_margin: float | None,
) -> bool:
    if tie_margin is not None and abs(score_a - score_b) < float(tie_margin):
        return True
    if tie_if_max_below is not None and max(score_a, score_b) < float(tie_if_max_below):
        return True
    return False


def resolve_pairs(config: dict[str, Any]) -> list[dict[str, str]]:
    raw = config.get("pairs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.pairs must be a non-empty list.")
    pairs: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        left = item.get("left")
        right = item.get("right")
        if not key or not left or not right:
            raise ValueError(f"Invalid pair entry: {item}")
        pairs.append({"key": str(key), "left": str(left), "right": str(right)})
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge model outputs with ArmoRM.")
    parser.add_argument(
        "--config",
        type=str,
        default="test/judge_RM/config/config_evaluation_rm.yaml",
        help="Path to RM evaluation config YAML.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="all",
        help="Run only one pair key from config.pairs, or 'all'.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of examples per pair.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not skip already-judged instructions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    rm_cfg = config.get("rm_judge", {})
    if not isinstance(rm_cfg, dict):
        raise ValueError("config.rm_judge must be a dict.")

    inputs_cfg = config.get("inputs", {})
    if not isinstance(inputs_cfg, dict) or not inputs_cfg:
        raise ValueError("config.inputs must be a non-empty dict.")

    output_cfg = config.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a dict.")

    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}

    pairs = resolve_pairs(config)
    if args.pair != "all":
        pairs = [pair for pair in pairs if pair["key"] == args.pair]
        if not pairs:
            raise ValueError(f"Pair key '{args.pair}' not found in config.pairs.")

    max_examples = (
        args.max_examples
        if args.max_examples is not None
        else run_cfg.get("max_examples")
    )
    resume = bool(run_cfg.get("resume", True)) and not args.no_resume

    results_dir = Path(output_cfg.get("results_dir", "test/judge_RM/results"))
    scores_dir = Path(output_cfg.get("scores_dir", "test/judge_RM/scores"))
    summary_file = Path(output_cfg.get("summary_file", results_dir / "summary.json"))
    results_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load all required model outputs.
    model_keys: set[str] = set()
    for pair in pairs:
        model_keys.add(pair["left"])
        model_keys.add(pair["right"])

    outputs: dict[str, dict[str, OutputRow]] = {}
    orders: dict[str, list[str]] = {}
    for key in sorted(model_keys):
        path = inputs_cfg.get(key)
        if not isinstance(path, str) or not path:
            raise ValueError(f"Missing inputs.{key} in config.")
        outputs[key], orders[key] = load_outputs(path)

    # Prepare per-pair instruction lists (intersection).
    pair_instructions: dict[str, list[str]] = {}
    instructions_needed: dict[str, set[str]] = {key: set() for key in model_keys}
    for pair in pairs:
        left_key = pair["left"]
        right_key = pair["right"]
        ordered = [
            instruction
            for instruction in orders[left_key]
            if instruction in outputs[right_key]
        ]
        if max_examples is not None:
            ordered = ordered[: int(max_examples)]
        pair_instructions[pair["key"]] = ordered
        instructions_needed[left_key].update(ordered)
        instructions_needed[right_key].update(ordered)

    rm_config = RMConfig(
        reward_model=str(rm_cfg.get("reward_model", "RLHFlow/ArmoRM-Llama3-8B-v0.1")),
        tokenizer_name=rm_cfg.get("tokenizer_name"),
        precision=rm_cfg.get("precision"),
        device_map=rm_cfg.get("device_map", "auto"),
        load_in_8bit=bool(rm_cfg.get("load_in_8bit", False)),
        batch_size=int(rm_cfg.get("batch_size", 8)),
        max_length=rm_cfg.get("max_length"),
    )
    scorer = RewardModelScorer(rm_config)

    tie_if_max_below = rm_cfg.get("tie_if_max_below", 2.0)
    tie_margin = rm_cfg.get("tie_margin")
    tie_if_max_below_f = float(tie_if_max_below) if tie_if_max_below is not None else None
    tie_margin_f = float(tie_margin) if tie_margin is not None else None

    # Score caches: one file per model key.
    scores_files: dict[str, Path] = {key: scores_dir / f"scores_{key}.jsonl" for key in model_keys}
    score_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for key, path in scores_files.items():
        score_cache[key] = load_score_cache(path)

    # Score any missing or stale entries.
    for model_key in sorted(model_keys):
        needed = list(instructions_needed[model_key])
        if not needed:
            continue

        to_score_instructions: list[str] = []
        to_score_responses: list[str] = []
        to_score_hashes: list[str] = []
        for instruction in needed:
            row = outputs[model_key].get(instruction)
            if row is None:
                continue
            cleaned = normalize_model_output(row.output)
            out_hash = sha256_text(cleaned)
            cached = score_cache[model_key].get(instruction)
            if (
                cached
                and cached.get("output_sha256") == out_hash
                and isinstance(cached.get("score"), (int, float))
            ):
                continue
            to_score_instructions.append(instruction)
            to_score_responses.append(cleaned)
            to_score_hashes.append(out_hash)

        if not to_score_instructions:
            continue

        new_rows: list[dict[str, Any]] = []
        for start in tqdm(
            range(0, len(to_score_instructions), rm_config.batch_size),
            desc=f"Scoring {model_key}",
            unit="batch",
        ):
            batch_instructions = to_score_instructions[start : start + rm_config.batch_size]
            batch_responses = to_score_responses[start : start + rm_config.batch_size]
            batch_hashes = to_score_hashes[start : start + rm_config.batch_size]
            batch_scores = scorer.score_instruction_responses(batch_instructions, batch_responses)
            for ins, score, out_hash in zip(batch_instructions, batch_scores, batch_hashes):
                generator = outputs[model_key].get(ins).generator if outputs[model_key].get(ins) else None
                new_rows.append(
                    {
                        "instruction": ins,
                        "output_sha256": out_hash,
                        "score": float(score),
                        "generator": generator,
                    }
                )

        append_jsonl(scores_files[model_key], new_rows)
        for row in new_rows:
            score_cache[model_key][row["instruction"]] = row

    results_files: dict[str, str] = {}
    summary_pairs: dict[str, Any] = {}

    for pair in pairs:
        pair_key = pair["key"]
        left_key = pair["left"]
        right_key = pair["right"]
        instructions = pair_instructions.get(pair_key, [])

        results_path = results_dir / f"rm_judgments_{pair_key}.jsonl"
        results_files[pair_key] = str(results_path)
        if resume:
            allowed_instructions = set(instructions)
            (
                already_done,
                left_wins,
                right_wins,
                ties,
                deltas,
                score_left_all,
                score_right_all,
            ) = load_existing_pair_stats(
                results_path, allowed_instructions=allowed_instructions
            )
        else:
            already_done = set()
            left_wins = 0
            right_wins = 0
            ties = 0
            deltas = []
            score_left_all = []
            score_right_all = []

        out_rows: list[dict[str, Any]] = []
        for instruction in tqdm(instructions, desc=f"Judging {pair_key}", unit="ex"):
            if instruction in already_done:
                continue

            left_row = outputs[left_key].get(instruction)
            right_row = outputs[right_key].get(instruction)
            if left_row is None or right_row is None:
                continue

            left_clean = normalize_model_output(left_row.output)
            right_clean = normalize_model_output(right_row.output)
            left_hash = sha256_text(left_clean)
            right_hash = sha256_text(right_clean)

            left_cached = score_cache[left_key].get(instruction)
            right_cached = score_cache[right_key].get(instruction)

            if not left_cached or left_cached.get("output_sha256") != left_hash:
                left_score = float(scorer.score_instruction_responses([instruction], [left_clean])[0])
                left_cached = {
                    "instruction": instruction,
                    "output_sha256": left_hash,
                    "score": left_score,
                    "generator": left_row.generator,
                }
                append_jsonl(scores_files[left_key], [left_cached])
                score_cache[left_key][instruction] = left_cached
            else:
                left_score = float(left_cached["score"])

            if not right_cached or right_cached.get("output_sha256") != right_hash:
                right_score = float(scorer.score_instruction_responses([instruction], [right_clean])[0])
                right_cached = {
                    "instruction": instruction,
                    "output_sha256": right_hash,
                    "score": right_score,
                    "generator": right_row.generator,
                }
                append_jsonl(scores_files[right_key], [right_cached])
                score_cache[right_key][instruction] = right_cached
            else:
                right_score = float(right_cached["score"])

            delta = left_score - right_score
            deltas.append(delta)
            score_left_all.append(left_score)
            score_right_all.append(right_score)

            winner: str
            winner_key: str
            if should_tie(
                left_score,
                right_score,
                tie_if_max_below=tie_if_max_below_f,
                tie_margin=tie_margin_f,
            ) or left_score == right_score:
                winner = "TIE"
                winner_key = "TIE"
                ties += 1
            elif left_score > right_score:
                winner = "A"
                winner_key = left_key
                left_wins += 1
            else:
                winner = "B"
                winner_key = right_key
                right_wins += 1

            out_rows.append(
                {
                    "instruction": instruction,
                    "pair_key": pair_key,
                    "labels": {"A": left_key, "B": right_key},
                    "scores": {"A": left_score, "B": right_score},
                    "winner": winner,
                    "winner_key": winner_key,
                    "output_sha256": {"A": left_hash, "B": right_hash},
                }
            )

            if len(out_rows) >= 100:
                append_jsonl(results_path, out_rows)
                out_rows = []

        if out_rows:
            append_jsonl(results_path, out_rows)

        total = left_wins + right_wins + ties
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        mean_left = sum(score_left_all) / len(score_left_all) if score_left_all else 0.0
        mean_right = sum(score_right_all) / len(score_right_all) if score_right_all else 0.0
        summary_pairs[pair_key] = {
            "left": left_key,
            "right": right_key,
            "total": total,
            "wins_left": left_wins,
            "wins_right": right_wins,
            "ties": ties,
            "win_rate_left": (left_wins / total) if total else 0.0,
            "win_rate_right": (right_wins / total) if total else 0.0,
            "tie_rate": (ties / total) if total else 0.0,
            "mean_score_left": mean_left,
            "mean_score_right": mean_right,
            "mean_delta": mean_delta,
        }

    summary = {
        "judge_type": "armorm",
        "reward_model": rm_config.reward_model,
        "tokenizer_name": rm_config.tokenizer_name,
        "precision": rm_config.precision,
        "device_map": rm_config.device_map,
        "load_in_8bit": rm_config.load_in_8bit,
        "batch_size": rm_config.batch_size,
        "max_length": rm_config.max_length,
        "tie_if_max_below": tie_if_max_below_f,
        "tie_margin": tie_margin_f,
        "pairs": summary_pairs,
        "results_files": results_files,
        "scores_files": {k: str(v) for k, v in scores_files.items()},
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote summary to {summary_file}")


if __name__ == "__main__":
    main()
