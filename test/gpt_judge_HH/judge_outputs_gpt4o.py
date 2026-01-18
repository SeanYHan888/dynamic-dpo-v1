"""
Pairwise judge model outputs against each other with GPT-4o.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError as exc:
    raise RuntimeError("openai package is required to run this script.") from exc


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_outputs(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of outputs in {path}")
    outputs = {}
    for row in data:
        instruction = row.get("instruction")
        output = row.get("output")
        if not instruction or output is None:
            continue
        if instruction not in outputs:
            outputs[instruction] = output
    return outputs


def _intersection_keys(*maps: dict[str, str]) -> list[str]:
    if not maps:
        return []
    keys = set(maps[0].keys())
    for m in maps[1:]:
        keys &= set(m.keys())
    return sorted(keys)


def _build_prompt(
    template: str, instruction: str, labeled_outputs: dict[str, str]
) -> str:
    return template.format(
        instruction=instruction,
        output_a=labeled_outputs.get("A", ""),
        output_b=labeled_outputs.get("B", ""),
        output_c=labeled_outputs.get("C", ""),
    )


def _normalize_winner(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().strip('"').strip().upper()
    if cleaned in {"A", "B", "TIE"}:
        return cleaned
    return None


def _parse_response(text: str) -> tuple[str | None, str | None]:
    text = text.strip()
    if not text:
        return None, None

    comparison: str | None = None
    winner: str | None = None

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            raw_comparison = payload.get("comparison") or payload.get("Comparison")
            if isinstance(raw_comparison, str):
                comparison = raw_comparison.strip()
            raw_winner = payload.get("winner") or payload.get("Winner")
            if isinstance(raw_winner, str):
                winner = _normalize_winner(raw_winner)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        lower = line.lower()
        if lower.startswith("comparison:"):
            comparison = line.split(":", 1)[1].strip()
        elif lower.startswith("winner:"):
            winner = _normalize_winner(line.split(":", 1)[1])

    if winner is None:
        match = re.search(r'"winner"\s*:\s*"(A|B|TIE)"', text, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()

    if winner is None:
        match = re.search(r"\b(A|B|TIE)\b", text, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()

    return comparison, winner


def _init_counts(model_keys: list[str]) -> dict[str, dict[str, int]]:
    return {
        key: {"wins": 0, "losses": 0, "ties": 0} for key in model_keys
    }


def _record_count(
    counts: dict[str, dict[str, int]],
    pair_key: str,
    winner_key: str | None,
    left_key: str,
    right_key: str,
) -> None:
    stats = counts.setdefault(pair_key, {"wins": 0, "losses": 0, "ties": 0})
    if winner_key == left_key:
        stats["wins"] += 1
    elif winner_key == right_key:
        stats["losses"] += 1
    else:
        stats["ties"] += 1


def _summarize(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model_key, stats in counts.items():
        total = stats["wins"] + stats["losses"] + stats["ties"]
        win_rate = stats["wins"] / total if total else 0.0
        summary[model_key] = {
            "total": total,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "ties": stats["ties"],
            "win_rate": win_rate,
        }
    return summary


def _iter_json_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    objects: list[dict[str, Any]] = []
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, next_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            objects.append(obj)
        idx = next_idx
    return objects


def _clear_results_dir(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _seed_for_pair(
    seed: int | None, instruction: str, pair_key: str
) -> int | None:
    if seed is None:
        return None
    digest = hashlib.sha256(
        f"{seed}-{pair_key}-{instruction}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _shuffle_pair_outputs(
    outputs: list[tuple[str, str]], rng: random.Random | None
) -> None:
    if rng is None:
        random.SystemRandom().shuffle(outputs)
        return
    rng.shuffle(outputs)


def _call_gpt4_oracle(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    system_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    attempt = 0
    backoff = initial_backoff
    last_error: Exception | None = None

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = (
                    response.usage.model_dump()
                    if hasattr(response.usage, "model_dump")
                    else dict(response.usage)
                )
            return text, usage
        except Exception as exc:  # Broad to handle API/network errors.
            last_error = exc
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    raise RuntimeError(f"OpenAI API failed after {max_retries} retries") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use GPT-4o to judge pairwise model outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge_HH/config_evaluation.yaml",
        help="Path to evaluation config YAML.",
    )
    parser.add_argument(
        "--sft",
        type=str,
        default=None,
        help="Override path to SFT outputs JSON.",
    )
    parser.add_argument(
        "--og_dpo",
        type=str,
        default=None,
        help="Override path to original DPO outputs JSON.",
    )
    parser.add_argument(
        "--dpo",
        type=str,
        default=None,
        help="Override path to DPO outputs JSON.",
    )
    parser.add_argument(
        "--beta_dpo",
        type=str,
        default=None,
        help="Override path to beta DPO outputs JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override GPT-4o model name for judging.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="all",
        choices=("all", "dpo_vs_sft", "og_dpo_vs_sft", "beta_dpo_vs_sft"),
        help="Run only a single pairwise comparison or all.",
    )
    parser.add_argument(
        "--max_instances",
        "--max_examples",
        dest="max_examples",
        type=int,
        default=None,
        help="Limit the number of instructions to judge.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed for output order randomization.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Override base path to write per-example judgments (pair key suffix is added).",
    )
    parser.add_argument(
        "--results_file_dpo_vs_sft",
        type=str,
        default=None,
        help="Override path to write dpo_vs_sft judgments JSONL.",
    )
    parser.add_argument(
        "--results_file_og_dpo_vs_sft",
        type=str,
        default=None,
        help="Override path to write og_dpo_vs_sft judgments JSONL.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Override path to write the summary JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instructions already present in the results file.",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    config = _load_config(args.config)
    oracle_cfg = config.get("gpt4_oracle", {})
    inputs_cfg = config.get("inputs", {})
    output_cfg = config.get("output", {})

    prompt_template = oracle_cfg.get("prompt_template")
    if not prompt_template:
        raise ValueError("gpt4_oracle.prompt_template must be set in config.")

    sft_path = args.sft or inputs_cfg.get("sft")
    og_dpo_path = args.og_dpo or inputs_cfg.get("og_dpo")
    dpo_path = args.dpo or inputs_cfg.get("dpo")
    if not sft_path or not og_dpo_path or not dpo_path:
        raise ValueError("inputs.sft, inputs.og_dpo, and inputs.dpo must be set.")
    beta_dpo_path = args.beta_dpo or inputs_cfg.get("beta_dpo")
    if args.pair == "beta_dpo_vs_sft" and not beta_dpo_path:
        raise ValueError("inputs.beta_dpo must be set for beta_dpo_vs_sft.")

    results_base_path = Path(
        args.results_file
        or output_cfg.get(
            "results_file", "test/gpt_judge_HH/results/gpt4o_judgments.jsonl"
        )
    )
    summary_path = Path(
        args.summary_file
        or output_cfg.get("summary_file", "test/gpt_judge_HH/results/summary.json")
    )

    model_name = args.model or oracle_cfg.get("model", "gpt-4o-2024-08-06")
    temperature = oracle_cfg.get("temperature", 0.0)
    max_tokens = oracle_cfg.get("max_tokens", 256)
    seed = args.seed if args.seed is not None else oracle_cfg.get("seed", 42)
    max_retries = oracle_cfg.get("max_retries", 5)
    initial_backoff = oracle_cfg.get("initial_backoff", 1.0)
    max_backoff = oracle_cfg.get("max_backoff", 60.0)
    system_prompt = oracle_cfg.get("system_prompt")

    max_examples = (
        args.max_examples
        if args.max_examples is not None
        else oracle_cfg.get("max_examples")
    )

    sft_map = _load_outputs(Path(sft_path))
    og_dpo_map = _load_outputs(Path(og_dpo_path))
    dpo_map = _load_outputs(Path(dpo_path))
    beta_dpo_map = (
        _load_outputs(Path(beta_dpo_path)) if beta_dpo_path else None
    )

    instructions = _intersection_keys(sft_map, og_dpo_map, dpo_map)
    shuffle_instructions = oracle_cfg.get("shuffle_instructions", False)
    if shuffle_instructions:
        rng = random.Random(seed) if seed is not None else random.Random()
        rng.shuffle(instructions)
    if max_examples is not None:
        instructions = instructions[:max_examples]

    seen = set()
    model_maps = {"sft": sft_map, "og_dpo": og_dpo_map, "dpo": dpo_map}
    if beta_dpo_map is not None:
        model_maps["beta_dpo"] = beta_dpo_map
    pairings = [("dpo", "sft"), ("og_dpo", "sft")]
    if beta_dpo_map is not None or args.pair == "beta_dpo_vs_sft":
        pairings.append(("beta_dpo", "sft"))
    pair_map = {f"{left}_vs_{right}": (left, right) for left, right in pairings}
    pair_keys = list(pair_map.keys())
    if args.pair != "all":
        pair_keys = [args.pair]
    counts = _init_counts(pair_keys)
    results_files_cfg = output_cfg.get("results_files")
    results_paths: dict[str, Path] = {}
    for pair_key in pair_keys:
        cli_override: str | None = None
        if pair_key == "dpo_vs_sft":
            cli_override = args.results_file_dpo_vs_sft
        elif pair_key == "og_dpo_vs_sft":
            cli_override = args.results_file_og_dpo_vs_sft

        cfg_override: str | None = None
        if isinstance(results_files_cfg, dict):
            cfg_value = results_files_cfg.get(pair_key)
            if isinstance(cfg_value, str) and cfg_value:
                cfg_override = cfg_value

        resolved = cli_override or cfg_override
        results_paths[pair_key] = (
            Path(resolved)
            if resolved
            else results_base_path.with_name(
                f"{results_base_path.stem}_{pair_key}{results_base_path.suffix}"
            )
        )

    if args.resume:
        for pair_key, pair_results_path in results_paths.items():
            if not pair_results_path.exists():
                continue
            for row in _iter_json_objects(pair_results_path):
                instruction = row.get("instruction")
                if not instruction:
                    continue
                row_pair_key = row.get("model_key")
                if row_pair_key and row_pair_key != pair_key:
                    continue
                seen_key = (instruction, pair_key)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                winner_key = row.get("winner_key") or row.get("winner")
                left_key, right_key = pair_map[pair_key]
                _record_count(counts, pair_key, winner_key, left_key, right_key)

    client = OpenAI()

    total = sum(
        stats["wins"] + stats["losses"] + stats["ties"]
        for stats in counts.values()
    )
    output_mode = "a" if args.resume else "w"

    for pair_key in pair_keys:
        pair_results_path = results_paths[pair_key]
        pair_results_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_log_path = (
            pair_results_path.parent / f"gpt4o_prompts_log_{pair_key}.jsonl"
        )

        left_key, right_key = pair_map[pair_key]
        pending_instructions = [
            instruction
            for instruction in instructions
            if (instruction, pair_key) not in seen
            and instruction in model_maps[left_key]
            and instruction in model_maps[right_key]
        ]

        with (
            pair_results_path.open(output_mode, encoding="utf-8") as out_f,
            prompt_log_path.open(output_mode, encoding="utf-8") as prompt_f,
        ):
            for instruction in tqdm(
                pending_instructions,
                desc=f"Judging {pair_key}",
                unit="examples",
            ):
                outputs = [
                    (left_key, model_maps[left_key][instruction]),
                    (right_key, model_maps[right_key][instruction]),
                ]
                pair_seed = _seed_for_pair(seed, instruction, pair_key)
                rng = random.Random(pair_seed) if pair_seed is not None else None
                _shuffle_pair_outputs(outputs, rng=rng)
                label_map = {"A": outputs[0][0], "B": outputs[1][0]}
                labeled_outputs = {
                    "A": outputs[0][1],
                    "B": outputs[1][1],
                }

                prompt = _build_prompt(prompt_template, instruction, labeled_outputs)
                prompt_f.write(
                    json.dumps(
                        {
                            "instruction": instruction,
                            "model_key": pair_key,
                            "labels": label_map,
                            "prompt": prompt,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n\n"
                )
                prompt_f.flush()
                content, usage = _call_gpt4_oracle(
                    client=client,
                    prompt=prompt,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    system_prompt=system_prompt,
                )
                comparison, winner = _parse_response(content)
                if winner is None:
                    winner = "TIE"

                winner_key = label_map.get(winner)
                if winner_key is None:
                    winner = "TIE"
                    winner_key = "TIE"
                _record_count(counts, pair_key, winner_key, left_key, right_key)
                total += 1
                seen.add((instruction, pair_key))

                out_f.write(
                    json.dumps(
                        {
                            "instruction": instruction,
                            "model_key": pair_key,
                            "comparison": comparison,
                            "winner": winner,
                            "winner_key": winner_key,
                            "labels": label_map,
                            "model": model_name,
                            "raw_response": content,
                            "usage": usage,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n\n"
                )
                out_f.flush()

    summary = _summarize(counts)
    summary["judge_model"] = model_name
    summary["results_files"] = {key: str(path) for key, path in results_paths.items()}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
        f.write("\n\n")

    print(f"Judged {total} comparisons.")
    print(summary)


if __name__ == "__main__":
    main()
