"""
Pairwise judge model outputs against HH chosen answers with GPT-4o.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError as exc:
    raise RuntimeError("openai package is required to run this script.") from exc

try:
    from datasets import load_dataset
except ImportError as exc:
    raise RuntimeError(
        "datasets package is required to load HH chosen answers."
    ) from exc

TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")


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
    model_key: str,
    winner_key: str | None,
) -> None:
    stats = counts.setdefault(model_key, {"wins": 0, "losses": 0, "ties": 0})
    if winner_key == model_key:
        stats["wins"] += 1
    elif winner_key == "chosen":
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


def _seed_for_pair(
    seed: int | None, instruction: str, model_key: str
) -> int | None:
    if seed is None:
        return None
    digest = hashlib.sha256(
        f"{seed}-{model_key}-{instruction}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


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


def _strip_one_leading_newline(text: str) -> str:
    return text[1:] if text.startswith("\n") else text


def _parse_hh_to_messages(text: str) -> list[dict[str, str]]:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("\n\nHuman:") and not text.startswith("\n\nAssistant:"):
        text = "\n\n" + text

    parts = TAG_RE.split(text)
    messages: list[dict[str, str]] = []
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        content = _strip_one_leading_newline(content).strip()
        if not content:
            continue
        role = "user" if role_tag == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def _extract_single_turn_pair(text: str) -> tuple[str, str] | None:
    messages = _parse_hh_to_messages(text)
    if len(messages) != 2:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    instruction = messages[0]["content"]
    response = messages[1]["content"]
    if not instruction or not response:
        return None
    return instruction, response


def _load_hh_chosen_map(repo_id: str, split: str) -> dict[str, str]:
    dataset = load_dataset(repo_id, split=split)
    chosen_map: dict[str, str] = {}
    for row in dataset:
        text = row.get("chosen") or row.get("prompt") or row.get("text")
        if text is None:
            continue
        pair = _extract_single_turn_pair(text)
        if pair is None:
            continue
        instruction, chosen = pair
        if instruction not in chosen_map:
            chosen_map[instruction] = chosen
    return chosen_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use GPT-4o to judge model outputs vs HH chosen answers."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge/config_evaluation.yaml",
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
        "--model",
        type=str,
        default=None,
        help="Override GPT-4o model name for judging.",
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
        help="Override path to write per-example judgments.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Override path to write the summary JSON.",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID for HH chosen answers.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Dataset split to load chosen answers from.",
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
    generation_cfg = config.get("generation", {})

    prompt_template = oracle_cfg.get("prompt_template")
    if not prompt_template:
        raise ValueError("gpt4_oracle.prompt_template must be set in config.")

    sft_path = args.sft or inputs_cfg.get("sft")
    og_dpo_path = args.og_dpo or inputs_cfg.get("og_dpo")
    dpo_path = args.dpo or inputs_cfg.get("dpo")
    if not sft_path or not og_dpo_path or not dpo_path:
        raise ValueError("inputs.sft, inputs.og_dpo, and inputs.dpo must be set.")

    results_path = Path(
        args.results_file
        or output_cfg.get("results_file", "test/gpt_judge/results/gpt4o_judgments.jsonl")
    )
    summary_path = Path(
        args.summary_file
        or output_cfg.get("summary_file", "test/gpt_judge/results/summary.json")
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
    dataset_repo = args.dataset_repo or generation_cfg.get(
        "dataset_repo", "Anthropic/hh-rlhf"
    )
    dataset_split = args.dataset_split or generation_cfg.get("dataset_split", "test")
    chosen_map = _load_hh_chosen_map(dataset_repo, dataset_split)

    instructions = _intersection_keys(sft_map, og_dpo_map, dpo_map, chosen_map)
    if max_examples is not None:
        instructions = instructions[:max_examples]
    if instructions:
        if seed is not None:
            order_rng = random.Random(seed)
            order_rng.shuffle(instructions)
        else:
            random.shuffle(instructions)

    seen = set()
    model_maps = {"sft": sft_map, "og_dpo": og_dpo_map, "dpo": dpo_map}
    model_keys = list(model_maps.keys())
    counts = _init_counts(model_keys)
    if args.resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = row.get("instruction")
                model_key = row.get("model_key")
                if not instruction or not model_key:
                    continue
                pair_key = (instruction, model_key)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                winner_key = row.get("winner_key") or row.get("winner")
                _record_count(counts, model_key, winner_key)

    client = OpenAI()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(
        stats["wins"] + stats["losses"] + stats["ties"]
        for stats in counts.values()
    )
    pending_pairs = [
        (instruction, model_key)
        for instruction in instructions
        for model_key in model_keys
        if (instruction, model_key) not in seen
    ]

    with results_path.open("a", encoding="utf-8") as out_f:
        for instruction, model_key in tqdm(
            pending_pairs, desc="Judging", unit="comparisons"
        ):
            outputs = [
                (model_key, model_maps[model_key][instruction]),
                ("chosen", chosen_map[instruction]),
            ]
            instruction_seed = _seed_for_pair(seed, instruction, model_key)
            rng = (
                random.Random(instruction_seed)
                if instruction_seed is not None
                else random.Random()
            )
            rng.shuffle(outputs)
            label_map = {"A": outputs[0][0], "B": outputs[1][0]}
            labeled_outputs = {
                "A": outputs[0][1],
                "B": outputs[1][1],
            }

            prompt = _build_prompt(prompt_template, instruction, labeled_outputs)
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
            _record_count(counts, model_key, winner_key)
            total += 1

            out_f.write(
                json.dumps(
                    {
                        "instruction": instruction,
                        "model_key": model_key,
                        "comparison": comparison,
                        "winner": winner,
                        "winner_key": winner_key,
                        "labels": label_map,
                        "model": model_name,
                        "raw_response": content,
                        "usage": usage,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

    summary = _summarize(counts)
    summary["judge_model"] = model_name
    summary["dataset_repo"] = dataset_repo
    summary["dataset_split"] = dataset_split
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"Judged {total} comparisons.")
    print(summary)


if __name__ == "__main__":
    main()
