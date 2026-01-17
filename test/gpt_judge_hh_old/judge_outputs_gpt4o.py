"""
Directly judge three output files with GPT-4o (no AlpacaEval).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from tqdm import tqdm

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


def _build_prompt(instruction: str, labeled_outputs: dict[str, str]) -> str:
    return (
        "You are a careful judge. Compare the three outputs for the same instruction.\n"
        "Pick the best output overall. If they are equivalent, reply TIE.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Output (A):\n{labeled_outputs['A']}\n\n"
        f"Output (B):\n{labeled_outputs['B']}\n\n"
        f"Output (C):\n{labeled_outputs['C']}\n\n"
        "Respond with a JSON object ONLY in this exact format:\n"
        '{"winner": "A"}\n'
        'Allowed values: "A", "B", "C", "TIE".'
    )


def _extract_winner(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            winner = payload.get("winner")
            if isinstance(winner, str):
                winner = winner.strip().upper()
                if winner in {"A", "B", "C", "TIE"}:
                    return winner
    match = re.search(r'"winner"\s*:\s*"(A|B|C|TIE)"', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(A|B|C|TIE)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _summarize(counts: dict[str, int], total: int) -> str:
    if total == 0:
        return "No judgments recorded."
    lines = []
    for key in ("sft", "og_dpo", "dpo", "TIE"):
        value = counts.get(key, 0)
        rate = value / total
        lines.append(f"{key}: {value} ({rate:.2%})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use GPT-4o to directly judge three output files."
    )
    parser.add_argument(
        "--sft",
        type=str,
        default="test/alpacaeval/outputs/sft_output_hh.json",
        help="Path to SFT outputs JSON.",
    )
    parser.add_argument(
        "--og_dpo",
        type=str,
        default="test/alpacaeval/outputs/og_dpo_output_hh.json",
        help="Path to original DPO outputs JSON.",
    )
    parser.add_argument(
        "--dpo",
        type=str,
        default="test/alpacaeval/outputs/dpo_output_hh.json",
        help="Path to DPO outputs JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-08-06",
        help="GPT-4o model name for judging.",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Limit the number of instructions to judge.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for output order randomization.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="test/gpt_judge_hh_chosen/gpt4o_judgments.jsonl",
        help="Path to write per-example judgments.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instructions already present in the results file.",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required to run this script.") from exc

    sft_map = _load_outputs(Path(args.sft))
    og_dpo_map = _load_outputs(Path(args.og_dpo))
    dpo_map = _load_outputs(Path(args.dpo))

    instructions = _intersection_keys(sft_map, og_dpo_map, dpo_map)
    if args.max_instances is not None:
        instructions = instructions[: args.max_instances]

    seen = set()
    results_path = Path(args.results_file)
    if args.resume and results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = row.get("instruction")
                if instruction:
                    seen.add(instruction)

    client = OpenAI()
    rng = random.Random(args.seed)

    counts = {"sft": 0, "og_dpo": 0, "dpo": 0, "TIE": 0}
    total = 0

    pending_instructions = [inst for inst in instructions if inst not in seen]

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as out_f:
        for instruction in tqdm(pending_instructions, desc="Judging", unit="examples"):
            if instruction in seen:
                continue
            outputs = [
                ("sft", sft_map[instruction]),
                ("og_dpo", og_dpo_map[instruction]),
                ("dpo", dpo_map[instruction]),
            ]
            rng.shuffle(outputs)
            label_map = {"A": outputs[0][0], "B": outputs[1][0], "C": outputs[2][0]}
            labeled_outputs = {
                "A": outputs[0][1],
                "B": outputs[1][1],
                "C": outputs[2][1],
            }
            prompt = _build_prompt(instruction, labeled_outputs)

            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a strict evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            winner = _extract_winner(content)
            if winner is None:
                winner = "TIE"

            winner_key = label_map.get(winner, winner)
            counts[winner_key] = counts.get(winner_key, 0) + 1
            total += 1

            out_f.write(
                json.dumps(
                    {
                        "instruction": instruction,
                        "winner": winner,
                        "winner_key": winner_key,
                        "labels": label_map,
                        "model": args.model,
                        "raw_response": content,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Judged {total} instructions.")
    print(_summarize(counts, total))


if __name__ == "__main__":
    main()
