from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

from .hh_parser import extract_prompt_and_reference, messages_have_raw_role_tags
from .rollout import RMJudge, RolloutGenerator
from .utils import load_model, load_tokenizer, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_dpo.yaml")
    for name, arg_type in (
        ("output_dir", str),
        ("limit", int),
        ("batch_size", int),
        ("k", int),
        ("responses_per_prompt", int),
        ("temperature", float),
        ("top_p", float),
        ("max_new_tokens", int),
        ("min_new_tokens", int),
        ("seed", int),
        ("device_map", str),
        ("judge", str),
        ("reward_model", str),
        ("reward_batch_size", int),
        ("reward_precision", str),
        ("reward_device_map", str),
        ("reward_max_length", int),
        ("reward_quantization", str),
    ):
        parser.add_argument(f"--{name}", type=arg_type, default=None)
    parser.add_argument("--reward_load_in_8bit", action="store_true", default=None)
    return parser.parse_args()


def resolve_rollout_cfg(config: Dict, args: argparse.Namespace) -> Dict:
    rollout_cfg = config.get("rollout", {})
    dataset_cfg = config.get("dataset", {})

    def pick(key: str, default):
        value = getattr(args, key)
        return rollout_cfg.get(key, default) if value is None else value

    return {
        "dataset_name": dataset_cfg.get("dataset_name", "Anthropic/hh-rlhf"),
        "subset": dataset_cfg.get("subset", "train"),
        "model_name": rollout_cfg.get("model_name"),
        "seed": args.seed if args.seed is not None else dataset_cfg.get("seed", 42),
        "output_dir": args.output_dir or rollout_cfg.get("output_dir", "rollout_output"),
        "limit": args.limit if args.limit is not None else rollout_cfg.get("limit"),
        "batch_size": pick("batch_size", 4),
        "responses_per_prompt": pick(
            "responses_per_prompt",
            rollout_cfg.get("responses_per_prompt", rollout_cfg.get("k", 8)),
        ),
        "temperature": pick("temperature", 0.7),
        "top_p": pick("top_p", 0.9),
        "max_new_tokens": pick("max_new_tokens", 512),
        "min_new_tokens": pick("min_new_tokens", 10),
        "device_map": args.device_map if args.device_map is not None else rollout_cfg.get("device_map"),
        "judge": pick("judge", "dummy"),
        "reward_model": pick("reward_model", "RLHFlow/ArmoRM-Llama3-8B-v0.1"),
        "reward_batch_size": pick("reward_batch_size", 4),
        "reward_precision": pick("reward_precision", None),
        "reward_device_map": pick("reward_device_map", None),
        "reward_max_length": pick("reward_max_length", None),
        "reward_quantization": pick("reward_quantization", rollout_cfg.get("reward_quantization", "8bit")),
        "reward_load_in_8bit": (
            rollout_cfg.get("reward_load_in_8bit", False)
            if args.reward_load_in_8bit is None
            else args.reward_load_in_8bit
        ),
    }


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rollout_cfg = resolve_rollout_cfg(config, args)

    model_name = rollout_cfg.get("model_name") or config.get("policy_name") or config.get("model_name")
    if not model_name:
        raise ValueError("Missing policy_name in config or --model_name override.")

    seed_everything(int(rollout_cfg["seed"]))
    tokenizer = load_tokenizer(model_name, padding_side="left")
    model = load_model(
        model_name,
        precision=config.get("precision"),
        device_map=rollout_cfg["device_map"],
    )
    model.eval()
    if rollout_cfg["device_map"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    responses_per_prompt = int(rollout_cfg["responses_per_prompt"])
    gen_kwargs = {
        "do_sample": True,
        "temperature": float(rollout_cfg["temperature"]),
        "top_p": float(rollout_cfg["top_p"]),
        "max_new_tokens": int(rollout_cfg["max_new_tokens"]),
        "min_new_tokens": int(rollout_cfg["min_new_tokens"]),
    }
    generator = RolloutGenerator(
        model=model,
        tokenizer=tokenizer,
        num_return_sequences=responses_per_prompt,
        **gen_kwargs,
    )
    judge_name = str(rollout_cfg["judge"]).lower()
    if judge_name not in ("rm", "reward", "pairrm"):
        raise ValueError(f"Unsupported judge '{judge_name}'. Use 'rm'.")
    quantization = str(rollout_cfg["reward_quantization"]).lower()
    load_in_8bit = quantization in ("8bit", "int8", "bnb8")
    if rollout_cfg["reward_load_in_8bit"]:
        load_in_8bit = True

    judge = RMJudge(
        model_name=rollout_cfg["reward_model"],
        precision=rollout_cfg["reward_precision"] or config.get("precision"),
        device_map=rollout_cfg["reward_device_map"],
        load_in_8bit=load_in_8bit,
        batch_size=int(rollout_cfg["reward_batch_size"]),
        max_length=rollout_cfg["reward_max_length"],
        seed=int(rollout_cfg["seed"]),
    )

    output_dir = rollout_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, "rollout_responses.jsonl")
    judged_path = os.path.join(output_dir, "rollout_judged.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")

    meta_base = {
        "source": "hh_rollout",
        "seed": int(rollout_cfg["seed"]),
        "k_candidates": responses_per_prompt,
        "generator_model": model_name,
    }
    meta_base["judge"] = "rm" if judge_name in ("rm", "reward", "pairrm") else judge_name
    if judge_name in ("rm", "reward", "pairrm"):
        meta_base["reward_model"] = rollout_cfg["reward_model"]
        meta_base["reward_quantization"] = "8bit" if load_in_8bit else "none"
    manifest = {
        "dataset_name": rollout_cfg["dataset_name"],
        "subset": rollout_cfg["subset"],
        "generation_kwargs": gen_kwargs,
        "responses_file": responses_path,
        "judged_file": judged_path,
        **meta_base,
    }

    raw_ds = load_dataset(rollout_cfg["dataset_name"], split=rollout_cfg["subset"])
    limit = rollout_cfg["limit"]
    batch_size = int(rollout_cfg["batch_size"])

    processed = 0
    generated = 0
    buffer: List[dict] = []

    def flush(batch: List[dict], responses_f, judged_f) -> bool:
        nonlocal processed, generated
        candidates = generator.generate_batch([b["prompt_text"] for b in batch])
        for item, cand_list in tqdm(
            list(zip(batch, candidates)),
            desc="RM judging",
            leave=False,
        ):
            responses = [c.strip() for c in cand_list]
            responses_f.write(
                json.dumps(
                    {
                        "prompt_messages": item["prompt_messages"],
                        "responses": responses,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            responses_f.flush()
            generated += 1

            nonempty = [(idx, resp) for idx, resp in enumerate(responses) if resp]
            if len(nonempty) < 2:
                continue
            idx_map, cleaned = zip(*nonempty)
            best_local, worst_local = judge.rank(item["prompt_messages"], list(cleaned))
            best_idx = idx_map[best_local]
            worst_idx = idx_map[worst_local]
            record = {
                "prompt_messages": item["prompt_messages"],
                "chosen": [{"role": "assistant", "content": responses[best_idx]}],
                "rejected": [{"role": "assistant", "content": responses[worst_idx]}],
                "metadata": {**meta_base, "reference_response": item["reference_response"]},
            }
            judged_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            judged_f.flush()
            processed += 1
            if limit is not None and processed >= int(limit):
                return True
        return False

    with open(responses_path, "w", encoding="utf-8") as responses_f, open(
        judged_path, "w", encoding="utf-8"
    ) as judged_f:
        for row in tqdm(raw_ds, desc="Rollout prompts"):
            text = row.get("chosen") if isinstance(row, dict) else None
            if not text:
                continue
            prompt_messages, reference_response = extract_prompt_and_reference(text)
            if not prompt_messages or messages_have_raw_role_tags(prompt_messages):
                continue

            buffer.append(
                {
                    "prompt_messages": prompt_messages,
                    "prompt_text": tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    ),
                    "reference_response": reference_response,
                }
            )

            if len(buffer) < batch_size:
                continue
            if flush(buffer, responses_f, judged_f):
                buffer = []
                break
            buffer = []

        if buffer and (limit is None or processed < int(limit)):
            flush(buffer, responses_f, judged_f)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {generated} rows to {responses_path}")
    print(f"Wrote {processed} rows to {judged_path}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
