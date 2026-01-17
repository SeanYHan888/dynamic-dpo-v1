"""
Generate model outputs for the HH dataset (single-turn prompts only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _strip_one_leading_newline(text: str) -> str:
    return text[1:] if text.startswith("\n") else text


def _parse_hh_to_messages(text: str) -> list[dict]:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("\n\nHuman:") and not text.startswith("\n\nAssistant:"):
        text = "\n\n" + text

    parts = TAG_RE.split(text)
    messages = []
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        content = _strip_one_leading_newline(content).strip()
        if not content:
            continue
        role = "user" if role_tag == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def _extract_single_turn_instruction(text: str) -> str | None:
    messages = _parse_hh_to_messages(text)
    if len(messages) != 2:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    return messages[0]["content"]


def load_hh_single_turn_instructions(
    repo_id: str = "Anthropic/hh-rlhf",
    split: str = "test",
) -> List[str]:
    dataset = load_dataset(repo_id, split=split)
    instructions: list[str] = []
    for row in dataset:
        text = row.get("chosen") or row.get("prompt") or row.get("text")
        if text is None:
            continue
        instruction = _extract_single_turn_instruction(text)
        if instruction is None:
            continue
        instructions.append(instruction)
    return instructions


def _format_prompt(
    tokenizer: AutoTokenizer, instruction: str, apply_chat_template: bool
) -> str:
    if (
        apply_chat_template
        and getattr(tokenizer, "apply_chat_template", None)
        and tokenizer.chat_template
    ):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if not apply_chat_template:
        bos_token = tokenizer.bos_token or ""
        return f"{bos_token}{instruction}"
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _resolve_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _lookup_token_id(tokenizer: AutoTokenizer, token: str) -> int | None:
    vocab = tokenizer.get_vocab()
    if token in vocab:
        return vocab[token]
    added_vocab = tokenizer.get_added_vocab()
    if token in added_vocab:
        return added_vocab[token]
    if token in tokenizer.all_special_tokens:
        return tokenizer.convert_tokens_to_ids(token)
    return None


def _resolve_eos_token_id(tokenizer: AutoTokenizer) -> int | list[int] | None:
    eos_token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
    elif tokenizer.eos_token:
        token_id = _lookup_token_id(tokenizer, tokenizer.eos_token)
        if token_id is not None:
            eos_token_ids.append(token_id)

    special_map = tokenizer.special_tokens_map or {}
    eot_tokens: list[str] = []
    if "eot_token" in special_map:
        eot_tokens.append(special_map["eot_token"])
    additional = special_map.get("additional_special_tokens") or []
    for token in additional:
        lowered = token.lower()
        if "eot" in lowered or "end_of_turn" in lowered:
            eot_tokens.append(token)
    for token in tokenizer.all_special_tokens:
        lowered = token.lower()
        if "eot" in lowered or "end_of_turn" in lowered:
            eot_tokens.append(token)

    for token in dict.fromkeys(eot_tokens):
        token_id = _lookup_token_id(tokenizer, token)
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    if not eot_tokens:
        fallback_token = "<|eot_id|>"
        token_id = _lookup_token_id(tokenizer, fallback_token)
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    if not eos_token_ids:
        return None
    if len(eos_token_ids) == 1:
        return eos_token_ids[0]
    return eos_token_ids


def generate_model_outputs(
    model_name: str,
    output_file: str,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_input_tokens: int = 2048,
    max_instances: int | None = None,
    seed: int | None = 42,
    dataset_repo: str = "Anthropic/hh-rlhf",
    dataset_split: str = "test",
    apply_chat_template: bool = True,
) -> None:
    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(resolved_device)

    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = _resolve_eos_token_id(tokenizer)

    model_kwargs = {"torch_dtype": dtype}
    if resolved_device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()

    instructions = load_hh_single_turn_instructions(
        repo_id=dataset_repo, split=dataset_split
    )
    if max_instances is not None:
        instructions = instructions[:max_instances]

    outputs = []
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    do_sample = temperature > 0

    progress = tqdm(total=len(instructions), desc="Generating", unit="examples")
    for batch in _batched(instructions, batch_size):
        prompts = [
            _format_prompt(tokenizer, instruction, apply_chat_template)
            for instruction in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        padded_input_length = (
            inputs["input_ids"].shape[1]
            if tokenizer.padding_side == "left"
            else None
        )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)

        for idx, instruction in enumerate(batch):
            if padded_input_length is not None:
                prompt_length = padded_input_length
            else:
                prompt_length = input_lengths[idx]
            output_ids = generated[idx][prompt_length:]
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            outputs.append(
                {
                    "instruction": instruction,
                    "output": text,
                    "generator": model_name,
                }
            )
        progress.update(len(batch))

    progress.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} outputs to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate outputs for HH single-turn prompts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge_HH/config_evaluation.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the outputs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=None,
        help="Maximum input token length",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Limit the number of HH prompts after filtering",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Dataset split to use",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        default=None,
        help="Whether to apply the tokenizer chat template (true/false).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )

    args = parser.parse_args()
    config = _load_config(args.config)
    generation_cfg = config.get("generation", {})

    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    generate_model_outputs(
        model_name=(
            args.model_name
            or generation_cfg.get("model_name", "W-61/hh-llama32-1b-sft")
        ),
        output_file=(
            args.output_file
            or generation_cfg.get(
                "output_file", "test/gpt_judge_HH/outputs/hh_model_outputs.json"
            )
        ),
        max_new_tokens=(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else generation_cfg.get("max_new_tokens", 512)
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else generation_cfg.get("batch_size", 1)
        ),
        device=(
            args.device
            or generation_cfg.get("device", default_device)
        ),
        temperature=(
            args.temperature
            if args.temperature is not None
            else generation_cfg.get("temperature", 0.7)
        ),
        top_p=(
            args.top_p
            if args.top_p is not None
            else generation_cfg.get("top_p", 0.9)
        ),
        max_input_tokens=(
            args.max_input_tokens
            if args.max_input_tokens is not None
            else generation_cfg.get("max_input_tokens", 2048)
        ),
        max_instances=(
            args.max_instances
            if args.max_instances is not None
            else generation_cfg.get("max_instances")
        ),
        seed=(
            args.seed
            if args.seed is not None
            else generation_cfg.get("seed", 42)
        ),
        dataset_repo=(
            args.dataset_repo
            or generation_cfg.get("dataset_repo", "Anthropic/hh-rlhf")
        ),
        dataset_split=(
            args.dataset_split
            or generation_cfg.get("dataset_split", "test")
        ),
        apply_chat_template=(
            args.apply_chat_template
            if args.apply_chat_template is not None
            else generation_cfg.get("apply_chat_template", True)
        ),
    )


if __name__ == "__main__":
    main()
