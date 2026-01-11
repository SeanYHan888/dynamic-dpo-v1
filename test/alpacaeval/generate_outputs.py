"""
Generate model outputs for AlpacaEval 2.0 evaluation.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


DEFAULT_MODEL = "W-61/hh-llama32-1b-sft"


def load_alpacaeval_dataset():
    return load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")


def build_prompt(tokenizer, instruction):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_model_outputs(
    model_name,
    output_file,
    max_new_tokens,
    max_input_length,
    temperature,
    top_p,
    device,
):
    use_cuda = device == "cuda" and torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model.to(device)
    model.eval()

    dataset = load_alpacaeval_dataset()
    outputs = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for example in tqdm(dataset, desc="Generating"):
        instruction = example["instruction"]
        prompt = build_prompt(tokenizer, instruction)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output_text = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        outputs.append(
            {
                "instruction": instruction,
                "output": output_text,
                "generator": model_name,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} outputs to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate AlpacaEval outputs")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output_file",
        type=str,
        default="test/alpacaeval/outputs/model_outputs.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    generate_model_outputs(
        model_name=args.model_name,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )


if __name__ == "__main__":
    main()
