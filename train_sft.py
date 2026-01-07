from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer import DataCollatorForCompletionOnlyLM
    except ImportError:
        DataCollatorForCompletionOnlyLM = None

if DataCollatorForCompletionOnlyLM is None:
    import torch

    class DataCollatorForCompletionOnlyLM:
        def __init__(self, tokenizer, response_template, pad_to_multiple_of=None):
            self.tokenizer = tokenizer
            self.response_template = response_template
            self.pad_to_multiple_of = pad_to_multiple_of
            self.response_ids = tokenizer.encode(
                response_template, add_special_tokens=False
            )
            self.header_start_ids = tokenizer.encode(
                "<|start_header_id|>", add_special_tokens=False
            )

        @staticmethod
        def _find_subsequence_positions(sequence, subsequence):
            if not subsequence:
                return []
            positions = []
            for i in range(len(sequence) - len(subsequence) + 1):
                if sequence[i : i + len(subsequence)] == subsequence:
                    positions.append(i)
            return positions

        def _build_labels(self, input_ids, attention_mask):
            labels = [-100] * len(input_ids)
            response_starts = self._find_subsequence_positions(
                input_ids, self.response_ids
            )
            if not response_starts:
                return labels

            header_starts = self._find_subsequence_positions(
                input_ids, self.header_start_ids
            )
            for start in response_starts:
                content_start = start + len(self.response_ids)
                next_header = None
                for hs in header_starts:
                    if hs > content_start:
                        next_header = hs
                        break
                content_end = next_header if next_header is not None else len(input_ids)
                for i in range(content_start, content_end):
                    labels[i] = input_ids[i]

            for i, mask in enumerate(attention_mask):
                if mask == 0:
                    labels[i] = -100
            return labels

        def __call__(self, examples):
            if "input_ids" in examples[0]:
                batch = self.tokenizer.pad(
                    examples,
                    padding=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )
            else:
                texts = [ex["text"] for ex in examples]
                batch = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )

            labels = torch.full_like(batch["input_ids"], -100)
            for i in range(batch["input_ids"].size(0)):
                labels_i = self._build_labels(
                    batch["input_ids"][i].tolist(),
                    batch["attention_mask"][i].tolist(),
                )
                labels[i] = torch.tensor(labels_i, dtype=batch["input_ids"].dtype)
            batch["labels"] = labels
            return batch

from data_process_sft import build_sft_dataset

import argparse
import yaml


LLAMA3_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_messages(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    parts = ["<|begin_of_text|>"]
    for message in messages:
        role = message["role"]
        content = str(message["content"]).strip()
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_dpo.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)

    model_name = config["policy_name"]
    sft_cfg = config["sft_training"]
    dataset_cfg = config["dataset"]

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    raw_ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["subset"])
    sft_ds = build_sft_dataset(raw_ds, tok)

    val_ratio = float(dataset_cfg["val_ratio"])
    seed = int(dataset_cfg["seed"])
    split = sft_ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    response_template = sft_cfg.get("response_template", LLAMA3_ASSISTANT_HEADER)
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tok,
        response_template=response_template,
    )

    prec = config["precision"].lower()
    fp16 = prec == "fp16"
    bf16 = prec == "bf16"

    training_args = SFTConfig(
        output_dir=sft_cfg["save_dir"],
        learning_rate=float(sft_cfg["learning_rate"]),
        per_device_train_batch_size=int(sft_cfg["batch_size"]),
        per_device_eval_batch_size=int(sft_cfg["eval_batch_size"]),
        num_train_epochs=int(sft_cfg["epochs"]),
        logging_steps=int(sft_cfg["log_steps"]),
        eval_strategy="steps",
        eval_steps=int(sft_cfg.get("eval_steps", sft_cfg["save_steps"])),
        save_strategy="steps",
        save_steps=int(sft_cfg["save_steps"]),
        warmup_steps=int(sft_cfg["warmup_steps"]),
        max_seq_length=int(sft_cfg["max_length"]),
        fp16=fp16,
        bf16=bf16,
        report_to=["wandb"] if sft_cfg.get("wandb_project") else [],
        run_name=str(sft_cfg.get("run_name", "sft")),
        remove_unused_columns=False,
        hub_model_id=sft_cfg.get("hub_model_id"),
        push_to_hub=bool(sft_cfg.get("push_to_hub")),
    )

    def formatting_func(example):
        return format_messages(tok, example["messages"])

    wandb_project = sft_cfg.get("wandb_project")
    if wandb_project:
        import torch.distributed as dist

        is_main = (not dist.is_available()) or (not dist.is_initialized()) or (
            dist.get_rank() == 0
        )
        if is_main:
            import wandb

            wandb.init(
                project=str(wandb_project),
                name=training_args.run_name,
                config=config,
            )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=data_collator,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model()

    if bool(sft_cfg.get("push_to_hub")):
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
