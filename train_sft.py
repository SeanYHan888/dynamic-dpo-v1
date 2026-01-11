from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from data_process_sft import LLAMA3_CHAT_TEMPLATE, build_sft_dataset, load_tokenizer

import argparse
import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_dpo.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)

    model_name = config["policy_name"]
    sft_cfg = config["sft_training"]
    dataset_cfg = config["dataset"]

    tok = load_tokenizer(model_name, padding_side="right")
    
    # Set the chat template for SFTTrainer to use
    if not tok.chat_template:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(model_name)

    raw_ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["subset"])
    sft_ds = build_sft_dataset(raw_ds, tok)

    val_ratio = float(dataset_cfg["val_ratio"])
    seed = int(dataset_cfg["seed"])
    split = sft_ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    prec = config["precision"].lower()
    fp16 = prec == "fp16"
    bf16 = prec == "bf16"
    
    # Use built-in completion_only_loss which leverages the tokenizer's chat template
    # SFTTrainer will automatically look for the "messages" column in the dataset
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
        max_length=int(sft_cfg["max_length"]),
        save_only_model=bool(sft_cfg.get("save_only_model", True)),
        fp16=fp16,
        bf16=bf16,
        report_to=["wandb"] if sft_cfg.get("wandb_project") else [],
        run_name=str(sft_cfg.get("run_name", "sft")),
        remove_unused_columns=False,
        hub_model_id=sft_cfg.get("hub_model_id"),
        push_to_hub=bool(sft_cfg.get("push_to_hub")),
        dataset_text_field="messages",  # Explicitly tell it to look for messages
        completion_only_loss=True,     # New TRL 0.20+ way to mask prompts
    )

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
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model()

    # if bool(sft_cfg.get("push_to_hub")):
    import torch.distributed as dist
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    if is_main:
        hub_model_id = sft_cfg.get("hub_model_id")
        if hub_model_id:
            try:
                push = input(f"\nDo you want to push the model to the Hub ({hub_model_id})? [y/N]: ").strip().lower()
                if push == "y":
                    print(f"Pushing model to {hub_model_id}...")
                    trainer.push_to_hub()
            except EOFError:
                # Handle case where input is not available (e.g. non-interactive monitoring)
                pass
        else:
            print("No hub_model_id configured, skipping interactive push.")


if __name__ == "__main__":
    main()
