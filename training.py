from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig

from dataset_process_hh import (
    build_HH_dataset,
    load_generated_dataset_from_config,
    apply_chat_template_to_dataset,
)
from risk_dpo_trainer import RiskBetaDPOTrainer, RiskBetaDPOConfig

import os
import argparse
import yaml

# load yaml config
def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_dpo.yaml")
    parser.add_argument("--output_dir", type=str, default="trl_dynamic_beta_out")
    args = parser.parse_args()

    config = load_yaml(args.config)

    # load policy, ref and tokenizer
    policy_name = config["policy_name"]
    ref_name = config['ref_name']

    tok = AutoTokenizer.from_pretrained(policy_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(policy_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # load dataset
    # transfer the ds into {prompt, chosen, rejected} triples
    dataset_name = config["dataset"]["dataset_name"]
    dataset_cfg = config["dataset"]
    raw_ds = load_dataset(dataset_name, split=dataset_cfg["subset"])
    if bool(dataset_cfg.get("generated_data", False)):
        hh_ds = load_generated_dataset_from_config(config)
    else:
        hh_ds = build_HH_dataset(raw_ds)  
    if bool(dataset_cfg.get("chat_template", False)):
        hh_ds = apply_chat_template_to_dataset(hh_ds, tok)

    # split train/val
    val_ratio = float(config["dataset"]["val_ratio"])
    seed = int(config["dataset"]["seed"])
    split = hh_ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    # max_len, maybe it's useful, keep it now
    max_len = int(config["dataset"]["max_len"])

    # training args
    prec = config['precision'].lower()
    fp16 = (prec == "fp16")
    bf16 = (prec == "bf16")

    # dpo config
    dpo_train_args = config["dpo_training"]
    training_args = DPOConfig(
        learning_rate=float(dpo_train_args["learning_rate"]),
        per_device_train_batch_size=int(dpo_train_args["batch_size"]),
        per_device_eval_batch_size=int(dpo_train_args['eval_batch_size']),
        num_train_epochs=int(dpo_train_args["epochs"]),
        logging_steps=int(dpo_train_args["log_steps"]),
        eval_strategy="steps",
        eval_steps=int(dpo_train_args['eval_steps']),
        save_strategy="steps",
        save_steps=int(dpo_train_args['save_steps']),
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=int(dpo_train_args['gradient_accumulation']),
        max_grad_norm=float(dpo_train_args['max_grad_norm']),
        warmup_steps=int(dpo_train_args['warmup_steps']),
        report_to=["wandb"] if dpo_train_args.get("report") else [],
        run_name=dpo_train_args['run_name'],
        remove_unused_columns=False,  
        output_dir=dpo_train_args['save_dir'],
    )

    # dynamic-beta config 
    risk = config['risk_test']
    beta_up = config['beta_update']
    margin_log = config['margin_log']

    dyn_cfg = RiskBetaDPOConfig(
        delta=float(risk['delta']),
        momentum=float(risk['lambda']),
        warmup_steps=int(risk['beta_warmup']),
        beta_0=float(beta_up['beta_0']),
        alpha=float(beta_up['alpha']),
        gamma=float(beta_up['gamma']),
        beta_min=float(beta_up['beta_min']),
        beta_max=float(beta_up['beta_max']),
        log_margins=True,
        log_dir=str(margin_log['log_dir']),
        jsonl_sample_size=int(margin_log['jsonl_sample_size']),
        save_per_rank=bool(margin_log['save_per_rank']),
    )

    # optional wandb init
    wandb_project = dpo_train_args.get("wandb_project") or dpo_train_args.get("report")
    if wandb_project:
        import torch.distributed as dist
        is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_main:
            import wandb
            wandb.init(
                project=str(wandb_project),
                name=dpo_train_args['run_name'],
                config=config,
            )

    trainer = RiskBetaDPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dynamic_cfg=dyn_cfg,
        processing_class=None
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()

