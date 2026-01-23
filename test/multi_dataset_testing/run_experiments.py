"""Main experiment runner for multi-dataset Dynamic Beta DPO testing.

This script runs the Dynamic Beta DPO training loop on three distinct datasets
using the same SFT base model, with normalized training steps for fair comparison.

Usage:
    # Run all experiments
    python -m test.multi_dataset_testing.run_experiments --all

    # Run specific experiment
    python -m test.multi_dataset_testing.run_experiments --dataset helpfulness

    # Dry run with limited steps
    python -m test.multi_dataset_testing.run_experiments --dataset helpfulness --max-steps 5

    # Validate configuration only
    python -m test.multi_dataset_testing.run_experiments --validate-only
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

# Ensure src is importable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from src.config.loader import load_yaml
from src.trainers.dynamic_beta_dpo import DynamicBetaDPOConfig, DynamicBetaDPOTrainer

from .datasets import load_dataset_by_name, get_normalized_sample_count


# Default configurations
EXPERIMENTS = ["helpfulness", "harmlessness", "rollout"]
CONFIG_DIR = Path(__file__).parent
DEFAULT_MAX_SAMPLES = 800
DEFAULT_WANDB_PROJECT = "multi-dataset-dpo-testing"


def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """Load configuration for a specific experiment.

    Args:
        experiment_name: One of "helpfulness", "harmlessness", or "rollout".

    Returns:
        Configuration dictionary.
    """
    config_path = CONFIG_DIR / f"config_{experiment_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return load_yaml(str(config_path))


def setup_models(config: Dict[str, Any]) -> tuple:
    """Load policy and reference models.

    Args:
        config: Experiment configuration.

    Returns:
        Tuple of (policy_model, ref_model, tokenizer).
    """
    policy_name = config["policy_name"]
    ref_name = config["ref_name"]

    print(f"Loading tokenizer from {policy_name}...")
    tokenizer = AutoTokenizer.from_pretrained(policy_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading policy model from {policy_name}...")
    policy = AutoModelForCausalLM.from_pretrained(policy_name)

    print(f"Loading reference model from {ref_name}...")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    return policy, ref_model, tokenizer


def setup_training_args(
    config: Dict[str, Any],
    max_steps: Optional[int] = None,
) -> DPOConfig:
    """Create DPO training arguments.

    Args:
        config: Experiment configuration.
        max_steps: Override for max training steps (for normalization).

    Returns:
        DPOConfig instance.
    """
    prec = config["precision"].lower()
    fp16 = prec == "fp16"
    bf16 = prec == "bf16"

    dpo_args = config["dpo_training"]

    training_args = DPOConfig(
        learning_rate=float(dpo_args["learning_rate"]),
        per_device_train_batch_size=int(dpo_args["batch_size"]),
        per_device_eval_batch_size=int(dpo_args["eval_batch_size"]),
        num_train_epochs=int(dpo_args["epochs"]),
        max_steps=max_steps if max_steps is not None else -1,
        logging_steps=int(dpo_args["log_steps"]),
        eval_strategy="steps",
        eval_steps=int(dpo_args["eval_steps"]),
        save_strategy="steps",
        save_steps=int(dpo_args["save_steps"]),
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=int(dpo_args["gradient_accumulation"]),
        max_grad_norm=float(dpo_args["max_grad_norm"]),
        warmup_steps=int(dpo_args["warmup_steps"]),
        report_to=["wandb"] if dpo_args.get("wandb_project") else [],
        run_name=dpo_args["run_name"],
        remove_unused_columns=False,
        output_dir=dpo_args["save_dir"],
    )

    return training_args


def setup_dynamic_config(config: Dict[str, Any]) -> DynamicBetaDPOConfig:
    """Create dynamic beta configuration.

    Args:
        config: Experiment configuration.

    Returns:
        DynamicBetaDPOConfig instance.
    """
    risk = config["risk_test"]
    beta_up = config["beta_update"]
    margin_log = config["margin_log"]

    return DynamicBetaDPOConfig(
        delta=float(risk["delta"]),
        momentum=float(risk["lambda"]),
        warmup_steps=int(risk["beta_warmup"]),
        beta_0=float(beta_up["beta_0"]),
        alpha=float(beta_up["alpha"]),
        gamma=float(beta_up["gamma"]),
        beta_min=float(beta_up["beta_min"]),
        beta_max=float(beta_up["beta_max"]),
        log_margins=True,
        log_dir=str(margin_log["log_dir"]),
        jsonl_sample_size=int(margin_log["jsonl_sample_size"]),
        save_per_rank=bool(margin_log["save_per_rank"]),
    )


def run_experiment(
    experiment_name: str,
    max_steps: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment.

    Args:
        experiment_name: Name of the experiment to run.
        max_steps: Override max training steps.
        dry_run: If True, run only 1 step for validation.

    Returns:
        Dictionary with experiment results.
    """
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'=' * 60}\n")

    # Load config
    config = load_experiment_config(experiment_name)

    # Override max_steps for dry run
    if dry_run:
        max_steps = 1
        print("DRY RUN MODE: Running 1 step only")

    # Setup models
    policy, ref_model, tokenizer = setup_models(config)

    # Load dataset
    dataset_cfg = config["dataset"]
    max_samples = dataset_cfg.get("max_samples", DEFAULT_MAX_SAMPLES)
    apply_chat_template = dataset_cfg.get("chat_template", True)
    seed = int(dataset_cfg.get("seed", 42))

    print(f"Loading {experiment_name} dataset (max_samples={max_samples})...")
    dataset = load_dataset_by_name(
        name=dataset_cfg["name"],
        max_samples=max_samples,
        tokenizer=tokenizer if apply_chat_template else None,
        apply_chat_template=apply_chat_template,
        seed=seed,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Split train/val
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    # Calculate normalized steps if not provided
    if max_steps is None:
        dpo_args = config["dpo_training"]
        step_info = get_normalized_sample_count(
            max_samples=len(train_ds),
            batch_size=int(dpo_args["batch_size"]),
            gradient_accumulation=int(dpo_args["gradient_accumulation"]),
        )
        max_steps = step_info["max_steps"]
        print(
            f"Normalized training: {max_steps} steps (batch={step_info['effective_batch_size']})"
        )

    # Setup training
    training_args = setup_training_args(config, max_steps=max_steps)
    dynamic_cfg = setup_dynamic_config(config)

    # Initialize WandB
    wandb_project = config["dpo_training"].get("wandb_project")
    if wandb_project:
        import torch.distributed as dist

        is_main = (
            (not dist.is_available())
            or (not dist.is_initialized())
            or (dist.get_rank() == 0)
        )
        if is_main:
            import wandb

            wandb.init(
                project=str(wandb_project),
                name=config["dpo_training"]["run_name"],
                config=config,
                group="multi-dataset-comparison",
                tags=[experiment_name, "dynamic-dpo"],
            )

    # Create trainer
    trainer = DynamicBetaDPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dynamic_cfg=dynamic_cfg,
        processing_class=None,
    )

    # Train
    print(f"\nStarting training for {experiment_name}...")
    trainer.train()

    # Save model
    output_dir = config["dpo_training"]["save_dir"]
    final_path = os.path.join(output_dir, "final")
    print(f"Saving model to {final_path}...")
    trainer.save_model(final_path)

    # Finish WandB run
    if wandb_project:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    return {
        "experiment": experiment_name,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "max_steps": max_steps,
        "output_dir": final_path,
    }


def validate_configs() -> bool:
    """Validate all experiment configurations.

    Returns:
        True if all configs are valid, False otherwise.
    """
    print("Validating configurations...")
    all_valid = True

    for exp in EXPERIMENTS:
        try:
            config = load_experiment_config(exp)

            # Check required keys
            required = [
                "policy_name",
                "ref_name",
                "dataset",
                "dpo_training",
                "risk_test",
                "beta_update",
            ]
            for key in required:
                if key not in config:
                    print(f"  [FAIL] {exp}: Missing required key '{key}'")
                    all_valid = False

            print(f"  [OK] {exp}: Configuration valid")
        except Exception as e:
            print(f"  [FAIL] {exp}: {e}")
            all_valid = False

    return all_valid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-dataset Dynamic Beta DPO experiments"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three experiments",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=EXPERIMENTS,
        help="Run specific experiment",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 step only for validation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configurations, don't train",
    )

    args = parser.parse_args()

    # Validate configs
    if args.validate_only:
        valid = validate_configs()
        sys.exit(0 if valid else 1)

    # Determine which experiments to run
    if args.all:
        experiments = EXPERIMENTS
    elif args.dataset:
        experiments = [args.dataset]
    else:
        parser.print_help()
        print("\nError: Specify --all or --dataset")
        sys.exit(1)

    # Run experiments
    results = []
    for exp in experiments:
        try:
            result = run_experiment(
                experiment_name=exp,
                max_steps=args.max_steps,
                dry_run=args.dry_run,
            )
            results.append(result)
            print(f"\n[SUCCESS] {exp} completed")
        except Exception as e:
            print(f"\n[ERROR] {exp} failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  {r['experiment']}: {r['train_samples']} samples, {r['max_steps']} steps -> {r['output_dir']}"
        )
    print()


if __name__ == "__main__":
    main()
