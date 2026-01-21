"""Dataset loading utilities for multi-dataset testing.

This module provides unified loaders for:
1. hh-rlhf-helpfulness (Anthropic/hh-rlhf helpful subsets)
2. hh-rlhf-harmlessness (Anthropic/hh-rlhf harmless subset)
3. Custom rollout dataset (jackf857/hh-llama32-1b-Instruct-rollout-judged)
"""

from typing import Any, Dict, Optional

from datasets import Dataset, concatenate_datasets, load_dataset

# Import from main src package
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.data.hh_dataset import (
    apply_chat_template_to_dataset,
    build_HH_dataset,
    build_rollout_dataset,
)


def load_helpfulness_dataset(
    max_samples: Optional[int] = None,
    tokenizer: Any = None,
    apply_chat_template: bool = True,
) -> Dataset:
    """Load the hh-rlhf helpfulness dataset.

    Combines helpful-base and helpful-online subsets from Anthropic/hh-rlhf.

    Args:
        max_samples: Maximum number of samples to load (for normalization).
        tokenizer: Tokenizer for chat template application.
        apply_chat_template: Whether to apply chat template to prompts.

    Returns:
        Processed dataset with {prompt, chosen, rejected} format.
    """
    # Load helpful subsets
    helpful_base = load_dataset(
        "Anthropic/hh-rlhf", data_dir="helpful-base", split="train"
    )
    helpful_online = load_dataset(
        "Anthropic/hh-rlhf", data_dir="helpful-online", split="train"
    )

    # Combine datasets
    combined = concatenate_datasets([helpful_base, helpful_online])

    # Limit samples if specified
    if max_samples is not None and len(combined) > max_samples:
        combined = combined.shuffle(seed=42).select(range(max_samples))

    # Convert to triplet format
    ds = build_HH_dataset(combined)

    # Apply chat template if requested
    if apply_chat_template and tokenizer is not None:
        ds = apply_chat_template_to_dataset(ds, tokenizer)

    return ds


def load_harmlessness_dataset(
    max_samples: Optional[int] = None,
    tokenizer: Any = None,
    apply_chat_template: bool = True,
) -> Dataset:
    """Load the hh-rlhf harmlessness dataset.

    Uses the harmless-base subset from Anthropic/hh-rlhf.

    Args:
        max_samples: Maximum number of samples to load (for normalization).
        tokenizer: Tokenizer for chat template application.
        apply_chat_template: Whether to apply chat template to prompts.

    Returns:
        Processed dataset with {prompt, chosen, rejected} format.
    """
    # Load harmless subset
    harmless = load_dataset(
        "Anthropic/hh-rlhf", data_dir="harmless-base", split="train"
    )

    # Limit samples if specified
    if max_samples is not None and len(harmless) > max_samples:
        harmless = harmless.shuffle(seed=42).select(range(max_samples))

    # Convert to triplet format
    ds = build_HH_dataset(harmless)

    # Apply chat template if requested
    if apply_chat_template and tokenizer is not None:
        ds = apply_chat_template_to_dataset(ds, tokenizer)

    return ds


def load_rollout_dataset(
    dataset_name: str = "jackf857/hh-llama32-1b-Instruct-rollout-judged",
    max_samples: Optional[int] = None,
    tokenizer: Any = None,
    apply_chat_template: bool = True,
) -> Dataset:
    """Load the custom rollout dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name.
        max_samples: Maximum number of samples to load (for normalization).
        tokenizer: Tokenizer for chat template application.
        apply_chat_template: Whether to apply chat template to prompts.

    Returns:
        Processed dataset with {prompt, chosen, rejected} format.
    """
    # Load from HuggingFace
    raw_ds = load_dataset(dataset_name, split="train")

    # Limit samples if specified
    if max_samples is not None and len(raw_ds) > max_samples:
        raw_ds = raw_ds.shuffle(seed=42).select(range(max_samples))

    # Convert to triplet format using rollout builder
    ds = build_rollout_dataset(raw_ds)

    # Apply chat template if requested
    if apply_chat_template and tokenizer is not None:
        ds = apply_chat_template_to_dataset(ds, tokenizer)

    return ds


def load_dataset_by_name(
    name: str,
    max_samples: Optional[int] = None,
    tokenizer: Any = None,
    apply_chat_template: bool = True,
) -> Dataset:
    """Load a dataset by name.

    Args:
        name: One of "helpfulness", "harmlessness", or "rollout".
        max_samples: Maximum number of samples.
        tokenizer: Tokenizer for chat template application.
        apply_chat_template: Whether to apply chat template.

    Returns:
        Processed dataset.

    Raises:
        ValueError: If name is not recognized.
    """
    loaders = {
        "helpfulness": load_helpfulness_dataset,
        "harmlessness": load_harmlessness_dataset,
        "rollout": load_rollout_dataset,
    }

    if name not in loaders:
        raise ValueError(
            f"Unknown dataset name: {name}. Choose from: {list(loaders.keys())}"
        )

    return loaders[name](
        max_samples=max_samples,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
    )


def get_normalized_sample_count(
    max_samples: int,
    batch_size: int,
    gradient_accumulation: int,
) -> Dict[str, int]:
    """Calculate normalized training metrics.

    Ensures all datasets train for the same number of effective steps.

    Args:
        max_samples: Number of samples per dataset.
        batch_size: Per-device batch size.
        gradient_accumulation: Gradient accumulation steps.

    Returns:
        Dictionary with samples, effective_batch_size, and max_steps.
    """
    effective_batch_size = batch_size * gradient_accumulation
    max_steps = max_samples // effective_batch_size

    return {
        "samples": max_samples,
        "effective_batch_size": effective_batch_size,
        "max_steps": max_steps,
    }
