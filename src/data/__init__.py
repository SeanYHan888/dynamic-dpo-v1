"""Dataset helpers and chat templates."""

from .hh_dataset import (
    apply_chat_template_to_dataset,
    build_HH_dataset,
    build_rollout_dataset,
    load_generated_dataset_from_config,
    load_generated_hf_dataset,
)
from .sft_dataset import build_sft_dataset, load_tokenizer
from .templates import LLAMA3_CHAT_TEMPLATE, parse_hh_to_messages, strip_one_leading_newline

__all__ = [
    "LLAMA3_CHAT_TEMPLATE",
    "parse_hh_to_messages",
    "strip_one_leading_newline",
    "apply_chat_template_to_dataset",
    "build_HH_dataset",
    "build_rollout_dataset",
    "load_generated_dataset_from_config",
    "load_generated_hf_dataset",
    "build_sft_dataset",
    "load_tokenizer",
]
