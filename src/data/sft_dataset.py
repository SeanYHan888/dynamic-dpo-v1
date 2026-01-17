from datasets import Dataset
from transformers import AutoTokenizer

from .templates import LLAMA3_CHAT_TEMPLATE, parse_hh_to_messages


def load_tokenizer(
    model_name: str,
    *,
    padding_side: str = "left",
    add_chat_template: bool = True,
    use_fast: bool = True,
) -> AutoTokenizer:
    """Load tokenizer with padding and template defaults for chat models."""
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if add_chat_template and not tok.chat_template:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    return tok


def build_sft_dataset(ds, tokenizer=None):
    """
    Convert HH dataset rows into a messages-format dataset for SFT,
    trimming any trailing assistant response to keep a prompt-only history.
    """
    rows = []
    for row in ds:
        text = row.get("chosen") if isinstance(row, dict) else None
        if text is None:
            text = row["chosen"] if "chosen" in row else None
        if text is None:
            continue

        messages = parse_hh_to_messages(text)
        if messages and messages[-1]["role"] != "assistant":
            messages = messages[:-1]
        if not any(m["role"] == "assistant" for m in messages):
            continue

        rows.append({"messages": messages})
    return Dataset.from_list(rows)
