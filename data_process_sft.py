import re
from datasets import Dataset

TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")


def strip_one_leading_newline(text: str) -> str:
    """Remove a single leading newline to normalize HH blocks."""
    return text[1:] if text.startswith("\n") else text


def parse_hh_to_messages(text: str):
    """
    Parse Anthropic HH multi-turn text into [{role, content}, ...].
    Ensures content is trimmed and skips empty blocks.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("\n\nHuman:") and not text.startswith("\n\nAssistant:"):
        text = "\n\n" + text

    parts = TAG_RE.split(text)
    messages = []
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        content = strip_one_leading_newline(content).strip()
        if not content:
            continue
        role = "user" if role_tag == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


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
