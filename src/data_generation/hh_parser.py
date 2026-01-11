from __future__ import annotations

import re
from typing import List, Optional, Tuple

TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")
RAW_ROLE_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):")


def strip_one_leading_newline(text: str) -> str:
    """Remove a single leading newline to normalize HH blocks. copy of data_process_sft.py"""
    return text[1:] if text.startswith("\n") else text


def parse_hh_to_messages(text: str):
    """
    Parse Anthropic HH multi-turn text into [{role, content}, ...].
    Ensures content is trimmed and skips empty blocks. copy of data_process_sft.py
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


def clean_content(text: str) -> str:
    """Normalize line endings and trim surrounding whitespace."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def extract_prompt_and_reference(text: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    Split HH text into a prompt (ending on a user turn) and a final assistant
    reference string. Returns (prompt_messages, reference_response).
    """
    messages = parse_hh_to_messages(text)
    if not messages:
        return None, None

    reference_response = None
    if messages[-1]["role"] == "assistant":
        reference_response = messages[-1]["content"]
        messages = messages[:-1]

    if not messages or messages[-1]["role"] != "user":
        return None, reference_response

    return messages, reference_response


def messages_have_raw_role_tags(messages: List[dict]) -> bool:
    """Return True if any content still contains raw HH role headers."""
    for msg in messages:
        content = msg.get("content", "")
        if RAW_ROLE_RE.search(content):
            return True
    return False
