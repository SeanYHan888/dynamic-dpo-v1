from datasets import load_dataset, Dataset
from typing import Any, Dict, Iterable, List, Optional

ASSISTANT_TAG = "\n\nAssistant:"

# delete the \n at the beginning of the response
def strip_one_leading_newline(s): 
    return s[1:] if s.startswith("\n") else s

def split_prompt_and_response(input_text):
    """
    HH format: multi-turn text containing many "\n\nAssistant:".
    We take the LAST Assistant tag as the start of the final assistant response.

    Returns:
    prompt: everything up to and including the final "\n\nAssistant:"
    response: the assistant completion after that tag (no leading newline)
    
    """
    input_text = str(input_text).replace("\r\n", "\n").replace("\r", "\n")
    index = input_text.rfind(ASSISTANT_TAG)
    if index < 0:
        raise ValueError("No '\\n\\nAssistant:' tag found in HH input.")
    prompt = input_text[:index + len(ASSISTANT_TAG)]
    response = input_text[index + len(ASSISTANT_TAG):]
    response = strip_one_leading_newline(response)
    return prompt, response


def convert_to_triples(chosen_text, rejected_text):
    """
    convert one HH row into an explicit triplet:
      {prompt, chosen, rejected}

    """
    # get prompt and response from chosen_text
    chosen_prompt, chosen_response = split_prompt_and_response(chosen_text)

    # assume the chosen and rejected prompts are same
    if not rejected_text.startswith(chosen_prompt):
        return None
    
    rejected_response = strip_one_leading_newline(rejected_text[len(chosen_prompt):])
    
    
    if len(chosen_prompt.strip()) == 0:
        return None
    if len(chosen_response.strip()) == 0 or len(rejected_response.strip()) == 0:
        return None
    
    return {"prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response}

# process entire dataset, build hh dataset
def build_HH_dataset(ds):
    hh_ds_raw = []
    for idx, row in enumerate(ds):
        output = convert_to_triples(
            chosen_text = row['chosen'],
            rejected_text = row['rejected']
        )
        if output is not None:
            hh_ds_raw.append(output)
    return Dataset.from_list(hh_ds_raw)


HUMAN_TAG = "\n\nHuman:"


def _normalize_text(text: Any) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _coerce_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list):
        return None
    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _normalize_text(msg.get("content", "")).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned if cleaned else None


def _messages_to_hh_prompt(messages: List[Dict[str, str]]) -> Optional[str]:
    if not messages or messages[-1]["role"] != "user":
        return None
    parts: List[str] = []
    for msg in messages:
        tag = HUMAN_TAG if msg["role"] == "user" else ASSISTANT_TAG
        parts.append(f"{tag} {msg['content']}")
    prompt = "".join(parts)
    if not prompt.endswith(ASSISTANT_TAG):
        prompt = f"{prompt}{ASSISTANT_TAG}"
    return prompt


def _extract_response_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = _normalize_text(value).strip()
        return text if text else None
    if isinstance(value, dict):
        content = _normalize_text(value.get("content", "")).strip()
        return content if content else None
    if isinstance(value, list):
        parts: List[str] = []
        for msg in value:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role is not None and role != "assistant":
                continue
            content = _normalize_text(msg.get("content", "")).strip()
            if content:
                parts.append(content)
        if parts:
            return "\n\n".join(parts)
    return None


def build_rollout_dataset(ds: Iterable[Dict[str, Any]]) -> Dataset:
    rollout_ds_raw: List[Dict[str, str]] = []
    for row in ds:
        prompt_messages = _coerce_messages(row.get("prompt_messages"))
        if prompt_messages is None:
            continue
        prompt_text = _messages_to_hh_prompt(prompt_messages)
        if not prompt_text:
            continue
        chosen_text = _extract_response_text(row.get("chosen"))
        rejected_text = _extract_response_text(row.get("rejected"))
        if not chosen_text or not rejected_text:
            continue
        rollout_ds_raw.append(
            {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        )
    return Dataset.from_list(rollout_ds_raw)


def load_generated_hf_dataset(
    dataset_name: str, *, subset: str = "train"
) -> Dataset:
    raw_ds = load_dataset(dataset_name, split=subset)
    return build_rollout_dataset(raw_ds)


def load_generated_dataset_from_config(config: Dict[str, Any]) -> Dataset:
    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("Missing dataset.dataset_name in config.")
    subset = dataset_cfg.get("subset", "train")
    return load_generated_hf_dataset(dataset_name, subset=subset)



