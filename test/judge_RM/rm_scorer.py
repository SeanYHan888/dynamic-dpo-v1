from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
_LEADING_ASSISTANT_RE = re.compile(r"^\s*(assistant\s*:?)\s*\n+", flags=re.IGNORECASE)


def normalize_model_output(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = _LEADING_ASSISTANT_RE.sub("", text).lstrip()
    text = _SPECIAL_TOKEN_RE.sub("", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_LLAMA3_FALLBACK_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = message['content'] %}"
    "{% if loop.index0 == 0 %}"
    "{{ '<|begin_of_text|>' }}"
    "{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + content | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def _resolve_dtype(precision: str | None) -> torch.dtype | None:
    if not precision:
        return None
    precision = precision.lower().strip()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _ensure_llama_docstrings() -> None:
    try:
        from transformers.models.llama import modeling_llama

        if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
            modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
        if not hasattr(modeling_llama, "LLAMA_START_DOCSTRING"):
            modeling_llama.LLAMA_START_DOCSTRING = ""
    except Exception:
        return


@dataclass(frozen=True)
class RMConfig:
    reward_model: str
    tokenizer_name: str | None = None
    precision: str | None = None
    device_map: str | None = "auto"
    load_in_8bit: bool = False
    batch_size: int = 8
    max_length: int | None = None


class RewardModelScorer:
    def __init__(self, config: RMConfig):
        self.config = config
        self.batch_size = max(1, int(config.batch_size))

        _ensure_llama_docstrings()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.reward_model, use_fast=True
        )
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = _LLAMA3_FALLBACK_CHAT_TEMPLATE
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {"trust_remote_code": True}
        dtype = _resolve_dtype(config.precision)
        if dtype is not None and not config.load_in_8bit:
            model_kwargs["torch_dtype"] = dtype

        if config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "load_in_8bit=True requires bitsandbytes + BitsAndBytesConfig."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = config.device_map or "auto"
        elif config.device_map is not None:
            model_kwargs["device_map"] = config.device_map

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.reward_model, **model_kwargs
            )
        except ImportError as exc:
            if "LLAMA_INPUTS_DOCSTRING" in str(exc) or "LLAMA_START_DOCSTRING" in str(exc):
                _ensure_llama_docstrings()
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    config.reward_model, **model_kwargs
                )
            else:
                raise

        self.model.eval()

        self._input_device: torch.device | None
        if config.device_map is None and not config.load_in_8bit:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self._input_device = device
        else:
            self._input_device = None

    def build_texts(
        self, instructions: Sequence[str], responses: Sequence[str]
    ) -> List[str]:
        if len(instructions) != len(responses):
            raise ValueError("instructions and responses must have the same length.")
        texts: list[str] = []
        for instruction, response in zip(instructions, responses):
            messages = [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(response)},
            ]
            used_chat_template = bool(
                getattr(self.tokenizer, "apply_chat_template", None)
                and self.tokenizer.chat_template
            )
            if used_chat_template:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = f"{instruction}\n\n{response}"
            if (
                not used_chat_template
                and self.tokenizer.eos_token
                and not text.endswith(self.tokenizer.eos_token)
            ):
                text = f"{text}{self.tokenizer.eos_token}"
            texts.append(text)
        return texts

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=self.config.max_length is not None,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            if self._input_device is not None:
                inputs = {k: v.to(self._input_device) for k, v in inputs.items()}

            with torch.inference_mode():
                output = self.model(**inputs)
                logits = output.logits
                batch_scores = logits.squeeze(-1).detach().float().cpu().tolist()
                if not isinstance(batch_scores, list):
                    batch_scores = [float(batch_scores)]
                scores.extend([float(value) for value in batch_scores])
        return scores

    def score_instruction_responses(
        self, instructions: Sequence[str], responses: Sequence[str]
    ) -> List[float]:
        texts = self.build_texts(instructions, responses)
        return self.score_texts(texts)
