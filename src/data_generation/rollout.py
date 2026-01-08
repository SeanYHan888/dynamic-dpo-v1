from __future__ import annotations

import random
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, StoppingCriteria


DEFAULT_STOP_STRINGS = ("\n\nHuman:",)
DEFAULT_STOP_TOKENS = ("<|eot_id|>",)


class BaseJudge:
    def rank(self, prompt: str, candidates: List[str]) -> Tuple[int, int]:
        """Return (best_idx, worst_idx) for the given prompt and candidates."""
        raise NotImplementedError


class RMJudge(BaseJudge):
    """Reward-model judge that ranks candidates by score."""

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
        precision: str | None = None,
        device_map: str | None = None,
        load_in_8bit: bool = False,
        batch_size: int = 4,
        seed: int = 42,
        max_length: int | None = None,
    ):
        self._rng = random.Random(seed)
        self.batch_size = int(batch_size)
        self.max_length = max_length

        # ArmoRM custom modeling expects these symbols; newer transformers dropped them.
        self._ensure_llama_docstring()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = None
        if precision and not load_in_8bit:
            prec = precision.lower()
            if prec == "fp16":
                dtype = torch.float16
            elif prec == "bf16":
                dtype = torch.bfloat16

        kwargs = {"trust_remote_code": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes quantization requested but BitsAndBytesConfig is unavailable."
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if device_map is None:
                device_map = "auto"
        if device_map is not None:
            kwargs["device_map"] = device_map

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        except ImportError as exc:
            msg = str(exc)
            if "LLAMA_INPUTS_DOCSTRING" in msg or "LLAMA_START_DOCSTRING" in msg:
                self._ensure_llama_docstring()
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
            else:
                raise
        self.model.eval()

        if device_map is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self._device = device
        else:
            self._device = None

        if self._device is None:
            try:
                self._device = next(self.model.parameters()).device
            except StopIteration:
                self._device = None

    def _build_texts(self, prompt: str | List[dict], candidates: List[str]) -> List[str]:
        texts: List[str] = []
        if isinstance(prompt, list):
            for cand in candidates:
                messages = prompt + [{"role": "assistant", "content": cand}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return texts

        suffix = self.tokenizer.eos_token or ""
        for cand in candidates:
            text = f"{prompt}{cand}"
            if suffix and not text.endswith(suffix):
                text = f"{text}{suffix}"
            texts.append(text)
        return texts

    @staticmethod
    def _ensure_llama_docstring() -> None:
        try:
            from transformers.models.llama import modeling_llama

            if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
                modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
            if not hasattr(modeling_llama, "LLAMA_START_DOCSTRING"):
                modeling_llama.LLAMA_START_DOCSTRING = ""
        except Exception:
            # If llama module moves again, we fall back to letting HF raise.
            pass

    def _score_texts(self, texts: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=self.max_length is not None,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self._device is not None:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_scores = logits.squeeze(-1).detach().float().cpu().tolist()
                scores.extend(batch_scores)
        return scores

    def rank(self, prompt: str | List[dict], candidates: List[str]) -> Tuple[int, int]:
        if not candidates:
            raise ValueError("No candidates to rank.")
        texts = self._build_texts(prompt, candidates)
        scores = self._score_texts(texts)
        max_score = max(scores)
        min_score = min(scores)
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        worst_indices = [i for i, s in enumerate(scores) if s == min_score]
        best_idx = self._rng.choice(best_indices)
        worst_idx = self._rng.choice(worst_indices)
        return best_idx, worst_idx


class StopOnTokenSequences(StoppingCriteria):
    """Stop generation once any stop token sequence appears in the new tokens."""

    def __init__(self, stop_sequences: Iterable[List[int]], start_index: int):
        self._stop_sequences = [list(seq) for seq in stop_sequences if seq]
        self._start_index = int(start_index)
        self._triggered: List[bool] | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch = input_ids.shape[0]
        if self._triggered is None or len(self._triggered) != batch:
            self._triggered = [False] * batch

        for i in range(batch):
            if self._triggered[i]:
                continue
            generated_len = input_ids.shape[1] - self._start_index
            if generated_len <= 0:
                continue
            for seq in self._stop_sequences:
                if generated_len < len(seq):
                    continue
                if input_ids[i, -len(seq) :].tolist() == seq:
                    self._triggered[i] = True
                    break

        return all(self._triggered)


class RolloutGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        num_return_sequences: int,
        stop_strings: Iterable[str] = DEFAULT_STOP_STRINGS,
        stop_tokens: Iterable[str | int] = DEFAULT_STOP_TOKENS,
        **generation_kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_return_sequences = int(num_return_sequences)
        self.stop_strings = tuple(stop_strings)
        self.stop_token_sequences = self._build_stop_sequences(stop_tokens)
        self.generation_kwargs = generation_kwargs

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _truncate_at_stop_strings(self, text: str) -> str:
        stop_positions = [text.find(s) for s in self.stop_strings if s in text]
        if stop_positions:
            text = text[: min(stop_positions)]
        return text.strip()

    def _build_stop_sequences(self, stop_tokens: Iterable[str | int]) -> List[List[int]]:
        sequences: List[List[int]] = []
        for token in stop_tokens:
            if isinstance(token, int):
                sequences.append([token])
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                sequences.append([token_id])
                continue
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                sequences.append(encoded)

        if self.tokenizer.eos_token_id is not None:
            sequences.append([self.tokenizer.eos_token_id])

        unique: List[List[int]] = []
        seen = set()
        for seq in sequences:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            unique.append(seq)
        return unique

    def generate_batch(self, prompt_texts: List[str], *, return_raw: bool = False):
        if not prompt_texts:
            return [] if not return_raw else ([], [])

        encoded = self.tokenizer(prompt_texts, padding=True, return_tensors="pt")
        input_len = encoded["input_ids"].shape[1]
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        eos_ids = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            eos_ids.append(eot_id)
        eos_ids = [eid for eid in eos_ids if eid is not None]
        if len(eos_ids) == 1:
            eos_ids = eos_ids[0]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=self.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                **self.generation_kwargs,
            )

        # Strip the padded prompt portion; generated tokens start after input_len.
        generated_ids = outputs[:, input_len:]
        raw_decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [self._truncate_at_stop_strings(text) for text in decoded]

        grouped: List[List[str]] = []
        raw_grouped: List[List[str]] = []
        for i in range(0, len(decoded), self.num_return_sequences):
            grouped.append(decoded[i : i + self.num_return_sequences])
            raw_grouped.append(raw_decoded[i : i + self.num_return_sequences])
        return (grouped, raw_grouped) if return_raw else grouped
