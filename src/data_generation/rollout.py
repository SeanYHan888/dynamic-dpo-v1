from __future__ import annotations

import random
from typing import Iterable, List, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


DEFAULT_STOP_STRINGS = ("\n\nHuman:", "<|start_header_id|>")
DEFAULT_STOP_TOKENS = ("<|eot_id|>", "<|start_header_id|>")


class BaseJudge:
    def rank(self, prompt: str, candidates: List[str]) -> Tuple[int, int]:
        """Return (best_idx, worst_idx) for the given prompt and candidates."""
        raise NotImplementedError


class DummyJudge(BaseJudge):
    """Simple judge for pipeline testing (length-based with seeded tie-breaks)."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def rank(self, prompt: str, candidates: List[str]) -> Tuple[int, int]:
        if not candidates:
            raise ValueError("No candidates to rank.")
        lengths = [len(c) for c in candidates]
        max_len = max(lengths)
        min_len = min(lengths)
        if max_len == min_len:
            # All candidates are the same length; pick two distinct indices.
            if len(candidates) < 2:
                raise ValueError("Need at least two candidates to rank.")
            best_idx, worst_idx = self._rng.sample(range(len(candidates)), 2)
            return best_idx, worst_idx
        best_indices = [i for i, l in enumerate(lengths) if l == max_len]
        worst_indices = [i for i, l in enumerate(lengths) if l == min_len]
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

    def generate_batch(self, prompt_texts: List[str]) -> List[List[str]]:
        if not prompt_texts:
            return []

        encoded = self.tokenizer(prompt_texts, padding=True, return_tensors="pt")
        input_len = encoded["input_ids"].shape[1]
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        stop_criteria = StopOnTokenSequences(
            stop_sequences=self.stop_token_sequences,
            start_index=input_len,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=self.num_return_sequences,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.generation_kwargs,
            )

        # Strip the padded prompt portion; generated tokens start after input_len.
        generated_ids = outputs[:, input_len:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [self._truncate_at_stop_strings(text) for text in decoded]

        grouped: List[List[str]] = []
        for i in range(0, len(decoded), self.num_return_sequences):
            grouped.append(decoded[i : i + self.num_return_sequences])
        return grouped
