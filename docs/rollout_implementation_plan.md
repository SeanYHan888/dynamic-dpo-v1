# HH Rollout Generation Implementation Plan

This document outlines the step-by-step plan to implement rollout dataset generation from the Anthropic HH dataset. The goal is to generate multiple candidates for the next assistant turn, rank them using a judge, and create a DPO-ready dataset.

## 1. Data Preprocessing Spec

**Goal:** Convert raw HH text strings into structured, template-compliant message lists, keeping the full history.

### Parsing Logic

1. **Splitter**: Create a regex-based parser to split the raw text `\n\nHuman: ... \n\nAssistant: ...`.
    * **Rule**: Split by `\n\n(Human|Assistant):`.
2. **Normalization**: Strip role headers (`\n\nHuman:`, `\n\nAssistant:`) and leading/trailing whitespace.
    * **Structure**: Convert to a list of dicts: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`.
3. **Truncation**:
    * If the last message is `assistant`, remove it (this is the ground truth "chosen"; store for reference but do not include in prompt).
    * Ensure the list ends with a `user` message.

### Prompt Construction

1. **Templating**: Use `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.
    * **Crucial**: `add_generation_prompt=True` appends the format-specific Assistant header (e.g., `<|start_header_id|>assistant<|end_header_id|>\n\n`).
2. **Validation**: Assert the resulting string does **not** contain "Human:" or "Assistant:" inside content fields (artifacts of raw HH).

### Storage Fields

* `original_text`: Raw string (debugging).
* `prompt_messages`: List of dicts `[{role, content}, ...]`.
* `prompt_text`: Fully formatted string (after template).
* `reference_response`: Original ground truth assistant response.

---

## 2. Generation Spec

**Goal:** Efficiently generate `K` candidates per prompt without continuing the conversation.

### Configuration

* **Sampling**: `do_sample=True`, `temperature=1.0` (or 0.7), `top_p=0.9`. Higher temp helps find diverse modes.
* **Count**: `num_return_sequences=K` (e.g., K=4 or 8).
* **Limits**: `max_new_tokens=512`, `min_new_tokens=10`.

### Stop Conditions & Role Leaking

* **Issue**: Models trained on HH might regurgitate `\n\nHuman:`.
* **Fix 1 (EOS)**: Ensure `tokenizer.eos_token_id` is set correctly (e.g., `<|eot_id|>` for Llama 3).
* **Fix 2 (Stop Strings)**: Pass `stopping_criteria` to `generate()` that halts on `\n\nHuman:` or `<|start_header_id|>`.

### Batching Strategy

* **Padding**: **Left Padding** (`tokenizer.padding_side = 'left'`) for decoder-only models.
* **Masking**: Pass attention mask to ignore padding.
* **Input**: `tokenizer(prompt_texts, padding=True, return_tensors="pt")`.

---

## 3. Pair Construction Spec

**Goal:** Abstract the "Judge" for modularity.

### Judge Interface

```python
class BaseJudge:
    def rank(self, prompt: str, candidates: List[str]) -> Tuple[int, int]:
        """Returns (best_idx, worst_idx)"""
        raise NotImplementedError
```

### Implementation

* **Phase 1**: Implement `DummyJudge` (random/length-based) for pipeline testing.
* **Phase 2**: Wrap PairRM or LLM-as-a-judge.

### DPO Output Format

* `prompt`: `prompt_messages` (preferred by `trl`) OR `prompt_text`.
* `chosen`: `candidates[best_idx]`.
* `rejected`: `candidates[worst_idx]`.

---

## 4. Dataset Output

**Goal:** Clean, reproducible JSONL for DPO training.

### Schema

```json
{
  "prompt_messages": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "Best candidate text..."}],
  "rejected": [{"role": "assistant", "content": "Worst candidate text..."}],
  "metadata": {
    "source": "hh_rollout",
    "seed": 42,
    "k_candidates": 8,
    "generator_model": "Llama-3.2-1B"
  }
}
```

### Reproducibility

* Set global seed (`torch.manual_seed`, `np.random.seed`) at start.
* Log generation kwargs in `manifest.json`.

---

## 5. Integration Plan

### New Files

1. **`src/data/hh_parser.py`**:
    * `parse_hh_to_messages(text)`
    * `clean_content(text)`
    * Unit tests.
2. **`src/generation/rollout.py`**:
    * `class RolloutGenerator`: `generate_batch(prompts)`
    * `class PairSelector`: Judge implementation.
3. **`src/data_generation/run_rollout.py`**:
    * Main entry point.
    * Load model/tokenizer (reuse `train_sft.py` logic).
    * Load HH dataset (raw text).
    * Loop: Parse -> Batch -> Generate -> Rank -> Save.

### Code Reuse

* Extract `LLAMA3_CHAT_TEMPLATE` and model loading from `train_sft.py` into `src/common/utils.py`.

---

## 6. Validation Checklist

1. **Template Verification**: Confirm `prompt_text` ends with `<|start_header_id|>assistant<|end_header_id|>\n\n` and lacks raw role headers.
2. **Stop Token Check**: Generate 10 samples; ensure 0% contain `\n\nHuman:`. Adjust `stopping_criteria` if needed.
3. **Length Check**: Verify `prompt_input_ids` aren't truncated by `model_max_length`.
4. **Integration Test**: Run `python -m src.data_generation.run_rollout --limit 10`, load output with `load_dataset`, and pass to DPO collator.
