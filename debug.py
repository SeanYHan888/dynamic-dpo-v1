import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _tensor_to_list(value: Any, idx: int) -> Optional[List[int]]:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        if value.ndim == 1:
            return value.tolist()
        return value[idx].tolist()
    return None


def _trim_by_mask(values: Optional[List[int]], mask: Optional[List[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    if not mask:
        return list(values)
    return [val for val, flag in zip(values, mask) if flag]


def _full_input_ids(batch: Dict[str, Any], idx: int, key: str) -> Optional[List[int]]:
    input_ids = _tensor_to_list(batch.get(f"{key}_input_ids"), idx)
    attention_mask = _tensor_to_list(batch.get(f"{key}_attention_mask"), idx)
    if input_ids is None:
        return None
    sequence = _trim_by_mask(input_ids, attention_mask)
    prompt_ids = _tensor_to_list(batch.get("prompt_input_ids"), idx)
    prompt_mask = _tensor_to_list(batch.get("prompt_attention_mask"), idx)
    prompt = _trim_by_mask(prompt_ids, prompt_mask)
    if prompt and sequence is not None:
        if sequence[: len(prompt)] != prompt:
            return prompt + sequence
    return sequence


def _get_labels(trainer: Any, batch: Dict[str, Any]) -> Tuple[Any, Any]:
    if "chosen_labels" in batch and "rejected_labels" in batch:
        return batch["chosen_labels"], batch["rejected_labels"]

    prompt_attn = batch.get("prompt_attention_mask", None)
    build_labels = getattr(trainer, "_build_labels_from_prompt", None)
    if callable(build_labels):
        chosen_labels = build_labels(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            prompt_attention_mask=prompt_attn,
        )
        rejected_labels = build_labels(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            prompt_attention_mask=prompt_attn,
        )
        return chosen_labels, rejected_labels

    chosen_labels = batch["chosen_input_ids"].clone()
    rejected_labels = batch["rejected_input_ids"].clone()
    chosen_labels[batch["chosen_attention_mask"] == 0] = -100
    rejected_labels[batch["rejected_attention_mask"] == 0] = -100
    return chosen_labels, rejected_labels


def _build_record(
    raw_record: Dict[str, Any],
    batch: Dict[str, Any],
    idx: int,
    chosen_labels: Any,
    rejected_labels: Any,
) -> Dict[str, Any]:
    return {
        "raw_record": raw_record,
        "chosen_input_ids": _tensor_to_list(batch.get("chosen_input_ids"), idx),
        "chosen_attention_mask": _tensor_to_list(batch.get("chosen_attention_mask"), idx),
        "chosen_labels": _tensor_to_list(chosen_labels, idx),
        "chosen_full_input_ids": _full_input_ids(batch, idx, "chosen"),
        "rejected_input_ids": _tensor_to_list(batch.get("rejected_input_ids"), idx),
        "rejected_attention_mask": _tensor_to_list(batch.get("rejected_attention_mask"), idx),
        "rejected_labels": _tensor_to_list(rejected_labels, idx),
        "rejected_full_input_ids": _full_input_ids(batch, idx, "rejected"),
        "prompt_input_ids": _tensor_to_list(batch.get("prompt_input_ids"), idx),
        "prompt_attention_mask": _tensor_to_list(batch.get("prompt_attention_mask"), idx),
    }


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_raw_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": sample.get("prompt"),
        "chosen": sample.get("chosen"),
        "rejected": sample.get("rejected"),
    }


def _log_samples_from_indices(
    trainer: Any,
    dataset: Any,
    indices: Sequence[int],
    *,
    output_path: str,
) -> None:
    if not indices:
        return
    raw_samples = [dataset[i] for i in indices]
    batch = trainer.data_collator(raw_samples)
    chosen_labels, rejected_labels = _get_labels(trainer, batch)
    records = []
    for i, raw in enumerate(raw_samples):
        rec = _build_record(_extract_raw_record(raw), batch, i, chosen_labels, rejected_labels)
        records.append(rec)
    _write_jsonl(output_path, records)


def _log_samples_from_dataloader(
    trainer: Any,
    dataloader: Iterable[Dict[str, Any]],
    *,
    first_n: int,
    random_n: int,
    output_path: str,
) -> None:
    rng = random.Random(0)
    first_written = 0
    random_seen = 0
    random_records: List[Dict[str, Any]] = []
    for batch in dataloader:
        chosen_labels, rejected_labels = _get_labels(trainer, batch)
        batch_size = int(batch["chosen_input_ids"].size(0))
        for idx in range(batch_size):
            raw_record = {
                "prompt": batch.get("prompt", [None])[idx] if "prompt" in batch else None,
                "chosen": batch.get("chosen", [None])[idx] if "chosen" in batch else None,
                "rejected": batch.get("rejected", [None])[idx] if "rejected" in batch else None,
            }
            record = _build_record(raw_record, batch, idx, chosen_labels, rejected_labels)
            if first_written < first_n:
                _write_jsonl(output_path, [record])
                first_written += 1
                continue
            random_seen += 1
            if len(random_records) < random_n:
                random_records.append(record)
            else:
                j = rng.randint(0, random_seen - 1)
                if j < random_n:
                    random_records[j] = record
    if random_records:
        _write_jsonl(output_path, random_records)


def log_dpo_debug_samples(
    trainer: Any,
    *,
    output_dir: str = "debug_logs",
    first_n: int = 10,
    random_n: int = 5,
) -> Optional[str]:
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        print(f"Debug logging skipped: cannot create {output_dir}")
        return None

    output_path = os.path.join(output_dir, "dpo_batch_debug.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")

    dataset = getattr(trainer, "train_dataset", None)
    if dataset is not None and hasattr(dataset, "__len__"):
        total = len(dataset)
        first_indices = list(range(min(first_n, total)))
        remaining = total - len(first_indices)
        random_indices: List[int] = []
        if random_n > 0 and remaining > 0:
            rng = random.Random(0)
            random_indices = rng.sample(
                range(len(first_indices), total),
                k=min(random_n, remaining),
            )
        _log_samples_from_indices(
            trainer, dataset, first_indices, output_path=output_path
        )
        _log_samples_from_indices(
            trainer, dataset, random_indices, output_path=output_path
        )
        return output_path

    dataloader = trainer.get_train_dataloader()
    _log_samples_from_dataloader(
        trainer,
        dataloader,
        first_n=first_n,
        random_n=random_n,
        output_path=output_path,
    )
    return output_path
