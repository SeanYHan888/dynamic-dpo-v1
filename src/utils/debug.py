"""Debug utilities for DPO training."""

import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _tensor_to_list(value: Any, idx: int) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        first = value[0]
        if isinstance(first, (list, tuple)):
            return list(value[idx])
        return list(value)
    if hasattr(value, "tolist"):
        if value.ndim == 1:
            return value.tolist()
        return value[idx].tolist()
    return None


def _safe_get(sample: Any, key: str) -> Any:
    if isinstance(sample, dict):
        return sample.get(key)
    try:
        return sample[key]
    except Exception:
        return getattr(sample, key, None)


def _trim_by_mask(values: Optional[List[int]], mask: Optional[List[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    if not mask:
        return list(values)
    return [val for val, flag in zip(values, mask) if flag]


def _concatenate_and_build_labels(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    completion_input_ids: torch.Tensor,
    completion_attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate prompt + completion and build labels for log prob computation.

    This mirrors the trainer's _concatenate_and_build_labels method.
    """
    input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)

    labels = input_ids.clone()
    prompt_len = prompt_input_ids.shape[1]
    labels[:, :prompt_len] = -100
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels


def _get_concatenated_data(
    trainer: Any, batch: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get concatenated input_ids, attention_masks, and labels for chosen and rejected.

    Returns:
        chosen_input_ids, chosen_attention_mask, chosen_labels,
        rejected_input_ids, rejected_attention_mask, rejected_labels
    """
    # Try using trainer's method first
    concat_fn = getattr(trainer, "_concatenate_and_build_labels", None)
    if callable(concat_fn):
        chosen_input_ids, chosen_attention_mask, chosen_labels = concat_fn(
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            completion_input_ids=batch["chosen_input_ids"],
            completion_attention_mask=batch["chosen_attention_mask"],
        )
        rejected_input_ids, rejected_attention_mask, rejected_labels = concat_fn(
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            completion_input_ids=batch["rejected_input_ids"],
            completion_attention_mask=batch["rejected_attention_mask"],
        )
    else:
        # Fallback to local implementation
        chosen_input_ids, chosen_attention_mask, chosen_labels = _concatenate_and_build_labels(
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            completion_input_ids=batch["chosen_input_ids"],
            completion_attention_mask=batch["chosen_attention_mask"],
        )
        rejected_input_ids, rejected_attention_mask, rejected_labels = _concatenate_and_build_labels(
            prompt_input_ids=batch["prompt_input_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            completion_input_ids=batch["rejected_input_ids"],
            completion_attention_mask=batch["rejected_attention_mask"],
        )

    return (
        chosen_input_ids, chosen_attention_mask, chosen_labels,
        rejected_input_ids, rejected_attention_mask, rejected_labels,
    )


def _count_non_masked_tokens(labels: List[int]) -> int:
    """Count tokens that are not masked (-100)."""
    return sum(1 for t in labels if t != -100)


def _build_record(
    raw_record: Dict[str, Any],
    batch: Dict[str, Any],
    idx: int,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    rejected_labels: torch.Tensor,
    batch_idx: Optional[int] = None,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a debug record with concatenated sequences."""
    # Get prompt info
    prompt_input_ids = _tensor_to_list(batch.get("prompt_input_ids"), idx)
    prompt_attention_mask = _tensor_to_list(batch.get("prompt_attention_mask"), idx)
    # Use attention mask sum to get actual token count (not padded length)
    prompt_len = sum(prompt_attention_mask) if prompt_attention_mask else 0

    # Get completion-only info (from original batch)
    chosen_completion_ids = _tensor_to_list(batch.get("chosen_input_ids"), idx)
    chosen_completion_mask = _tensor_to_list(batch.get("chosen_attention_mask"), idx)
    rejected_completion_ids = _tensor_to_list(batch.get("rejected_input_ids"), idx)
    rejected_completion_mask = _tensor_to_list(batch.get("rejected_attention_mask"), idx)

    # Get concatenated info
    chosen_concat_ids = _tensor_to_list(chosen_input_ids, idx)
    chosen_concat_mask = _tensor_to_list(chosen_attention_mask, idx)
    chosen_concat_labels = _tensor_to_list(chosen_labels, idx)
    rejected_concat_ids = _tensor_to_list(rejected_input_ids, idx)
    rejected_concat_mask = _tensor_to_list(rejected_attention_mask, idx)
    rejected_concat_labels = _tensor_to_list(rejected_labels, idx)

    # Count non-masked tokens (these contribute to log prob)
    chosen_valid_tokens = _count_non_masked_tokens(chosen_concat_labels) if chosen_concat_labels else 0
    rejected_valid_tokens = _count_non_masked_tokens(rejected_concat_labels) if rejected_concat_labels else 0

    record = {
        "raw_record": raw_record,
        # Metadata
        "step": step,
        "batch_idx": batch_idx,
        "sample_idx": idx,
        # Prompt info
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "prompt_len": prompt_len,  # actual tokens (attention_mask sum)
        "prompt_padded_len": len(prompt_input_ids) if prompt_input_ids else 0,
        # Completion-only (original batch data)
        "chosen_completion_input_ids": chosen_completion_ids,
        "chosen_completion_attention_mask": chosen_completion_mask,
        "chosen_completion_len": sum(chosen_completion_mask) if chosen_completion_mask else 0,
        "chosen_completion_padded_len": len(chosen_completion_ids) if chosen_completion_ids else 0,
        "rejected_completion_input_ids": rejected_completion_ids,
        "rejected_completion_attention_mask": rejected_completion_mask,
        "rejected_completion_len": sum(rejected_completion_mask) if rejected_completion_mask else 0,
        "rejected_completion_padded_len": len(rejected_completion_ids) if rejected_completion_ids else 0,
        # Concatenated (prompt + completion)
        "chosen_concat_input_ids": chosen_concat_ids,
        "chosen_concat_attention_mask": chosen_concat_mask,
        "chosen_concat_labels": chosen_concat_labels,
        "chosen_concat_len": sum(chosen_concat_mask) if chosen_concat_mask else 0,
        "chosen_concat_padded_len": len(chosen_concat_ids) if chosen_concat_ids else 0,
        "rejected_concat_input_ids": rejected_concat_ids,
        "rejected_concat_attention_mask": rejected_concat_mask,
        "rejected_concat_labels": rejected_concat_labels,
        "rejected_concat_len": sum(rejected_concat_mask) if rejected_concat_mask else 0,
        "rejected_concat_padded_len": len(rejected_concat_ids) if rejected_concat_ids else 0,
        # Token counts for log prob computation
        "chosen_valid_tokens": chosen_valid_tokens,
        "rejected_valid_tokens": rejected_valid_tokens,
        # Bug detection: if valid_tokens == 0, log prob will be 0
        "chosen_all_masked": chosen_valid_tokens == 0,
        "rejected_all_masked": rejected_valid_tokens == 0,
    }

    return record


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]], mode: str = "a") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_raw_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": _safe_get(sample, "prompt"),
        "chosen": _safe_get(sample, "chosen"),
        "rejected": _safe_get(sample, "rejected"),
    }


def _log_samples_from_indices(
    trainer: Any,
    collate_dataset: Any,
    indices: Sequence[int],
    *,
    output_path: str,
    raw_dataset: Any = None,
    step: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not indices:
        return []
    raw_samples = [collate_dataset[i] for i in indices]
    batch = trainer.data_collator(raw_samples)

    # Get concatenated data
    (
        chosen_input_ids, chosen_attention_mask, chosen_labels,
        rejected_input_ids, rejected_attention_mask, rejected_labels,
    ) = _get_concatenated_data(trainer, batch)

    records = []
    for i, sample_idx in enumerate(indices):
        raw_sample = raw_samples[i]
        if raw_dataset is not None and hasattr(raw_dataset, "__getitem__"):
            try:
                raw_sample = raw_dataset[sample_idx]
            except Exception:
                raw_sample = raw_samples[i]
        rec = _build_record(
            _extract_raw_record(raw_sample),
            batch,
            i,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels,
            batch_idx=0,
            step=step,
        )
        records.append(rec)
    _write_jsonl(output_path, records)
    return records


def _sample_from_iterable(
    dataset: Iterable[Dict[str, Any]],
    *,
    first_n: int,
    random_n: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(0)
    first_samples: List[Dict[str, Any]] = []
    random_samples: List[Dict[str, Any]] = []
    seen_after_first = 0
    for sample in dataset:
        if len(first_samples) < first_n:
            first_samples.append(sample)
            continue
        if random_n <= 0:
            continue
        seen_after_first += 1
        if len(random_samples) < random_n:
            random_samples.append(sample)
        else:
            j = rng.randint(0, seen_after_first - 1)
            if j < random_n:
                random_samples[j] = sample
    return first_samples, random_samples


def _log_all_batches(
    trainer: Any,
    dataloader: Iterable[Dict[str, Any]],
    *,
    output_path: str,
    max_batches: Optional[int] = None,
    console_n: int = 0,
) -> int:
    """Log all samples from all batches in the dataloader.

    Returns:
        Total number of records logged.
    """
    total_records = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Get concatenated data
        (
            chosen_input_ids, chosen_attention_mask, chosen_labels,
            rejected_input_ids, rejected_attention_mask, rejected_labels,
        ) = _get_concatenated_data(trainer, batch)

        batch_size = int(batch["chosen_input_ids"].size(0))
        records = []

        for idx in range(batch_size):
            raw_record = {
                "prompt": batch.get("prompt", [None])[idx] if "prompt" in batch else None,
                "chosen": batch.get("chosen", [None])[idx] if "chosen" in batch else None,
                "rejected": batch.get("rejected", [None])[idx] if "rejected" in batch else None,
            }
            record = _build_record(
                raw_record,
                batch,
                idx,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_labels,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_labels,
                batch_idx=batch_idx,
                step=None,
            )
            records.append(record)

            if total_records < console_n:
                print(json.dumps(record, ensure_ascii=False))

            total_records += 1

        _write_jsonl(output_path, records)

    return total_records


def _log_samples_from_dataloader(
    trainer: Any,
    dataloader: Iterable[Dict[str, Any]],
    *,
    first_n: int,
    random_n: int,
    console_n: int,
    output_path: str,
) -> None:
    rng = random.Random(0)
    first_written = 0
    random_seen = 0
    random_records: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(dataloader):
        # Get concatenated data
        (
            chosen_input_ids, chosen_attention_mask, chosen_labels,
            rejected_input_ids, rejected_attention_mask, rejected_labels,
        ) = _get_concatenated_data(trainer, batch)

        batch_size = int(batch["chosen_input_ids"].size(0))
        for idx in range(batch_size):
            raw_record = {
                "prompt": batch.get("prompt", [None])[idx] if "prompt" in batch else None,
                "chosen": batch.get("chosen", [None])[idx] if "chosen" in batch else None,
                "rejected": batch.get("rejected", [None])[idx] if "rejected" in batch else None,
            }
            record = _build_record(
                raw_record,
                batch,
                idx,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_labels,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_labels,
                batch_idx=batch_idx,
                step=None,
            )
            if first_written < first_n:
                _write_jsonl(output_path, [record])
                if first_written < console_n:
                    print(json.dumps(record, ensure_ascii=False))
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
    raw_dataset: Any = None,
    console_n: int = 5,
    log_all_batches: bool = False,
    max_batches: Optional[int] = None,
) -> Optional[str]:
    """Log debug samples from DPO training.

    Args:
        trainer: The DPO trainer instance.
        output_dir: Directory to save debug logs.
        first_n: Number of first samples to log (ignored if log_all_batches=True).
        random_n: Number of random samples to log (ignored if log_all_batches=True).
        raw_dataset: Optional raw dataset for extracting original records.
        console_n: Number of samples to print to console.
        log_all_batches: If True, log every sample from every batch.
        max_batches: Maximum number of batches to log (only used if log_all_batches=True).

    Returns:
        Path to the output JSONL file, or None if logging failed.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        print(f"Debug logging skipped: cannot create {output_dir}")
        return None

    output_path = os.path.join(output_dir, "dpo_batch_debug.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")

    # If log_all_batches is True, iterate through all batches
    if log_all_batches:
        dataloader = trainer.get_train_dataloader()
        try:
            total = _log_all_batches(
                trainer,
                dataloader,
                output_path=output_path,
                max_batches=max_batches,
                console_n=console_n,
            )
            print(f"Debug logging: logged {total} samples to {output_path}")
        except Exception as exc:
            print(f"Debug logging failed on dataloader: {exc}")
        return output_path

    collate_dataset = getattr(trainer, "train_dataset", None)
    dataset = raw_dataset if raw_dataset is not None else collate_dataset
    if collate_dataset is not None:
        if hasattr(collate_dataset, "with_format"):
            try:
                collate_dataset = collate_dataset.with_format("python")
            except Exception:
                pass
        if dataset is not None and hasattr(dataset, "with_format"):
            try:
                dataset = dataset.with_format("python")
            except Exception:
                pass
        if hasattr(collate_dataset, "__len__") and hasattr(collate_dataset, "__getitem__"):
            total = len(collate_dataset)
            first_indices = list(range(min(first_n, total)))
            remaining = total - len(first_indices)
            random_indices: List[int] = []
            if random_n > 0 and remaining > 0:
                rng = random.Random(0)
                random_indices = rng.sample(
                    range(len(first_indices), total),
                    k=min(random_n, remaining),
                )
            try:
                first_records = _log_samples_from_indices(
                    trainer,
                    collate_dataset,
                    first_indices,
                    output_path=output_path,
                    raw_dataset=dataset,
                )
                for rec in first_records[: max(0, console_n)]:
                    print(json.dumps(rec, ensure_ascii=False))
                _log_samples_from_indices(
                    trainer,
                    collate_dataset,
                    random_indices,
                    output_path=output_path,
                    raw_dataset=dataset,
                )
                return output_path
            except Exception as exc:
                print(f"Debug logging failed on indexed dataset: {exc}")
        if hasattr(collate_dataset, "__iter__"):
            first_samples, random_samples = _sample_from_iterable(
                collate_dataset, first_n=first_n, random_n=random_n
            )
            if first_samples:
                first_records = _log_samples_from_indices(
                    trainer,
                    first_samples,
                    list(range(len(first_samples))),
                    output_path=output_path,
                    raw_dataset=None,
                )
                for rec in first_records[: max(0, console_n)]:
                    print(json.dumps(rec, ensure_ascii=False))
            if random_samples:
                _log_samples_from_indices(
                    trainer,
                    random_samples,
                    list(range(len(random_samples))),
                    output_path=output_path,
                    raw_dataset=None,
                )
            return output_path

    dataloader = trainer.get_train_dataloader()
    try:
        _log_samples_from_dataloader(
            trainer,
            dataloader,
            first_n=first_n,
            random_n=random_n,
            console_n=console_n,
            output_path=output_path,
        )
    except Exception as exc:
        print(f"Debug logging failed on dataloader: {exc}")
    return output_path


def log_batch_debug(
    trainer: Any,
    batch: Dict[str, Any],
    *,
    output_path: str,
    step: int,
    batch_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Log debug info for a single batch during training.

    This can be called from within compute_loss to log every batch.

    Args:
        trainer: The DPO trainer instance.
        batch: The current batch dict.
        output_path: Path to the output JSONL file.
        step: Current training step.
        batch_idx: Index of batch within step (for gradient accumulation).

    Returns:
        List of debug records.
    """
    # Get concatenated data
    (
        chosen_input_ids, chosen_attention_mask, chosen_labels,
        rejected_input_ids, rejected_attention_mask, rejected_labels,
    ) = _get_concatenated_data(trainer, batch)

    batch_size = int(batch["chosen_input_ids"].size(0))
    records = []

    for idx in range(batch_size):
        raw_record = {
            "prompt": batch.get("prompt", [None])[idx] if "prompt" in batch else None,
            "chosen": batch.get("chosen", [None])[idx] if "chosen" in batch else None,
            "rejected": batch.get("rejected", [None])[idx] if "rejected" in batch else None,
        }
        record = _build_record(
            raw_record,
            batch,
            idx,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels,
            batch_idx=batch_idx,
            step=step,
        )
        records.append(record)

    _write_jsonl(output_path, records)
    return records
