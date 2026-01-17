"""Margin logging utilities."""

import json
import os

import numpy as np
import torch


def compute_and_log_model_margin(
    model_margin: torch.Tensor,
    epoch_dir: str,
    epoch: int,
    step: int,
    jsonl_path: str,
) -> None:
    """Compute and log model margin statistics.
    
    Saves margins as both .npy files (raw) and to a JSONL file (summary stats).
    
    Args:
        model_margin: Tensor of margin values.
        epoch_dir: Directory to save .npy files.
        epoch: Current epoch number.
        step: Current batch step.
        jsonl_path: Path to the JSONL summary file.
    """
    m = model_margin.detach().float().cpu().numpy()

    # Save full margins as .npy (raw, lossless)
    npy_path = os.path.join(epoch_dir, f"step_{step:05d}.npy")
    np.save(npy_path, m)

    # Write readable per-batch record to JSONL file
    p10, p50, p90 = np.percentile(m, [10, 50, 90])

    record = {
        "epoch": int(epoch),
        "step": int(step),
        "batch_size": int(m.shape[0]),
        "mean": float(m.mean()),
        "std": float(m.std(ddof=0)),
        "min": float(m.min()),
        "p10": float(p10),
        "median": float(p50),
        "p90": float(p90),
        "max": float(m.max()),
        "pos_frac": float((m > 0).mean()),
        "npy": npy_path,
        "sample": [float(x) for x in m[:]],
    }

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
