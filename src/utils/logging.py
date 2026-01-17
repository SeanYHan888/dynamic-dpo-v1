import json
import os

import numpy as np


# compute and log model margin
def compute_and_log_model_margin(model_margin, epoch_dir, epoch, step, JSONL_PATH):
    # full array
    # using numpy to process, so only on cpu
    m = model_margin.detach().float().cpu().numpy()

    # 1) save full margins as .npy (raw, lossless)
    # step: batch index
    npy_path = os.path.join(epoch_dir, f"step_{step:05d}.npy")
    np.save(npy_path, m)

    # 2) write a readable per-batch record to ONE jsonl file
    # summary stats
    # quantiles
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

    # save in the jsonl file
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
