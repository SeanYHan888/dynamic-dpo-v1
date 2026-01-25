# Repository Guidelines

## Project Structure & Module Organization
- Core training scripts live at `training.py` (dynamic-beta DPO), `train_sft.py` (SFT reference), and `risk_dpo_trainer.py`/`dpo_loss.py` (custom trainer + loss). Helper math lives in `quantile_compute.py`.
- Dataset prep is in `dataset_process_hh.py`: HH parsing via `build_HH_dataset`, plus generated-rollout loaders (`load_generated_dataset_from_config`) that map judged JSONL into `{prompt, chosen, rejected}`.
- Data utilities for Anthropic HH parsing and rollout generation are under `test/data_generation/` (`run_rollout.py`, `rollout.py`, `hh_parser.py`, `utils.py`).
- Configuration is centralized in `config_dpo.yaml`; adjust datasets, precision, logging, and rollout knobs here. Plans and notes reside in `docs/`. Shell automation for RunPod is in `run_and_shutdown.sh`.

## Setup & Dependencies
- Python 3.11 (`pyproject.toml`). Install with `uv sync` (preferred) or `pip install -r requirements.txt`. Match PyTorch build to your CUDA.
- Login to Hugging Face for pulls/pushes; login to Weights & Biases only if logging runs.

## Build, Train, and Data Generation Commands
- SFT reference model: `python train_sft.py --config config_dpo.yaml` (uses `LLAMA3_CHAT_TEMPLATE` + `SFTTrainer`; tune `sft_training` in the config).
- Dynamic-beta DPO: `python training.py --config config_dpo.yaml` (uses `dataset.generated_data` to pick `build_HH_dataset` vs `load_generated_dataset_from_config`; saves to `trl_dynamic_beta_out`).
- Rollouts for synthetic pairs: `python -m test.data_generation.run_rollout --config config_dpo.yaml --limit 50` (k candidates per prompt; JSONL under `rollout_output/`).

## Coding Style & Naming Conventions
- Python, 4-space indent, type hints where practical. Prefer snake_case and config keys that mirror `config_dpo.yaml`.
- Reuse helpers (`load_tokenizer`, `load_model`, `LLAMA3_CHAT_TEMPLATE`) instead of re-implementing setup. Keep logging rank-aware in distributed runs.

## Testing & Validation
- No dedicated unit tests; validate with short runs:
  - SFT: reduce `epochs` or `max_steps`, run `train_sft.py`.
  - DPO: shrink `dataset.subset`/`val_ratio`, run `training.py`, confirm JSONL under `logs/margins/`.
  - Rollout: run with `--limit 10`, inspect `rollout_output/manifest.json`.
- Call `seed_everything` in new scripts for reproducibility.

## Commit & Pull Request Guidelines
- Commit messages are short and action-oriented (see “init RM judge…”). Use imperative mood and keep subjects under ~72 chars.
- PRs should include summary, config changes, commands run (with duration), produced artifacts (e.g., model dir), and any wandb links. Note GPU/CPU environment and required credentials.

## Configuration & Safety Tips
- Default HH dataset pull is large; use slicing (`train[:1%]`) for local checks. Avoid committing artifacts under `logs/`, `rollout_output/`, or model folders.
- Generated datasets: set `dataset.generated_data: true` and `dataset.dataset_name` to the HF dataset ID; keep the judged JSONL in the rollout format shown in `docs/rollout_judged.jsonl`.
- When enabling `push_to_hub`, set `hub_model_id` and ensure credentials; `run_and_shutdown.sh` auto-confirms hub push in unattended runs.
