# Repository Guidelines

## Project Structure & Module Organization
- CLI entry points live in `src/cli.py`; console scripts are defined in `pyproject.toml` (`train-sft`, `train-dpo`, `train-beta-dpo`).
- Trainers are in `src/trainers/` (`dynamic_beta_dpo.py`, `beta_dpo.py`, `sft_trainer.py`); losses in `src/losses/` (`dpo_loss.py`, `beta_dpo.py`, `margin.py`, `beta_update.py`).
- Dataset prep lives in `src/data/`: `hh_dataset.py` (HH parsing + generated-rollout loading), `sft_dataset.py`, and `templates.py` (`LLAMA3_CHAT_TEMPLATE`, `parse_hh_to_messages`).
- Quantile/EMA helpers are in `src/quantile/accumulator.py` (legacy `quantile_compute.py` is unused).
- Config loading and utilities: `src/config/loader.py`, `src/utils/logging.py` (margin logs), `src/utils/debug.py` (DPO debug samples).
- Rollout + evaluation scripts live under `test/` (`test/data_generation/`, `test/gpt_judge_HH/`, `test/gpt_judge_TLDR/`, `test/alpacaeval/`, `test/multi_dataset_testing/`). Plans/notes are in `docs/`.

## Setup & Dependencies
- Python 3.11 (`pyproject.toml`). Install with `uv sync` (preferred) or `pip install -r requirements.txt`. Match PyTorch build to your CUDA.
- Login to Hugging Face for pulls/pushes; login to Weights & Biases only if logging runs.

## Build, Train, and Data Generation Commands
- SFT reference model: `uv run train-sft --config config_sft.yaml` (uses `src/trainers/sft_trainer.py`).
- Dynamic-beta DPO: `uv run train-dpo --config config_dpo.yaml` (uses `dataset.generated_data` to pick HH parsing vs rollout loading; logs margins under `logs/margins/`).
- Beta DPO: `uv run train-beta-dpo --config config_beta_dpo.yaml` (logs margins under `logs/beta_dpo_margins/`).
- Rollouts for synthetic pairs: `python -m test.data_generation.run_rollout --config config_rollout.yaml --limit 50` (JSONL under `rollout_output/`; `manifest.json` summarizes runs).

## Coding Style & Naming Conventions
- Python, 4-space indent, type hints where practical. Prefer snake_case and config keys that mirror YAML configs.
- Reuse helpers (`src/data/sft_dataset.load_tokenizer`, `test/data_generation/utils.load_model/load_tokenizer`, `LLAMA3_CHAT_TEMPLATE`) instead of re-implementing setup. Keep logging rank-aware in distributed runs.

## Testing & Validation
- No dedicated unit tests; validate with short runs:
  - SFT: reduce `sft_training.epochs` or dataset subset, run `train-sft`.
  - Dynamic DPO: shrink `dataset.subset`/`val_ratio`, run `train-dpo`, confirm JSONL under `logs/margins/`.
  - Beta DPO: shrink `dataset.subset`/`val_ratio`, run `train-beta-dpo`, confirm JSONL under `logs/beta_dpo_margins/`.
  - Rollout: run with `--limit 10`, inspect `rollout_output/manifest.json`.
- Call `seed_everything` from `test/data_generation/utils.py` in new scripts for reproducibility.

## Commit & Pull Request Guidelines
- Commit messages are short and action-oriented (see “init RM judge…”). Use imperative mood and keep subjects under ~72 chars.
- PRs should include summary, config changes, commands run (with duration), produced artifacts (e.g., model dir), and any wandb links. Note GPU/CPU environment and required credentials.

## Configuration & Safety Tips
- Configs are in `config_dpo.yaml`, `config_sft.yaml`, `config_beta_dpo.yaml`, and `config_rollout.yaml`; adjust datasets, precision, logging, and rollout knobs here.
- Default HH dataset pull is large; use slicing (`train[:1%]`) for local checks. Avoid committing artifacts under `logs/`, `rollout_output/`, or model folders.
- Generated datasets: set `dataset.generated_data: true` and `dataset.dataset_name` to the HF dataset ID; judged JSONL follows the rollout format in `rollout_output/rollout_judged.jsonl`.
- When enabling `push_to_hub`, set `hub_model_id` and ensure credentials; `train-beta-dpo` and `train-sft` can push when configured.
