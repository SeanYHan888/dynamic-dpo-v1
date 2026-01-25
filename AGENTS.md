# Repository Guidelines

## Project Structure & Module Organization
- CLI entry points live in `src/cli.py` (`main_dpo`, `main_beta_dpo`, `main_sft`) and are exposed as `train-dpo`, `train-beta-dpo`, and `train-sft` via `pyproject.toml`.
- Trainers are in `src/trainers/` (`dynamic_beta_dpo.py`, `beta_dpo.py`, `sft_trainer.py`); losses are in `src/losses/` (`dpo_loss.py`, `beta_dpo.py`, `margin.py`, `beta_update.py`).
- Quantile/risk helpers live in `src/quantile/accumulator.py`.
- Dataset prep is in `src/data/hh_dataset.py` (`build_HH_dataset`, `load_generated_dataset_from_config`, `apply_chat_template_to_dataset`), with templates in `src/data/templates.py` and SFT data utilities in `src/data/sft_dataset.py`.
- Rollout generation utilities are under `test/data_generation/` (`run_rollout.py`, `rollout.py`, `hh_parser.py`, `utils.py`).
- Configs are `config_sft.yaml`, `config_dpo.yaml`, `config_beta_dpo.yaml`, `config_rollout.yaml` (plus `config_dpo_original.yaml` for baseline); plans and notes live in `docs/`.

## Setup & Dependencies
- Python 3.11 (`pyproject.toml`). Install with `uv sync` (preferred) or `pip install -r requirements.txt`. Match PyTorch build to your CUDA.
- Login to Hugging Face for pulls/pushes; login to Weights & Biases only if logging runs.

## Build, Train, and Data Generation Commands
- SFT reference model: `uv run train-sft --config config_sft.yaml` (uses `sft_training` in the config).
- Dynamic-beta DPO: `uv run train-dpo --config config_dpo.yaml` (uses `dataset.generated_data` to pick `build_HH_dataset` vs `load_generated_dataset_from_config`; saves checkpoints under `dpo_training.save_dir` and the final model under `--output_dir`).
- Beta DPO: `uv run train-beta-dpo --config config_beta_dpo.yaml`.
- Rollouts for synthetic pairs: `python -m test.data_generation.run_rollout --config config_rollout.yaml --limit 50` (uses the `rollout` section; outputs under `rollout_output/`).

## Coding Style & Naming Conventions
- Python, 4-space indent, type hints where practical. Prefer snake_case and config keys that mirror `config_*.yaml`.
- Reuse helpers (`src/config/loader.py`, `src/data/templates.py`, `test/data_generation/utils.py`) instead of re-implementing setup. Keep logging rank-aware in distributed runs.

## Testing & Validation
- No dedicated unit tests; validate with short runs:
  - SFT: reduce `sft_training.epochs` or `max_steps`, run `uv run train-sft --config config_sft.yaml`.
  - Dynamic-beta DPO: shrink `dataset.subset`/`val_ratio`, run `uv run train-dpo --config config_dpo.yaml`, confirm JSONL/NPY under `margin_log.log_dir` (default `logs/margins/`).
  - Beta DPO: run `uv run train-beta-dpo --config config_beta_dpo.yaml`, confirm logs under `beta_dpo.log_dir` if enabled.
  - Rollout: run with `--limit 10`, inspect `rollout_output/manifest.json`.
- Call `seed_everything` (see `test/data_generation/utils.py`) in new scripts for reproducibility.

## Commit & Pull Request Guidelines
- Commit messages are short and action-oriented (see “init RM judge…”). Use imperative mood and keep subjects under ~72 chars.
- PRs should include summary, config changes, commands run (with duration), produced artifacts (e.g., model dir), and any wandb links. Note GPU/CPU environment and required credentials.

## Configuration & Safety Tips
- Default HH dataset pull is large; use slicing (`train[:1%]`) for local checks. Avoid committing artifacts under `logs/`, `rollout_output/`, or model folders like `sft_model/`, `trl_dynamic_beta_dpo/`, `trl_dynamic_beta_out/`, `beta_dpo_model/`, or `beta_dpo_out/`.
- Generated datasets: set `dataset.generated_data: true` and `dataset.dataset_name` to the HF dataset ID produced by the rollout pipeline in `rollout_output/`.
- When enabling `hub_model_id`, ensure Hugging Face credentials; SFT prompts for an interactive push and Beta DPO auto-uploads from the main process.
