# Codebase Reorganization: Modular `src/` Structure

This plan reorganizes the dynamic-dpo codebase into a modular structure under `src/`, making it easy to add new DPO training methods like Beta DPO.

## Target Directory Structure

```
dynamic-dpo-v1/
├── src/
│   ├── __init__.py
│   ├── cli.py                    # Entry points for train_dpo, train_sft
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py             # load_yaml() utility
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hh_dataset.py         # HH dataset processing
│   │   ├── sft_dataset.py        # SFT data processing
│   │   └── templates.py          # Chat templates, HH parsing
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── dpo_loss.py           # Core DPO loss, log-prob computation
│   │   ├── margin.py             # Margin computation, risk test
│   │   └── beta_update.py        # Beta update equations
│   │
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py        # SFT training logic
│   │   ├── dpo_trainer.py        # Standard DPO trainer (placeholder)
│   │   └── dynamic_beta_dpo.py   # Dynamic-beta DPO trainer
│   │
│   ├── quantile/
│   │   ├── __init__.py
│   │   └── accumulator.py        # WarmupQuantileAccumulator, EMAUpdate
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py            # Margin logging utilities
│       └── debug.py              # Debug sample logging
│
├── test/                         # Unchanged location
├── docs/                         # Unchanged
├── config_dpo.yaml               # Unchanged location
├── config_sft.yaml
├── config_rollout.yaml
├── config_dpo_original.yaml
├── pyproject.toml                # Updated for src package
└── README.md
```

---

## Proposed Changes

### src/config/

#### [NEW] [loader.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/config/loader.py)
Extract `load_yaml()` function from `training.py`.

---

### src/data/

#### [NEW] [hh_dataset.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/data/hh_dataset.py)
Move content from [dataset_process_hh.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/dataset_process_hh.py):
- `build_HH_dataset()`, `load_generated_dataset_from_config()`, `apply_chat_template_to_dataset()`
- Helper functions: `split_prompt_and_response()`, `convert_to_triples()`, `build_rollout_dataset()`, etc.

#### [NEW] [sft_dataset.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/data/sft_dataset.py)
Move content from [data_process_sft.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/data_process_sft.py):
- `build_sft_dataset()`, `load_tokenizer()`

#### [NEW] [templates.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/data/templates.py)
Move content from [util.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/util.py):
- `LLAMA3_CHAT_TEMPLATE`, `parse_hh_to_messages()`, `strip_one_leading_newline()`

---

### src/losses/

#### [NEW] [dpo_loss.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/losses/dpo_loss.py)
From [dpo_loss.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/dpo_loss.py):
- `dpo_loss()`, `compute_log_prob()`

#### [NEW] [margin.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/losses/margin.py)
From [dpo_loss.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/dpo_loss.py):
- `margin_compute()`, `empirical_over_threshold_proportion()`, `risk_test()`

#### [NEW] [beta_update.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/losses/beta_update.py)
From [dpo_loss.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/dpo_loss.py):
- `update_beta()`

---

### src/trainers/

#### [NEW] [dynamic_beta_dpo.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/trainers/dynamic_beta_dpo.py)
Move from [risk_dpo_trainer.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/risk_dpo_trainer.py):
- `RiskBetaDPOConfig` → rename to `DynamicBetaDPOConfig`
- `RiskBetaDPOTrainer` → rename to `DynamicBetaDPOTrainer`
- Update imports to use new `src.losses.*` and `src.quantile.*` paths

#### [NEW] [sft_trainer.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/trainers/sft_trainer.py)
Extract training logic from [train_sft.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/train_sft.py) into a reusable function:
- `run_sft_training(config: dict) -> SFTTrainer`

#### [NEW] [dpo_trainer.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/trainers/dpo_trainer.py)
Placeholder for standard DPO trainer (static beta). Can be added later or left as empty file with TODO comment.

---

### src/quantile/

#### [NEW] [accumulator.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/quantile/accumulator.py)
Move from [quantile_compute.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/quantile_compute.py):
- `WarmupQuantileAccumulator`, `EMAUpdate`

---

### src/utils/

#### [NEW] [logging.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/utils/logging.py)
From [dpo_loss.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/dpo_loss.py):
- `compute_and_log_model_margin()`

#### [NEW] [debug.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/utils/debug.py)
Move from [debug.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/debug.py):
- All functions unchanged

---

### src/cli.py

#### [NEW] [cli.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/cli.py)
Consolidate entry points from [training.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/training.py) and [train_sft.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/train_sft.py):
- `main_dpo()` - DPO training entry point
- `main_sft()` - SFT training entry point

---

### Root Files

#### [MODIFY] [pyproject.toml](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/pyproject.toml)
Add package configuration and entry points:
```toml
[project.scripts]
train-dpo = "src.cli:main_dpo"
train-sft = "src.cli:main_sft"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

#### [DELETE] Root-level Python files after migration
After verification, the following files at root can be deleted:
- `training.py` → moved to `src/cli.py`
- `train_sft.py` → moved to `src/cli.py` + `src/trainers/sft_trainer.py`
- `dpo_loss.py` → split into `src/losses/`
- `risk_dpo_trainer.py` → moved to `src/trainers/dynamic_beta_dpo.py`
- `quantile_compute.py` → moved to `src/quantile/accumulator.py`
- `dataset_process_hh.py` → moved to `src/data/hh_dataset.py`
- `data_process_sft.py` → moved to `src/data/sft_dataset.py`
- `util.py` → moved to `src/data/templates.py`
- `debug.py` → moved to `src/utils/debug.py`

---

### Test Files

#### [MODIFY] Test imports
Update imports in 19 test files across `test/` subdirectories. Example changes:
```python
# Before
from dpo_loss import dpo_loss

# After
from src.losses.dpo_loss import dpo_loss
```

Files to update (if they import from root modules):
- `test/data_generation/utils.py`
- `test/data_generation/rollout.py`
- Other files as needed (will check each file during execution)

---

## File Mapping Summary

| Current File | New Location |
|--------------|--------------|
| `training.py` | `src/cli.py` (main_dpo) |
| `train_sft.py` | `src/cli.py` (main_sft) + `src/trainers/sft_trainer.py` |
| `risk_dpo_trainer.py` | `src/trainers/dynamic_beta_dpo.py` |
| `dpo_loss.py` | Split: `src/losses/dpo_loss.py`, `margin.py`, `beta_update.py`, `src/utils/logging.py` |
| `quantile_compute.py` | `src/quantile/accumulator.py` |
| `dataset_process_hh.py` | `src/data/hh_dataset.py` |
| `data_process_sft.py` | `src/data/sft_dataset.py` |
| `util.py` | `src/data/templates.py` |
| `debug.py` | `src/utils/debug.py` |

---

## Verification Plan

### 1. Import Validation (Automated)

Run Python import check to verify all modules are importable:
```bash
cd /Users/seanmacbook/Research/dpo/dynamic-dpo-v1
uv run python -c "
from src.config.loader import load_yaml
from src.data.hh_dataset import build_HH_dataset
from src.data.sft_dataset import build_sft_dataset
from src.data.templates import LLAMA3_CHAT_TEMPLATE
from src.losses.dpo_loss import dpo_loss, compute_log_prob
from src.losses.margin import margin_compute
from src.losses.beta_update import update_beta
from src.trainers.dynamic_beta_dpo import DynamicBetaDPOTrainer, DynamicBetaDPOConfig
from src.quantile.accumulator import WarmupQuantileAccumulator, EMAUpdate
from src.utils.debug import log_dpo_debug_samples
print('All imports successful!')
"
```

### 2. CLI Entry Points (Automated)

Verify CLI commands are registered:
```bash
uv run train-dpo --help
uv run train-sft --help
```

### 3. DPO Training Dry Run (Manual)

Run a minimal DPO training to verify the full pipeline works:
```bash
# Create a minimal test config or use existing one with small dataset
uv run train-dpo --config config_dpo.yaml
```
**Expected:** Training starts without import errors. Can be stopped after a few steps.

### 4. SFT Training Dry Run (Manual)

```bash
uv run train-sft --config config_sft.yaml
```
**Expected:** Training starts without import errors.

> [!NOTE]
> The test files in `test/` are evaluation/generation scripts (AlpacaEval, GPT judge), not unit tests. I'll update their imports but there's no automated test suite to run. Manual verification via dry runs is the primary validation method.

---

## User Review Required

> [!IMPORTANT]
> **Class Renaming:** The plan renames `RiskBetaDPOTrainer` → `DynamicBetaDPOTrainer` and `RiskBetaDPOConfig` → `DynamicBetaDPOConfig`. This may require updates to any external scripts or documentation referencing the old names.

> [!IMPORTANT]
> **Deletion of Root Files:** After migration and verification, original files will be deleted from root. Ensure you have committed the current state before proceeding.
