# Codebase Reorganization Plan

## Overview

Reorganize the dynamic-dpo codebase into a modular `src/` package structure for better maintainability, extensibility, and clarity.

## Current Structure

```
dynamic-dpo-v1/
├── data_process_sft.py         # SFT data processing
├── dataset_process_hh.py       # HH dataset processing
├── debug.py                    # Debugging utilities
├── dpo_loss.py                 # DPO loss functions
├── quantile_compute.py         # Quantile/EMA calculations
├── risk_dpo_trainer.py         # Custom DPO trainer
├── train_sft.py                # SFT training entry point
├── training.py                 # DPO training entry point
├── util.py                     # General utilities
├── config_*.yaml               # Configuration files
├── test/                       # Evaluation scripts
│   ├── alpacaeval/
│   ├── data_generation/
│   ├── gpt_judge_HH/
│   └── gpt_judge_TLDR/
└── src/                        # Empty (placeholder)
```

## Proposed Structure

```
dynamic-dpo-v1/
├── src/
│   └── dynamic_dpo/
│       ├── __init__.py                 # Package exports
│       ├── config/
│       │   ├── __init__.py
│       │   └── loader.py               # YAML config loading
│       ├── data/
│       │   ├── __init__.py
│       │   ├── hh_dataset.py           # HH-RLHF dataset processing
│       │   ├── sft_dataset.py          # SFT data processing
│       │   └── tldr_dataset.py         # TLDR dataset processing
│       ├── loss/
│       │   ├── __init__.py
│       │   ├── dpo_loss.py             # Core DPO loss functions
│       │   └── risk.py                 # Risk test and beta update
│       ├── trainers/
│       │   ├── __init__.py
│       │   └── risk_dpo_trainer.py     # RiskBetaDPOTrainer class
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── debug.py                # Debugging utilities
│       │   ├── quantile.py             # Quantile/EMA calculations
│       │   └── templates.py            # Chat templates
│       └── eval/
│           ├── __init__.py
│           ├── gpt_oracle.py           # GPT-4 evaluation
│           └── generate.py             # Model output generation
├── scripts/
│   ├── train_sft.py                    # SFT training CLI
│   ├── train_dpo.py                    # DPO training CLI
│   └── evaluate.py                     # Evaluation CLI
├── configs/
│   ├── dpo.yaml
│   ├── sft.yaml
│   └── evaluation.yaml
├── tests/                              # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_loss.py
│   └── test_trainer.py
├── docs/
└── pyproject.toml
```

---

## Proposed Changes

### Component 1: Package Core (`src/dynamic_dpo`)

#### [NEW] src/dynamic_dpo/__init__.py

Package-level exports for easy imports:

```python
from dynamic_dpo.trainers.risk_dpo_trainer import RiskBetaDPOTrainer, RiskBetaDPOConfig
from dynamic_dpo.loss.dpo_loss import dpo_loss, compute_log_prob, margin_compute
from dynamic_dpo.data.hh_dataset import build_HH_dataset
```

---

### Component 2: Config Module (`src/dynamic_dpo/config`)

#### [NEW] src/dynamic_dpo/config/loader.py

Move YAML loading logic:

```python
def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    
def get_training_config(config: dict) -> TrainingConfig:
    """Parse raw config into structured TrainingConfig."""
```

**Source**: Extract from `training.py` (lines 18-20)

---

### Component 3: Data Module (`src/dynamic_dpo/data`)

#### [MOVE+REFACTOR] src/dynamic_dpo/data/hh_dataset.py

**Source**: `dataset_process_hh.py`

Functions to move:
- `strip_one_leading_newline()`
- `split_prompt_and_response()`
- `convert_to_triples()`
- `build_HH_dataset()`
- `build_rollout_dataset()`
- `load_generated_dataset_from_config()`
- `apply_chat_template_to_dataset()`

#### [MOVE+REFACTOR] src/dynamic_dpo/data/sft_dataset.py

**Source**: `data_process_sft.py`

Functions to move:
- `build_sft_dataset()`

#### [NEW] src/dynamic_dpo/data/tldr_dataset.py

For TLDR evaluation dataset processing (new module).

---

### Component 4: Loss Module (`src/dynamic_dpo/loss`)

#### [MOVE] src/dynamic_dpo/loss/dpo_loss.py

**Source**: `dpo_loss.py`

Functions to move:
- `compute_log_prob()`
- `margin_compute()`
- `dpo_loss()`
- `compute_and_log_model_margin()`

#### [MOVE] src/dynamic_dpo/loss/risk.py

**Source**: `dpo_loss.py` (risk-related functions)

Functions to move:
- `empirical_over_threshold_proportion()`
- `risk_test()`
- `update_beta()`

---

### Component 5: Trainers Module (`src/dynamic_dpo/trainers`)

#### [MOVE] src/dynamic_dpo/trainers/risk_dpo_trainer.py

**Source**: `risk_dpo_trainer.py`

Classes to move:
- `RiskBetaDPOConfig`
- `RiskBetaDPOTrainer`

---

### Component 6: Utils Module (`src/dynamic_dpo/utils`)

#### [MOVE] src/dynamic_dpo/utils/quantile.py

**Source**: `quantile_compute.py`

Classes/functions to move:
- `WarmupQuantileAccumulator`
- `EMAUpdate`

#### [MOVE] src/dynamic_dpo/utils/templates.py

**Source**: `util.py`

Constants/functions to move:
- `LLAMA3_CHAT_TEMPLATE`
- `parse_hh_to_messages()`

#### [MOVE] src/dynamic_dpo/utils/debug.py

**Source**: `debug.py`

Functions to move:
- `log_dpo_debug_samples()`
- Other debugging utilities

---

### Component 7: Evaluation Module (`src/dynamic_dpo/eval`)

#### [MOVE] src/dynamic_dpo/eval/gpt_oracle.py

**Source**: `test/gpt_judge_TLDR/gpt4_oracle.py` and `test/gpt_judge_HH/judge_outputs_gpt4o.py`

Consolidate GPT-4 evaluation logic.

#### [MOVE] src/dynamic_dpo/eval/generate.py

**Source**: `test/gpt_judge_TLDR/generate_summaries.py`

Model output generation logic.

---

### Component 8: Scripts (`scripts/`)

#### [MOVE+REFACTOR] scripts/train_sft.py

**Source**: `train_sft.py`

Refactored to import from `dynamic_dpo` package:

```python
from dynamic_dpo.data.sft_dataset import build_sft_dataset
from dynamic_dpo.config.loader import load_yaml
```

#### [MOVE+REFACTOR] scripts/train_dpo.py

**Source**: `training.py`

Refactored to import from `dynamic_dpo` package:

```python
from dynamic_dpo.trainers import RiskBetaDPOTrainer, RiskBetaDPOConfig
from dynamic_dpo.data.hh_dataset import build_HH_dataset
from dynamic_dpo.config.loader import load_yaml
```

---

### Component 9: Configuration Files (`configs/`)

#### [MOVE] configs/dpo.yaml

**Source**: `config_dpo.yaml`

#### [MOVE] configs/sft.yaml

**Source**: `config_sft.yaml`

#### [MOVE] configs/evaluation.yaml

**Source**: `test/model_judge/config_evaluation.yaml`

---

### Component 10: pyproject.toml Updates

#### [MODIFY] pyproject.toml

Add package configuration:

```toml
[project]
name = "dynamic-dpo"
version = "0.2.0"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[project.scripts]
train-sft = "dynamic_dpo.scripts.train_sft:main"
train-dpo = "dynamic_dpo.scripts.train_dpo:main"
evaluate = "dynamic_dpo.scripts.evaluate:main"
```

---

## Migration Steps

### Phase 1: Create Package Structure

1. Create directory structure under `src/dynamic_dpo/`
2. Create all `__init__.py` files
3. Set up package configuration in `pyproject.toml`

### Phase 2: Move Core Modules

1. Move `dpo_loss.py` → `src/dynamic_dpo/loss/dpo_loss.py`
2. Move risk functions → `src/dynamic_dpo/loss/risk.py`
3. Move `quantile_compute.py` → `src/dynamic_dpo/utils/quantile.py`
4. Move `util.py` → `src/dynamic_dpo/utils/templates.py`
5. Move `debug.py` → `src/dynamic_dpo/utils/debug.py`

### Phase 3: Move Data Processing

1. Move `dataset_process_hh.py` → `src/dynamic_dpo/data/hh_dataset.py`
2. Move `data_process_sft.py` → `src/dynamic_dpo/data/sft_dataset.py`
3. Update imports to use new package paths

### Phase 4: Move Trainer

1. Move `risk_dpo_trainer.py` → `src/dynamic_dpo/trainers/risk_dpo_trainer.py`
2. Update imports to reference new module paths

### Phase 5: Refactor Entry Points

1. Move `train_sft.py` → `scripts/train_sft.py`
2. Move `training.py` → `scripts/train_dpo.py`
3. Update all imports to use `from dynamic_dpo import ...`

### Phase 6: Organize Configs

1. Create `configs/` directory
2. Move all `config_*.yaml` files to `configs/`

### Phase 7: Clean Up Root Directory

1. Remove old Python files from root (after verification)
2. Update `.gitignore` if needed
3. Update documentation

---

## Import Examples After Reorganization

### Training Script Example

```python
# scripts/train_dpo.py
from dynamic_dpo.trainers import RiskBetaDPOTrainer, RiskBetaDPOConfig
from dynamic_dpo.data import build_HH_dataset, apply_chat_template_to_dataset
from dynamic_dpo.config import load_yaml
from dynamic_dpo.utils.debug import log_dpo_debug_samples
```

### Loss Functions Example

```python
# Use in custom training loop
from dynamic_dpo.loss import dpo_loss, margin_compute, compute_log_prob
from dynamic_dpo.loss.risk import update_beta, risk_test
```

### Evaluation Example

```python
# Run GPT-4 evaluation
from dynamic_dpo.eval import GPT4Oracle
from dynamic_dpo.data import load_tldr_dataset
```

---

## Verification Plan

### 1. Package Installation Test

```bash
# Install package in editable mode
cd /Users/seanmacbook/Research/dpo/dynamic-dpo-v1
pip install -e .

# Verify imports work
python -c "from dynamic_dpo.trainers import RiskBetaDPOTrainer; print('OK')"
python -c "from dynamic_dpo.loss import dpo_loss; print('OK')"
python -c "from dynamic_dpo.data import build_HH_dataset; print('OK')"
```

### 2. Training Script Test

```bash
# Test DPO training script runs (dry run)
python scripts/train_dpo.py --config configs/dpo.yaml --help

# Test SFT training script  
python scripts/train_sft.py --config configs/sft.yaml --help
```

### 3. Unit Tests (New)

Create basic unit tests in `tests/`:

```bash
# Run tests
python -m pytest tests/ -v
```

### 4. Manual Verification

1. Verify all Python files moved successfully
2. Check no broken imports
3. Run existing training workflow end-to-end

---

## Benefits of New Structure

1. **Modularity**: Each component is self-contained and can evolve independently
2. **Reusability**: Easy to `pip install` and use in other projects
3. **Testability**: Clear module boundaries enable unit testing
4. **Extensibility**: New trainers, losses, or data processors can be added easily
5. **Clean Root**: Root directory only contains configs, scripts, and documentation
6. **Standard Layout**: Follows Python packaging best practices

---

## Files to Delete After Migration

After verifying the new structure works:

- `data_process_sft.py`
- `dataset_process_hh.py`
- `debug.py`
- `dpo_loss.py`
- `quantile_compute.py`
- `risk_dpo_trainer.py`
- `train_sft.py`
- `training.py`
- `util.py`
- Old `config_*.yaml` files (after moving to `configs/`)
