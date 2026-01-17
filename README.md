# Dynamic DPO Training Framework

A modular framework for Direct Preference Optimization (DPO) training with multiple algorithms:

- **Dynamic-Beta DPO**: Risk-based beta adjustment using quantile estimation
- **Beta DPO**: Adaptive data filtering with per-batch beta adjustment
- **SFT**: Supervised Fine-Tuning for base model preparation

## Installation

```bash
# Clone and install with uv
git clone <repo-url>
cd dynamic-dpo-v1
uv sync
```

## Quick Start

### 1. Supervised Fine-Tuning (SFT)

Train a base model before DPO:

```bash
uv run train-sft --config config_sft.yaml
```

### 2. Dynamic-Beta DPO Training

Uses risk-based beta adjustment with quantile threshold estimation:

```bash
uv run train-dpo --config config_dpo.yaml
```

### 3. Beta DPO Training

Uses adaptive data filtering and per-batch beta:

```bash
uv run train-beta-dpo --config config_beta_dpo.yaml
```

## Configuration

### SFT Config (`config_sft.yaml`)

```yaml
policy_name: meta-llama/Llama-3.2-1B
precision: bf16

dataset:
  dataset_name: Anthropic/hh-rlhf
  subset: train
  val_ratio: 0.1
  seed: 42

sft_training:
  learning_rate: 5e-6
  batch_size: 16
  epochs: 1
  save_dir: sft_model
```

### DPO Config (`config_dpo.yaml`)

```yaml
policy_name: your-sft-model
ref_name: your-sft-model
precision: bf16

dataset:
  dataset_name: Anthropic/hh-rlhf
  subset: train
  val_ratio: 0.1
  seed: 42

dpo_training:
  learning_rate: 5e-7
  batch_size: 16
  epochs: 1
  save_dir: dpo_model

# Dynamic-Beta specific
risk_test:
  delta: 0.1
  lambda: 0.1
  beta_warmup: 120

beta_update:
  beta_0: 0.1
  alpha: 0.005
  gamma: 2.0
  beta_min: 0.0
  beta_max: 2.0
```

### Beta DPO Config (`config_beta_dpo.yaml`)

```yaml
# Beta DPO specific
beta_dpo:
  beta_0: 0.1
  m: 0.9        # EMA momentum
  rho: 0.8      # Keep ratio for data filtering
  alpha: 0.6    # Beta adjustment factor
  min_beta: 0.001
```

## Project Structure

```
src/
├── cli.py                  # CLI entry points
├── config/                 # Configuration utilities
├── data/                   # Dataset processing
│   ├── hh_dataset.py       # HH format processing
│   ├── sft_dataset.py      # SFT data processing
│   └── templates.py        # Chat templates
├── losses/                 # Loss functions
│   ├── dpo_loss.py         # Core DPO loss
│   ├── beta_dpo.py         # Beta DPO loss
│   ├── margin.py           # Margin computation
│   └── beta_update.py      # Beta update equations
├── trainers/               # Trainer implementations
│   ├── dynamic_beta_dpo.py # Dynamic-Beta DPO
│   ├── beta_dpo.py         # Beta DPO
│   └── sft_trainer.py      # SFT training
├── quantile/               # Quantile estimation
└── utils/                  # Utilities
```

## Algorithms

### Dynamic-Beta DPO

Uses a warmup phase to estimate the margin quantile threshold, then updates beta based on risk control:

1. **Warmup**: Accumulate margins to estimate τ₀
2. **Training**: Update τ via EMA, compute p̂ = P(M ≥ τ)
3. **Beta Update**: β ← β × exp(α × tanh(γ × uₖ))

### Beta DPO

Uses adaptive data filtering and per-batch beta adjustment:

1. **Gap Computation**: gap = (π_chosen - ref_chosen) - (π_rejected - ref_rejected)
2. **Threshold Update**: EMA of mean and std of gap
3. **Data Filtering**: Gaussian-weighted sampling around threshold
4. **Beta Update**: β = β₀ × (1 + α × (gap - M₀))

## Python API

```python
from src.trainers import DynamicBetaDPOTrainer, DynamicBetaDPOConfig
from src.trainers import BetaDPOTrainer, BetaDPOConfig

# Dynamic-Beta DPO
dyn_cfg = DynamicBetaDPOConfig(beta_0=0.1, delta=0.1)
trainer = DynamicBetaDPOTrainer(
    model=policy,
    ref_model=ref,
    args=training_args,
    dynamic_cfg=dyn_cfg,
    ...
)

# Beta DPO
beta_cfg = BetaDPOConfig(beta_0=0.1, m=0.9, rho=0.8)
trainer = BetaDPOTrainer(
    model=policy,
    ref_model=ref,
    args=training_args,
    beta_dpo_cfg=beta_cfg,
    ...
)
```

## License

MIT
