# Beta DPO V2 Integration Plan

**Goal**: Integrate the `beta_dpo_V2` DPO training variant into the existing `src/` codebase, enabling users to switch between risk-based dynamic beta and beta-DPO algorithms via configuration.

---

## Background

The existing codebase (`src/`) implements **Risk-based Dynamic Beta DPO**, which:
1. Uses a warmup phase to estimate threshold τ₀ via quantile accumulation
2. Updates τ via EMA of per-batch quantiles
3. Adjusts β based on `p_hat = P(M ≥ τ)` compared to δ
4. Uses the formula: `β ← β * exp(α * tanh(γ * u_k))`

The `beta_dpo_V2/` folder contains an alternative **Beta-DPO** algorithm that:
1. Computes margin gap differently: `gap = (π_chosen - ref_chosen) - (π_rejected - ref_rejected)`  
2. Maintains EMA-based running statistics: `M₀` (mean) and `σ` (std)
3. Applies Gaussian-weighted data filtering with probability proportional to `exp(-0.5 * z²)`
4. Updates β per-batch: `β = β₀ * (1 + α * (gap_selected_mean - M₀))`
5. Applies masked loss computation over filtered samples

---

## Proposed Changes

### src/losses

#### [NEW] [beta_dpo.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/losses/beta_dpo.py)

New module containing beta-DPO specific loss functions extracted from `beta_dpo_V2/dpo_loss.py`:

| Function | Purpose |
|----------|---------|
| `compute_beta_dpo_margin()` | Computes `(π_chosen - ref_chosen) - (π_rejected - ref_rejected)` |
| `compute_beta_dpo_threshold()` | EMA update for `M₀` and `σ` statistics |
| `beta_dpo_data_filter()` | Gaussian-weighted sampling returning mask + selected indices |
| `beta_dpo_beta_update()` | Computes `β = β₀ * (1 + α * (gap - threshold))` clamped to min |
| `beta_dpo_dpo_loss()` | DPO loss with dynamically computed beta |

---

#### [MODIFY] [__init__.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/losses/__init__.py)

Add exports for the new beta-DPO functions:

```diff
+from .beta_dpo import (
+    compute_beta_dpo_margin,
+    compute_beta_dpo_threshold,
+    beta_dpo_data_filter,
+    beta_dpo_beta_update,
+    beta_dpo_dpo_loss,
+)
```

---

### src/trainers

#### [MODIFY] [dynamic_beta_dpo.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/trainers/dynamic_beta_dpo.py)

**Config Extension** — Add beta-DPO fields to `DynamicBetaDPOConfig`:

```python
@dataclass
class DynamicBetaDPOConfig:
    # ... existing fields ...
    
    # NEW: mode selection
    mode_loss: str = "risk"  # "risk" | "beta_dpo"
    
    # NEW: beta-DPO specific parameters
    bdpo_m: float = 0.9       # EMA momentum for threshold
    bdpo_rho: float = 0.8     # Sample retention ratio
    bdpo_a: float = 0.6       # Beta scaling factor
    bdpo_min_beta: float = 1e-3
    bdpo_eps: float = 1e-6
```

**Trainer State** — Add to `__init__`:

```python
# beta-DPO EMA state
self.bdpo_M0: torch.Tensor | None = None
self.bdpo_sigma: torch.Tensor | None = None
```

**Loss Computation** — Modify `compute_loss()` to branch on `mode_loss`:

```python
mode = str(self.dynamic_cfg.mode_loss).lower()

if mode == "beta_dpo":
    # Beta-DPO pipeline
    gap = compute_beta_dpo_margin(...)
    self.bdpo_M0, self.bdpo_sigma, _, _ = compute_beta_dpo_threshold(...)
    mask, selected_idx, _ = beta_dpo_data_filter(...)
    beta_used = beta_dpo_beta_update(...)
    loss_ten, chosen_rewards, rejected_rewards = beta_dpo_dpo_loss(..., beta_used)
    loss = (loss_ten * mask).sum() / mask.sum().clamp_min(1.0)
else:
    # Existing risk-based pipeline (unchanged)
    ...
```

**Logging** — Add beta-DPO specific metrics when in beta_dpo mode:

```python
if mode == "beta_dpo":
    log_payload.update({
        "beta_dpo/beta_used": float(beta_used.item()),
        "beta_dpo/M0": float(self.bdpo_M0.item()),
        "beta_dpo/sigma": float(self.bdpo_sigma.item()),
        "beta_dpo/keep_ratio": float(mask.mean().item()),
    })
```

---

### Configuration

#### [MODIFY] [config_dpo.yaml](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/config_dpo.yaml)

Add new `beta_dpo:` configuration section:

```yaml
beta_dpo:
  mode_loss: risk       # "risk" or "beta_dpo"
  bdpo_m: 0.9           # EMA momentum
  rho: 0.8              # Sample retention ratio
  alpha: 0.5            # Beta scaling factor (note: different from risk alpha)
  min_beta: 1e-3
  eps: 1e-6
```

---

### CLI

#### [MODIFY] [cli.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/src/cli.py)

Update `main_dpo()` to parse and pass beta-DPO config:

```python
# Parse beta_dpo config (with defaults for backward compatibility)
beta_dpo_cfg = config.get("beta_dpo", {})

dyn_cfg = DynamicBetaDPOConfig(
    # ... existing fields ...
    
    # NEW: beta-DPO fields
    mode_loss=str(beta_dpo_cfg.get("mode_loss", "risk")),
    bdpo_m=float(beta_dpo_cfg.get("bdpo_m", 0.9)),
    bdpo_rho=float(beta_dpo_cfg.get("rho", 0.8)),
    bdpo_a=float(beta_dpo_cfg.get("alpha", 0.5)),
    bdpo_min_beta=float(beta_dpo_cfg.get("min_beta", 1e-3)),
    bdpo_eps=float(beta_dpo_cfg.get("eps", 1e-6)),
)
```

---

## Verification Plan

### Automated Tests

There are no existing unit tests in the `test/` directory for the trainer or loss modules. The `test/` folder contains evaluation/judge scripts rather than unit tests.

**Proposed verification approach:**

1. **Syntax/Import Check**
   ```bash
   cd /Users/seanmacbook/Research/dpo/dynamic-dpo-v1
   uv run python -c "from src.trainers.dynamic_beta_dpo import DynamicBetaDPOTrainer, DynamicBetaDPOConfig; from src.losses.beta_dpo import compute_beta_dpo_margin, beta_dpo_dpo_loss; print('Imports OK')"
   ```

2. **Dry-run with risk mode (regression test)**
   ```bash
   cd /Users/seanmacbook/Research/dpo/dynamic-dpo-v1
   uv run python -m src.cli --config config_dpo.yaml --output_dir test_risk_out
   # Verify training starts and logs appear after ~10 steps, then Ctrl+C
   ```

3. **Dry-run with beta_dpo mode**
   Create a test config `config_dpo_beta.yaml` with `beta_dpo.mode_loss: beta_dpo`, then:
   ```bash
   cd /Users/seanmacbook/Research/dpo/dynamic-dpo-v1
   uv run python -m src.cli --config config_dpo_beta.yaml --output_dir test_beta_out
   # Verify training starts and beta_dpo/* metrics appear in logs
   ```

### Manual Verification

1. **WandB Dashboard Check**: After running with beta_dpo mode, verify that the following metrics appear in WandB:
   - `beta_dpo/beta_used`
   - `beta_dpo/M0`
   - `beta_dpo/sigma`
   - `beta_dpo/keep_ratio`

2. **User Confirmation**: The user should review that training behavior matches their expectations for the beta-DPO algorithm.

---

## User Review Required

> [!IMPORTANT]
> **Mode Selection**: The plan uses a `mode_loss` config parameter to switch between "risk" (existing) and "beta_dpo" (new) algorithms. Please confirm this approach works for your use case.

> [!WARNING]
> **Backward Compatibility**: Existing configs without a `beta_dpo:` section will default to `mode_loss: risk`, preserving current behavior.

> [!NOTE]
> The `beta_dpo_V2/` folder will **not** be deleted after integration. It will remain as a reference/archive. Let me know if you'd like it removed or moved to `archive/`.
