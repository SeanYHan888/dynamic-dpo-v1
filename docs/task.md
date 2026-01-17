# Codebase Reorganization Task

## Objective

Reorganize the dynamic-dpo codebase into a modular `src/` structure to make it easier to add new DPO training methods (like Beta DPO).

## Checklist

### Planning

- [ ] Explore current codebase structure
- [ ] Discuss reorganization options with user
- [ ] Create detailed implementation plan
- [ ] Get user approval on plan

### Execution

- [ ] Create new directory structure under `src/`
- [ ] Move and split `dpo_loss.py` into `losses/` modules
- [ ] Move `quantile_compute.py` to `quantile/accumulator.py`
- [ ] Move `dataset_process_hh.py` to `data/hh_dataset.py`
- [ ] Move `data_process_sft.py` to `data/sft_dataset.py`
- [ ] Move `util.py` to `data/templates.py`
- [ ] Move `risk_dpo_trainer.py` to `trainers/dynamic_beta_dpo.py`
- [ ] Create `trainers/sft_trainer.py` from `train_sft.py`
- [ ] Move `debug.py` to `utils/debug.py`
- [ ] Create `cli.py` entry point
- [ ] Create `config/loader.py`
- [ ] Update all `__init__.py` files with exports
- [ ] Update `pyproject.toml`

### Verification

- [ ] Run tests to ensure imports work
- [ ] Verify training scripts still function
