# SFT Implementation Plan

## Goal Description
Implement a Supervised Fine-Tuning (SFT) pipeline to produce a reference model for the existing DPO stage. This includes a data processing module and a training script that mimics the style and configuration of the existing DPO codebase.

## Key Decisions
- **Tokenizer**: Use `tokenizer` explicitly in `SFTTrainer` to ensure correct data processing and artifact saving.
- **Configuration**: Extend `config_dpo.yaml` with a new `sft_training` section to maintain a centralized configuration style.

## Proposed Changes

### Configuration
#### [MODIFY] [config_dpo.yaml](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/config_dpo.yaml)
- Add `sft_training` section with parameters:
    - `learning_rate`
    - `batch_size`
    - `eval_batch_size`
    - `epochs`
    - `log_steps`
    - `wandb_project`: string
    - `save_steps`
    - `warmup_steps`
    - `max_length`
    - `save_dir`: e.g., "sft_model"

### Data Processing
#### [NEW] [data_process_sft.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/data_process_sft.py)
- **Functions**:
    - `parse_hh_to_messages(text)`: 
        - [NEW] Parses the raw multi-turn Anthropic HH text (`\n\nHuman: ... \n\nAssistant: ...`) into a standard list of messages: `[{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]`.
        - Handles multiple turns, preserving the full conversation history.
    - `build_sft_dataset(ds, tokenizer)`:
        - Maps the raw dataset to the list-of-messages format.
        - **IMPORTANT**: For SFT, we train on **every** assistant response in the history (not just the last one).
        - We will effectively format the data so `trl.DataCollatorForCompletionOnlyLM` (or manual masking equivalent) can mask all user prompts but keep all assistant tokens.
- **Collator Strategy**:
    - Use `trl.DataCollatorForCompletionOnlyLM` with `response_template="<|start_header_id|>assistant<|end_header_id|>\n\n"` (standard for Llama 3) to automatically mask all user instructions while keeping all assistant responses (multi-turn training).

### Training Loop
#### [NEW] [train_sft.py](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/train_sft.py)
- **Imports**: `transformers`, `trl`, `datasets`, local `data_process_sft`.
- **Logic**:
    - Load config from `config_dpo.yaml`.
    - Load Tokenizer (handle `pad_token` and `chat_template`).
    - Load Model (`AutoModelForCausalLM`).
    - Load Dataset -> `build_sft_dataset` (Result: dataset of message lists).
    - **Dataset Split**: Perform `train_test_split` (using `val_ratio` from config) to create `train_dataset` and `eval_dataset`.
    - Initialize `DataCollatorForCompletionOnlyLM` with the Llama 3 assistant header as the response template.
    - Initialize `SFTConfig` (from `trl`) with `warmup_steps` and `report_to="wandb"`.
    - Initialize `SFTTrainer` passing the `data_collator`, `train_dataset`, and `eval_dataset`.
    - Run `trainer.train()`.
    - Run `trainer.save_model()`.
    - **Hub Upload**: If configured, run `trainer.push_to_hub()`.

### Configuration Update
#### [MODIFY] [config_dpo.yaml](file:///Users/seanmacbook/Research/dpo/dynamic-dpo-v1/config_dpo.yaml)
- Add to `sft_training`:
     - `push_to_hub`: boolean
     - `hub_model_id`: string ("W-61/hh-llama32-1b-sft")
- **Imports**: `transformers`, `trl`, `datasets`, local `data_process_sft`.
- **Logic**:
    - Load config from `config_dpo.yaml`.
    - Load Tokenizer (handle pad_token).
    - Load Model (`AutoModelForCausalLM`).
    - Load Dataset (`load_dataset`, `build_sft_dataset`).
    - Initialize `SFTConfig` (from `trl`) with `warmup_steps` and `report_to="wandb"`.
    - Initialize `SFTTrainer` with `formatting_func` or formatted dataset.
    - Run `trainer.train()`.
    - Run `trainer.save_model()`.

## Verification Plan

### Automated Tests
- **Dry Run**: Run the SFT training script for a few steps to ensure pipeline works.
    ```bash
    python3 train_sft.py --config config_dpo.yaml
    ```
    *Note: I will temporarily modify config or args to run a very short training (e.g., 10 steps) to verify.*

### Manual Verification
- **Artifact Check**: Check if `sft_model` directory is created and contains `config.json`, `model.safetensors`, `tokenizer.json`.
- **Loading Check**: Verify `AutoModelForCausalLM.from_pretrained("sft_model")` works.

## Monitoring
The following metrics will be tracked in WandB:
- **Loss**: `train/loss` and `eval/loss` to monitor convergence and overfitting.
- **Learning Rate**: `train/learning_rate` to verify scheduler behavior.
- **Gradient Norm**: `train/grad_norm` to detect instability.
- **Eval Accuracy**: `eval/accuracy` (Accuracy on the evaluation dataset during the evaluation loop).
