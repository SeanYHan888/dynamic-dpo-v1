Checklist

Verify the bug: inspect dpo_batch_debug.jsonl to confirm batches where prompt_attention_mask length exceeds chosen_input_ids length and labels become all -100.
Update both trainers to compute log‑probs on prompt+completion sequences (as TRL DPOTrainer does), and remove prompt‑masking on completion‑only inputs.
Wire truncation controls into DPOConfig (max_prompt_length, max_completion_length, max_length, truncation_mode) from config so prompts cannot swallow completions.
Re‑run a short slice and confirm margin samples are no longer all‑zero; spot‑check log‑prob sums.
Suggested Code Changes

Replace manual forward/label logic with TRL’s concatenated pipeline:

dynamic_beta_dpo.py (lines 177-220)
Use model_output = self.concatenated_forward(model, inputs) and take model_output["chosen_logps"] / model_output["rejected_logps"].
Use ref_chosen_log_prob, ref_rejected_log_prob = self.compute_ref_log_probs(inputs) inside torch.no_grad().
Keep dpo_loss_fn and margin_compute as‑is.
beta_dpo.py (lines 175-221)
Same changes as above.
This removes _build_labels_from_prompt from the loss path, which is the current source of all‑-100 labels.
Thread truncation settings into DPOConfig:

cli.py (lines 76-95) and cli.py (lines 242-262)
Add max_prompt_length, max_completion_length, max_length, truncation_mode pulled from config["dataset"].
run_experiments.py (lines 113-133)
Same additions for multi‑dataset runs.
Update configs to supply the lengths explicitly (example schema):

config_dpo.yaml / config_beta_dpo.yaml / config_*.yaml
Add dataset.max_prompt_length, dataset.max_completion_length, or map existing dataset.max_len to max_length.