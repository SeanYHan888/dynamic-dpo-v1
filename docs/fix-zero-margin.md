Pulled TRL v0.26.2 sources (per pyproject.toml), and re‑checked the earlier findings against the actual DPOTrainer/DPOConfig behavior. Here’s the updated review, grounded in TRL 0.26.2:

Updated Findings (TRL v0.26.2 grounded)

High: Your custom compute_loss bypasses TRL’s prompt+completion concatenation, but still masks using prompt_attention_mask. In TRL 0.26.2, the dataset is tokenized into separate prompt_input_ids and chosen/rejected_input_ids, and DPOTrainer concatenates them internally before masking (dpo_trainer.py lines ~666–730, ~1463–1520, ~1318–1324). In your code, prompt_attention_mask length can exceed chosen_input_ids length, so _build_labels_from_prompt masks the entire completion (-100), yielding log‑prob sums of 0 and thus margin=0. This is consistent with your “many zeros” symptom. dynamic_beta_dpo.py (lines 157-219) beta_dpo.py (lines 159-221)
Medium (downgraded): Not threading dataset.max_len into DPOConfig means TRL defaults apply (max_prompt_length=512, max_length=1024, truncation_mode=keep_end per dpo_config.py lines ~102–111, ~321–340). This can truncate long samples, but it shouldn’t create all‑zero margins unless completions are fully removed. Still worth noting as a secondary risk.
Medium (unchanged): Multi‑dataset rollout configs ignore the hf_name in config_rollout.yaml; load_rollout_dataset always uses its default dataset name. This can silently swap datasets and confound debugging. datasets.py (lines 112-145) config_rollout.yaml (lines 12-16)
Low (unchanged): apply_chat_template_to_dataset appends a Llama3 header when the prompt/template already contains one; this can bloat prompts and increase truncation likelihood. hh_dataset.py (lines 181-233)
If you want, I can outline non‑code checks to validate the top issue (e.g., quantify how often chosen_labels are all -100 in dpo_batch_debug.jsonl).



Example raw records:
prompt: "<|begin_of_text|><|start_header_id|>user<|end_header_id|> Where does it say that Humpty Dumpty is an egg in the nursery rhyme?<|eot_id|><|start_header_id|>assistant<|end_header_id|> I think Humpty Dumpty may actually be something like the “king’s” egg? But we know the rhyme didn’t appear in print until after the King’s Egg was already a well-known children’s nursery rhyme.<|eot_id|><|start_header_id|>user<|end_header_id|> you look like a egg<|eot_id|><|start_header_id|>assistant<|end_header_id|> "
chosen: "Humpty Dumpty's a lucky boy<|eot_id|>"
rejected: "I don’t look like an egg to me!<|eot_id|>"










Checklist

Verify the bug: inspect dpo_batch_debug.jsonl to confirm batches where prompt_attention_mask length exceeds chosen_input_ids length and labels become all -100.
Update both trainers to compute log‑probs on prompt+completion sequences (as TRL DPOTrainer does), and remove prompt‑masking on completion‑only inputs.
Wire truncation controls into DPOConfig (max_prompt_length, max_completion_length, max_length, truncation_mode) from config so prompts cannot swallow completions.
Re‑run a short slice and confirm margin samples are no longer all‑zero; spot‑check log‑prob sums.
Suggested Code Changes

