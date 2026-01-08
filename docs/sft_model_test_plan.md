# SFT Model Evaluation Plan

This document outlines the workflow to evaluate the trained SFT model against the HH dataset and a baseline reference model.

## 1. Overview & Objectives

1. **Rollout**: Generate `n` candidates (e.g., 5) for each of 100 HH test samples using the SFT model.
2. **Reward Scoring**: Score all candidates for the first 50 prompts using ArmoRM. Calculate Mean and Variance over all `50 * n` scores.
3. **Head-to-Head**: Compare the **1st SFT candidate** vs. Base Model (Ref) using GPT-4 as a judge.

## 2. Configuration (`config_dpo.yaml`)

Add a new section `sft_evaluation` to the existing config:

```yaml
sft_evaluation:
  # Paths
  sft_model_path: "sft_model"            # Path to saved SFT checkpoint
  ref_model_name: "meta-llama/Llama-3.2-1B" # Original Base model
  
  # Generation Settings
  output_file: "eval_results/sft_eval_metrics.json"
  num_prompts: 100                  # Total prompts to process
  num_rollouts: 5                   # Candidates per prompt (K)
  batch_size: 8
  max_new_tokens: 512
  temperature: 1.0                  # Higher temp for diversity in rollouts
  top_p: 0.95
  
  # Evaluation Settings
  reward_model: "RLHFlow/ArmoRM-Llama3-8B-v0.1"
  judge_model: "gpt-4o"
  score_num_prompts: 50             # Score candidates for first 50 prompts
```

## 3. Implementation Steps

### Step 1: Rollout Generation (K Candidates)

**Script**: `scripts/test_sft_model.py`

**Process**:

1. **Data Loading**:
    * Load `Anthropic/hh-rlhf` (test split).
    * Take first `num_prompts` (100).
    * **Parsing**: Convert to message format (User/Assistant history) stripping the final Assistant turn.
2. **SFT Generation**:
    * Load `sft_model` + Tokenizer.
    * For each prompt, generate `num_rollouts` (e.g., 5) distinct responses using sampling (`do_sample=True`).
    * **Store**: list of candidates `[cand_1, cand_2, ..., cand_5]`.
3. **Reference Generation**:
    * Load `ref_model`.
    * Generate **1 response** per prompt (baseline for the Judge).
4. **Saving format**:

    ```json
    [
      {
        "prompt_id": 0,
        "prompt": "...",
        "model_responses": ["c1", "c2", "c3", "c4", "c5"], # SFT model responses
        "ref_response": "baseline_gen", # Reference model response
        "ground_truth": "original_text" # Original HH text
      },
      ... # 100 prompts total
    ]
    ```

### Step 2: Reward Model Scoring (First 50 Prompts)

**Goal**: Assess distribution of quality across rollouts.

**Process**:

1. Load `ArmoRM`.
2. Select the first 50 items from the results.
3. Flatten the list: `50 prompts * 5 candidates = 250 total pairs` of `(prompt, candidate)`.
4. **Score**: Run RM inference on all 250 pairs.
5. **Metrics**:
    * Calculate global **Mean** and **Variance** across all 250 scores.
    * (Optional) Calculate per-prompt mean/var to see stability.

### Step 3: LLM Judge (GPT-4)

**Goal**: SFT vs. Ref comparison.

**Process**:

1. **Selection**: Compare `model_responses[0]` (the first sample) against `ref_response`.
    * *Rationale*: Tests the model's "standard" performance path, or we could compare the "best of K" if we used the RM to select it (but plan says 1st response).
2. **Prompt**: Standard "Helpfulness/Harmlessness" pairwise judge prompt.
3. **Bias Control**: Randomize order (Response A vs B).
4. **Metrics**: Win Rate for SFT.

## 4. Code Structure

### `scripts/test_sft_model.py`

```python
def main():
    # ... setup ...
    
    # 1. Generate SFT Rollouts (K per prompt)
    # Output: results = [{ "model_responses": [...], ... }]
    results = generate_rollouts(sft_model, prompts, k=5)
    
    # 2. Generate Ref Responses (1 per prompt)
    raw_ref_gens = generate_rollouts(ref_model, prompts, k=1)
    merge_results(results, raw_ref_gens)
    
    # 3. RM Scoring
    # Limit to first 50 prompts for scoring
    subset_for_scoring = results[:50] 
    all_scores = []
    for item in subset_for_scoring:
        scores = score_candidates(rm_model, item['prompt'], item['model_responses'])
        all_scores.extend(scores)
        item['scores'] = scores # Store individual scores
        
    print(f"Global RM Mean: {np.mean(all_scores):.4f}")
    print(f"Global RM Variance: {np.var(all_scores):.4f}")
    
    # 4. GPT Judge
    # Compare  model_responses[0] vs ref_response
    judge_stats = run_judge(results)
    
    save_json(results)
```

## 5. Requirements

* `transformers`, `accelerate`, `torch`
* `openai`
* High VRAM availability if running ArmoRM (8B) alongside generation, or sequential loading.
