# AlpacaEval 2.0 Implementation Plan for SFT Model Evaluation

## Overview

This document outlines the implementation plan for evaluating the SFT model ([W-61/hh-llama32-1b-sft](https://huggingface.co/W-61/hh-llama32-1b-sft)) using AlpacaEval 2.0, a state-of-the-art automatic evaluation framework for instruction-following language models. And save every code into test folder

### What is AlpacaEval 2.0?

AlpacaEval 2.0 is an LLM-based automatic evaluation suite that:

- Uses GPT-4 Turbo as a judge to compare model outputs against reference outputs
- Evaluates on 805 diverse, open-ended prompts from the AlpacaEval dataset
- Provides **length-controlled win rates** to mitigate bias toward longer responses
- Has high correlation with human evaluations (validated against human annotations)
- Is cost-effective (~$10-15 per full evaluation run with GPT-4 Turbo)

### Evaluation Workflow

The AlpacaEval 2.0 evaluation process consists of two main steps:

1. **Inference**: Generate model responses for the 805 AlpacaEval prompts
2. **Judgment**: Use GPT-4 Turbo to compare your model's responses against reference outputs (default: GPT-4 Turbo) and calculate win rates

---

## Prerequisites

### 1. System Requirements

- **Python Version**: Python ≥ 3.10 (verify your current version)
- **GPU**: Not required for evaluation (inference can be done on CPU or GPU)
- **Disk Space**: ~2-5 GB for AlpacaEval dataset and generated outputs

### 2. API Access

> [!IMPORTANT]
> You will need an **OpenAI API key** with access to GPT-4 Turbo for the evaluation judge.

- Create an account at [OpenAI Platform](https://platform.openai.com/)
- Generate an API key from the API settings
- Set up billing (estimated cost: $10-15 per full evaluation)
- Export the key as an environment variable: `export OPENAI_API_KEY=<your_key>`

### 3. Dependencies

Install the required Python packages:

```bash
pip install alpaca-eval transformers torch accelerate
```

For the latest development version:

```bash
pip install git+https://github.com/tatsu-lab/alpaca_eval
```

---

## Implementation Plan

### Phase 1: Environment Setup

#### Step 1.1: Verify Python Version

```bash
python --version  # Should be ≥ 3.10
```

If your Python version is < 3.10, consider:

- Using `pyenv` to install Python 3.10+
- Creating a new conda environment with Python 3.10+

#### Step 1.2: Create Evaluation Directory Structure

```bash
mkdir -p evaluation/alpacaeval
mkdir -p evaluation/alpacaeval/outputs
mkdir -p evaluation/alpacaeval/results
```

#### Step 1.3: Install Dependencies

Create a new requirements file or add to existing `requirements.txt`:

```txt
alpaca-eval>=0.6.0
transformers>=4.40.0
torch>=2.0.0
accelerate>=0.27.0
datasets>=2.18.0
```

Install:

```bash
pip install -r requirements.txt
# OR using uv (if you're using it in your project)
uv pip install alpaca-eval transformers torch accelerate datasets
```

#### Step 1.4: Configure OpenAI API Key

```bash
export OPENAI_API_KEY=<your_openai_api_key>
```

Consider adding this to your shell profile (`.bashrc`, `.zshrc`) for persistence.

---

### Phase 2: Model Inference - Generate Outputs

Create a Python script to generate model outputs for the 805 AlpacaEval prompts.

#### Step 2.1: Create Inference Script

**File**: `evaluation/alpacaeval/generate_outputs.py`

```python
"""
Generate model outputs for AlpacaEval 2.0 evaluation.
This script loads the SFT model and generates responses for all AlpacaEval prompts.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse


def load_alpacaeval_dataset():
    """Load the AlpacaEval dataset (805 prompts)."""
    # AlpacaEval uses a specific evaluation set
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
    return dataset["eval"]


def generate_model_outputs(model_name: str, output_file: str, max_new_tokens: int = 512, 
                          batch_size: int = 1, device: str = "cuda"):
    """
    Generate outputs for AlpacaEval dataset using the specified model.
  
    Args:
        model_name: HuggingFace model ID or local path
        output_file: Path to save the outputs (JSON format)
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for inference
        device: Device to use for inference
    """
    print(f"Loading model: {model_name}")
  
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
  
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
  
    print("Loading AlpacaEval dataset...")
    dataset = load_alpacaeval_dataset()
  
    outputs = []
  
    print(f"Generating outputs for {len(dataset)} prompts...")
    for example in tqdm(dataset, desc="Generating"):
        instruction = example["instruction"]
      
        # Format the prompt using chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "user", "content": instruction}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to simple prompt formatting
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
      
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
      
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
      
        # Decode the output (remove the input prompt)
        output = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
      
        # AlpacaEval expects 'instruction' and 'output' keys
        outputs.append({
            "instruction": instruction,
            "output": output,
            "generator": model_name
        })
  
    # Save outputs to JSON
    print(f"Saving outputs to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
  
    print(f"✓ Generated {len(outputs)} outputs successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate AlpacaEval outputs")
    parser.add_argument(
        "--model_name",
        type=str,
        default="W-61/hh-llama32-1b-sft",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation/alpacaeval/outputs/model_outputs.json",
        help="Path to save the outputs"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
  
    args = parser.parse_args()
  
    generate_model_outputs(
        model_name=args.model_name,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )


if __name__ == "__main__":
    main()
```

#### Step 2.2: Run Inference

```bash
# Using GPU (recommended for faster inference)
python evaluation/alpacaeval/generate_outputs.py \
    --model_name W-61/hh-llama32-1b-sft \
    --output_file evaluation/alpacaeval/outputs/model_outputs.json \
    --max_new_tokens 512 \
    --device cuda

# Or using CPU (slower)
python evaluation/alpacaeval/generate_outputs.py \
    --model_name W-61/hh-llama32-1b-sft \
    --output_file evaluation/alpacaeval/outputs/model_outputs.json \
    --max_new_tokens 512 \
    --device cpu
```

**Expected Output**: A JSON file with 805 entries, each containing:

```json
{
  "instruction": "Give three tips for staying healthy.",
  "output": "Model's generated response...",
  "generator": "W-61/hh-llama32-1b-sft"
}
```

---

### Phase 3: AlpacaEval Evaluation

#### Step 3.1: Run AlpacaEval 2.0 Evaluation

Once you have the model outputs, run the AlpacaEval evaluation:

```bash
export OPENAI_API_KEY=<your_api_key>

alpaca_eval \
    --model_outputs evaluation/alpacaeval/outputs/model_outputs.json \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --reference_outputs gpt4_turbo \
    --output_path evaluation/alpacaeval/results \
    --name "hh-llama32-1b-sft"
```

**Key Parameters**:

- `--model_outputs`: Path to your generated outputs (JSON file)
- `--annotators_config`: Use `weighted_alpaca_eval_gpt4_turbo` (AlpacaEval 2.0 default)
- `--reference_outputs`: `gpt4_turbo` is the default for AlpacaEval 2.0
- `--output_path`: Directory to save evaluation results
- `--name`: Name for your model on the leaderboard

#### Step 3.2: Alternative - Direct Model Evaluation

AlpacaEval can also generate outputs directly from a HuggingFace model:

```bash
alpaca_eval evaluate_from_model \
    W-61/hh-llama32-1b-sft \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --output_path evaluation/alpacaeval/results \
    --max_instances 805
```

However, **the manual approach in Phase 2 gives you more control** over:

- Generation parameters (temperature, top_p, etc.)
- Prompt formatting
- Batch processing
- Debugging

---

### Phase 4: Analyze Results

#### Step 4.1: Understand the Metrics

AlpacaEval 2.0 provides several metrics:

- **win_rate**: Percentage of times your model's output is preferred over the reference
- **length_controlled_winrate** (LC win rate): Win rate adjusted for response length bias
- **avg_length**: Average length of model outputs
- **standard_error**: Statistical uncertainty of the win rate

The **length-controlled win rate** is the primary metric for AlpacaEval 2.0 as it correlates better with human preferences.

#### Step 4.2: View Results

Results will be saved to:

- `evaluation/alpacaeval/results/leaderboard.csv` - Comparison with other models
- `evaluation/alpacaeval/results/annotations.json` - Detailed per-example annotations

```bash
# View leaderboard
cat evaluation/alpacaeval/results/leaderboard.csv

# View detailed annotations (first 10 examples)
head -n 10 evaluation/alpacaeval/results/annotations.json
```

#### Step 4.3: Create Analysis Script

**File**: `evaluation/alpacaeval/analyze_results.py`

```python
"""
Analyze AlpacaEval 2.0 evaluation results.
"""

import json
import pandas as pd
import argparse


def analyze_results(results_dir: str):
    """Analyze and visualize AlpacaEval results."""
  
    # Load leaderboard
    leaderboard_path = f"{results_dir}/leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path)
  
    print("\n" + "="*60)
    print("ALPACAEVAL 2.0 EVALUATION RESULTS")
    print("="*60)
  
    # Find your model's row
    model_row = leaderboard[leaderboard['name'].str.contains('hh-llama32-1b-sft', na=False)]
  
    if not model_row.empty:
        print(f"\nModel: {model_row.iloc[0]['name']}")
        print(f"Win Rate: {model_row.iloc[0]['win_rate']:.2%}")
      
        if 'length_controlled_winrate' in model_row.columns:
            print(f"Length-Controlled Win Rate: {model_row.iloc[0]['length_controlled_winrate']:.2%}")
      
        if 'avg_length' in model_row.columns:
            print(f"Average Output Length: {model_row.iloc[0]['avg_length']:.0f} characters")
      
        print("\nTop 10 Models on Leaderboard:")
        print(leaderboard.head(10)[['name', 'win_rate']].to_string(index=False))
    else:
        print("\nModel not found in leaderboard. Showing all results:")
        print(leaderboard.to_string())
  
    # Load annotations for detailed analysis
    annotations_path = f"{results_dir}/annotations.json"
    try:
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
      
        print(f"\n\nTotal Evaluations: {len(annotations)}")
      
        # Count preferences
        preferences = [a.get('preference', 0) for a in annotations]
        wins = preferences.count(2)  # 2 means model wins
        losses = preferences.count(1)  # 1 means reference wins
        draws = preferences.count(0)  # 0 means draw
      
        print(f"Wins: {wins} ({wins/len(annotations)*100:.1f}%)")
        print(f"Losses: {losses} ({losses/len(annotations)*100:.1f}%)")
        print(f"Draws: {draws} ({draws/len(annotations)*100:.1f}%)")
      
    except FileNotFoundError:
        print(f"\nAnnotations file not found at {annotations_path}")
  
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze AlpacaEval results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/alpacaeval/results",
        help="Directory containing evaluation results"
    )
  
    args = parser.parse_args()
    analyze_results(args.results_dir)


if __name__ == "__main__":
    main()
```

Run the analysis:

```bash
python evaluation/alpacaeval/analyze_results.py \
    --results_dir evaluation/alpacaeval/results
```

---

## Project Integration

### Option 1: Add to Existing Training Pipeline

You can integrate AlpacaEval evaluation into your training workflow:

**File**: `train_sft.py` (modify)

Add an optional evaluation step after training:

```python
# At the end of train_sft.py, after trainer.save_model()

def run_alpacaeval_evaluation(model_path: str, output_dir: str):
    """Run AlpacaEval 2.0 evaluation after training."""
    import subprocess
  
    print("\n" + "="*60)
    print("Running AlpacaEval 2.0 Evaluation...")
    print("="*60 + "\n")
  
    # Check if OPENAI_API_KEY is set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Skipping AlpacaEval evaluation.")
        return
  
    eval_script = "evaluation/alpacaeval/generate_outputs.py"
    outputs_file = f"{output_dir}/alpacaeval_outputs.json"
    results_dir = f"{output_dir}/alpacaeval_results"
  
    # Step 1: Generate outputs
    subprocess.run([
        "python", eval_script,
        "--model_name", model_path,
        "--output_file", outputs_file
    ], check=True)
  
    # Step 2: Run evaluation
    subprocess.run([
        "alpaca_eval",
        "--model_outputs", outputs_file,
        "--annotators_config", "weighted_alpaca_eval_gpt4_turbo",
        "--output_path", results_dir,
        "--name", os.path.basename(model_path)
    ], check=True)
  
    print(f"\n✓ AlpacaEval results saved to: {results_dir}")


# Add to main() function:
if __name__ == "__main__":
    main()
  
    # Optional: Run AlpacaEval evaluation
    run_alpacaeval = input("\nRun AlpacaEval 2.0 evaluation? (y/n): ").lower().strip() == 'y'
    if run_alpacaeval:
        run_alpacaeval_evaluation(
            model_path=sft_cfg["save_dir"],
            output_dir=sft_cfg["save_dir"]
        )
```

### Option 2: Standalone Evaluation Script

Keep evaluation separate as a standalone workflow:

**File**: `evaluate_model.py`

```python
"""
Standalone script to evaluate any model with AlpacaEval 2.0.
"""

import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with AlpacaEval 2.0")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="evaluation/alpacaeval", help="Output directory")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation if outputs exist")
  
    args = parser.parse_args()
  
    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it: export OPENAI_API_KEY=<your_key>")
        return
  
    outputs_file = f"{args.output_dir}/outputs/model_outputs.json"
    results_dir = f"{args.output_dir}/results"
  
    # Step 1: Generate outputs
    if not args.skip_generation or not os.path.exists(outputs_file):
        print("Step 1: Generating model outputs...")
        subprocess.run([
            "python", "evaluation/alpacaeval/generate_outputs.py",
            "--model_name", args.model_name,
            "--output_file", outputs_file
        ], check=True)
    else:
        print(f"Skipping generation. Using existing outputs: {outputs_file}")
  
    # Step 2: Run evaluation
    print("\nStep 2: Running AlpacaEval 2.0 evaluation...")
    subprocess.run([
        "alpaca_eval",
        "--model_outputs", outputs_file,
        "--annotators_config", "weighted_alpaca_eval_gpt4_turbo",
        "--output_path", results_dir,
        "--name", args.model_name.split("/")[-1]
    ], check=True)
  
    # Step 3: Analyze results
    print("\nStep 3: Analyzing results...")
    subprocess.run([
        "python", "evaluation/alpacaeval/analyze_results.py",
        "--results_dir", results_dir
    ], check=True)


if __name__ == "__main__":
    main()
```

Usage:

```bash
python evaluate_model.py --model_name W-61/hh-llama32-1b-sft
```

---

## Cost Estimation

### OpenAI API Costs (GPT-4 Turbo)

AlpacaEval 2.0 uses GPT-4 Turbo for judging 805 examples:

- **Input tokens**: ~805 prompts × 2 responses × ~300 tokens avg = ~482,000 tokens
- **Output tokens**: ~805 outputs × ~200 tokens = ~161,000 tokens

**Pricing** (as of 2024):

- GPT-4 Turbo Input: $10/1M tokens
- GPT-4 Turbo Output: $30/1M tokens

**Estimated Cost**:

- Input: 482K × $10/1M = $4.82
- Output: 161K × $30/1M = $4.83
- **Total: ~$10-15 per full evaluation**

> [!TIP]
> To reduce costs during development/testing:
>
> - Use `--max_instances 100` flag to evaluate on a subset first
> - This will cost roughly $1-2 for 100 examples

---

## Timeline Estimation

| Phase             | Task                            | Estimated Time                                         |
| ----------------- | ------------------------------- | ------------------------------------------------------ |
| **Phase 1** | Environment setup, dependencies | 15-30 minutes                                          |
| **Phase 2** | Implement inference script      | 30-60 minutes                                          |
| **Phase 2** | Run inference (805 prompts)     | 30-90 minutes (GPU) or 3-6 hours (CPU)                 |
| **Phase 3** | Run AlpacaEval evaluation       | 20-40 minutes (API calls)                              |
| **Phase 4** | Results analysis                | 15-30 minutes                                          |
| **Total**   |                                 | **2-4 hours (GPU)** or **5-8 hours (CPU)** |

---

## Troubleshooting

### Common Issues

#### 1. Python Version Mismatch

```
ERROR: Python version must be >= 3.10
```

**Solution**: Use pyenv or conda to install Python 3.10+

#### 2. OpenAI API Key Not Found

```
ERROR: OpenAI API key not found
```

**Solution**:

```bash
export OPENAI_API_KEY=<your_key>
# Or add to ~/.bashrc or ~/.zshrc for persistence
```

#### 3. Out of Memory During Inference

```
CUDA out of memory error
```

**Solution**:

- Reduce batch size to 1
- Use CPU inference instead
- Use model quantization (e.g., `load_in_8bit=True`)

#### 4. AlpacaEval Dataset Not Found

```
Dataset not found: tatsu-lab/alpaca_eval
```

**Solution**:

```bash
# Manually download the dataset
python -c "from datasets import load_dataset; load_dataset('tatsu-lab/alpaca_eval', 'alpaca_eval')"
```

#### 5. Rate Limiting from OpenAI API

```
RateLimitError: Rate limit exceeded
```

**Solution**:

- Wait and retry
- Check your OpenAI account tier and rate limits
- Use `--chunksize` parameter to reduce concurrent requests

---

## Success Criteria

✅ **Completed Setup**:

- [ ] Python 3.10+ installed
- [ ] AlpacaEval package installed
- [ ] OpenAI API key configured
- [ ] Evaluation directory structure created

✅ **Generated Outputs**:

- [ ] 805 model outputs generated successfully
- [ ] Outputs saved in JSON format with correct schema
- [ ] Sample outputs look reasonable

✅ **Evaluation Complete**:

- [ ] AlpacaEval 2.0 evaluation completed
- [ ] Leaderboard results saved
- [ ] Win rate calculated
- [ ] Length-controlled win rate available

✅ **Results Analysis**:

- [ ] Results analyzed and interpreted
- [ ] Comparison with baseline models
- [ ] Identified model strengths and weaknesses

---

## Next Steps After Evaluation

### 1. Interpret Results

- **Win Rate > 50%**: Your model outperforms the reference (GPT-4 Turbo)
- **Win Rate 30-50%**: Competitive but has room for improvement
- **Win Rate < 30%**: Significant gap to state-of-the-art

### 2. Identify Improvement Areas

Review individual annotations to identify:

- Common failure patterns
- Categories where the model underperforms
- Length vs. quality trade-offs

### 3. Iterate on Training

Based on results, consider:

- Adjusting SFT hyperparameters
- Using higher quality training data
- Extending training duration
- Running DPO training for alignment

### 4. Re-evaluate

After making improvements:

- Re-run AlpacaEval on the new model
- Compare results to track progress
- Document improvements in a tracking sheet

---

## References

- [AlpacaEval GitHub Repository](https://github.com/tatsu-lab/alpaca_eval)
- [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- [AlpacaEval 2.0 Paper](https://arxiv.org/abs/2404.04475)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Your SFT Model](https://huggingface.co/W-61/hh-llama32-1b-sft)

---

## Appendix: Quick Reference Commands

```bash
# Environment setup
export OPENAI_API_KEY=<your_key>
pip install alpaca-eval transformers torch accelerate

# Generate outputs
python evaluation/alpacaeval/generate_outputs.py \
    --model_name W-61/hh-llama32-1b-sft \
    --output_file evaluation/alpacaeval/outputs/model_outputs.json

# Run evaluation
alpaca_eval \
    --model_outputs evaluation/alpacaeval/outputs/model_outputs.json \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --output_path evaluation/alpacaeval/results \
    --name hh-llama32-1b-sft

# Analyze results
python evaluation/alpacaeval/analyze_results.py \
    --results_dir evaluation/alpacaeval/results
```

---

**Document Version**: 1.0
**Created**: January 11, 2026
**Author**: Senior ML Engineer
**Model Under Evaluation**: [W-61/hh-llama32-1b-sft](https://huggingface.co/W-61/hh-llama32-1b-sft)
