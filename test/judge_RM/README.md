# Reward-Model (ArmoRM) Judge

This folder provides a deterministic alternative to GPT-based judging by scoring each response
with a reward model (default: `RLHFlow/ArmoRM-Llama3-8B-v0.1`) and selecting the higher score.

## Input format

All input files are JSON lists with entries like:

```json
{"instruction": "...", "output": "...", "generator": "..."}
```

This matches the outputs produced by:
- `test/gpt_judge_HH/generate_hh_output.py`
- `test/alpacaeval/generate_hh_output.py`

## Quickstart

1) Edit the config:
- `test/judge_RM/config/config_evaluation_rm.yaml`

2) Run:

```bash
python -m test.judge_RM.judge_outputs_rm --config test/judge_RM/config/config_evaluation_rm.yaml
```

Outputs:
- Per-pair judgments: `test/judge_RM/results/rm_judgments_<pair>.jsonl`
- Score caches: `test/judge_RM/scores/scores_<model_key>.jsonl`
- Summary: `test/judge_RM/results/summary.json`

## Tie rules

Configured under `rm_judge`:
- `tie_if_max_below`: if `max(score_a, score_b)` is below this, output `TIE`.
- `tie_margin`: if `abs(score_a - score_b)` is below this, output `TIE`.

