"""Beta DPO Trainer with adaptive data filtering and beta adjustment.

This implements the Beta-DPO algorithm from:
"Beta-DPO: Direct Preference Optimization with Dynamic β"

Key features:
- Data filtering: Samples points based on Gaussian weights around threshold
- Adaptive beta: Per-batch beta adjustment based on gap statistics
- EMA thresholding: Running estimates of mean and std of gap
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from trl import DPOTrainer

from ..losses.dpo_loss import compute_log_prob
from ..losses.margin import margin_compute
from ..losses.beta_dpo import (
    compute_beta_dpo_margin,
    compute_beta_dpo_threshold,
    beta_dpo_data_filter,
    beta_dpo_beta_update,
    beta_dpo_loss,
)


@dataclass
class BetaDPOConfig:
    """Configuration for Beta DPO training.
    
    Attributes:
        beta_0: Base beta value.
        m: EMA momentum for threshold updates (0-1, higher = slower).
        rho: Fraction of data to keep after filtering (0-1).
        alpha: Scaling factor for beta adjustment.
        min_beta: Minimum allowed beta value.
        eps: Small constant for numerical stability.
        log_margins: Whether to log margins to disk.
        log_dir: Directory for margin logs.
        jsonl_name: Name of the JSONL summary file.
        save_per_rank: If True, each rank writes to separate folder.
        jsonl_sample_size: Max samples to store in JSONL.
    """
    # Beta DPO parameters
    beta_0: float = 0.1
    m: float = 0.9  # EMA momentum
    rho: float = 0.8  # Keep ratio for data filtering
    alpha: float = 0.6  # Beta adjustment factor
    min_beta: float = 1e-3
    eps: float = 1e-6

    # Margin logging
    log_margins: bool = True
    log_dir: str = "logs/beta_dpo_margins"
    jsonl_name: str = "margins.jsonl"
    save_per_rank: bool = False
    jsonl_sample_size: int = 32


class BetaDPOTrainer(DPOTrainer):
    """DPO Trainer implementing the Beta-DPO algorithm.
    
    Beta-DPO uses:
    1. EMA-based threshold estimation for gap statistics
    2. Gaussian-weighted data filtering to focus on informative samples
    3. Adaptive per-batch beta based on selected samples' gap
    
    This differs from Dynamic-Beta DPO which uses quantile-based risk control.
    """

    def __init__(self, *args, beta_dpo_cfg: BetaDPOConfig, **kwargs):
        """Initialize BetaDPOTrainer.
        
        Args:
            *args: Arguments passed to DPOTrainer.
            beta_dpo_cfg: Beta DPO configuration.
            **kwargs: Keyword arguments passed to DPOTrainer.
        """
        super().__init__(*args, **kwargs)

        self.beta_dpo_cfg = beta_dpo_cfg

        # Beta DPO state - initialized on first batch
        self.M_0: Optional[torch.Tensor] = None  # Threshold mean
        self.sigma: Optional[torch.Tensor] = None  # Threshold std

        # Bookkeeping for logging
        self._last_stats: Dict[str, Any] = {}

        # Rank/process info
        self._rank = self._get_rank()

        # Margin logging paths
        if self.beta_dpo_cfg.log_margins:
            base = self.beta_dpo_cfg.log_dir
            if self.beta_dpo_cfg.save_per_rank:
                base = os.path.join(base, f"rank_{self._rank}")
            os.makedirs(base, exist_ok=True)
            self._margin_base_dir = base
        else:
            self._margin_base_dir = None

    def _get_rank(self) -> int:
        """Get the process rank."""
        acc = getattr(self, "accelerator", None)
        if acc is not None:
            try:
                return int(acc.process_index)
            except Exception:
                pass
        return int(os.environ.get("RANK", "0"))

    def _maybe_log_margins(self, model_margin: torch.Tensor, beta_used: float) -> None:
        """Log margins and beta to disk if enabled."""
        if not self.beta_dpo_cfg.log_margins:
            return
        if (not self.beta_dpo_cfg.save_per_rank) and self._rank != 0:
            return

        epoch = getattr(self.state, "epoch", None)
        epoch_i = int(epoch) if epoch is not None else 0
        epoch_dir = os.path.join(self._margin_base_dir, f"epoch_{epoch_i:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        jsonl_path = os.path.join(epoch_dir, self.beta_dpo_cfg.jsonl_name)
        step = int(getattr(self.state, "global_step", 0))

        m = model_margin.detach().float().cpu().numpy()
        npy_path = os.path.join(epoch_dir, f"step_{step:05d}.npy")
        np.save(npy_path, m)

        p10, p50, p90 = np.percentile(m, [10, 50, 90])
        rec = {
            "epoch": int(epoch_i),
            "step": int(step),
            "batch_size": int(m.shape[0]),
            "mean": float(m.mean()),
            "std": float(m.std(ddof=0)),
            "min": float(m.min()),
            "p10": float(p10),
            "median": float(p50),
            "p90": float(p90),
            "max": float(m.max()),
            "pos_frac": float((m > 0).mean()),
            "beta_used": beta_used,
            "M_0": float(self.M_0.item()) if self.M_0 is not None else None,
            "sigma": float(self.sigma.item()) if self.sigma is not None else None,
            "npy": npy_path,
            "sample": [float(x) for x in m[: self.beta_dpo_cfg.jsonl_sample_size]],
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _build_labels_from_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build token-level labels for logprob computation."""
        labels = input_ids.clone()
        if prompt_attention_mask is not None:
            prompt_len = prompt_attention_mask.long().sum(dim=1)
            for i, pl in enumerate(prompt_len.tolist()):
                pl = min(pl, labels.size(1))
                labels[i, :pl] = -100
        labels[attention_mask == 0] = -100
        return labels

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Compute Beta DPO loss with adaptive filtering and beta."""
        # Policy model forward passes
        policy_chosen_out = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        ).logits

        policy_rejected_out = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        ).logits

        # Reference model forward passes (no gradients)
        with torch.no_grad():
            ref_chosen_out = self.ref_model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
            ).logits

            ref_rejected_out = self.ref_model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
            ).logits

        # Build labels
        if "chosen_labels" in inputs and "rejected_labels" in inputs:
            chosen_labels = inputs["chosen_labels"]
            rejected_labels = inputs["rejected_labels"]
        else:
            prompt_attn = inputs.get("prompt_attention_mask", None)
            chosen_labels = self._build_labels_from_prompt(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
                prompt_attention_mask=prompt_attn,
            )
            rejected_labels = self._build_labels_from_prompt(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
                prompt_attention_mask=prompt_attn,
            )

        # Compute log probabilities
        policy_chosen_log_prob = compute_log_prob(logits=policy_chosen_out, labels=chosen_labels)
        policy_rejected_log_prob = compute_log_prob(logits=policy_rejected_out, labels=rejected_labels)
        ref_chosen_log_prob = compute_log_prob(logits=ref_chosen_out, labels=chosen_labels)
        ref_rejected_log_prob = compute_log_prob(logits=ref_rejected_out, labels=rejected_labels)

        # Step 1: Compute gap
        gap = compute_beta_dpo_margin(
            policy_chosen_log_prob,
            policy_rejected_log_prob,
            ref_chosen_log_prob,
            ref_rejected_log_prob,
        )

        # Step 2: Update threshold (M_0, sigma) using EMA
        self.M_0, self.sigma, batch_mean, batch_std = compute_beta_dpo_threshold(
            M_0=self.M_0,
            sigma=self.sigma,
            m=float(self.beta_dpo_cfg.m),
            gap=gap,
            eps=float(self.beta_dpo_cfg.eps),
        )

        # Step 3: Data filtering - select samples based on Gaussian weights
        mask, selected_idx, weights = beta_dpo_data_filter(
            gap=gap,
            M_0=self.M_0,
            sigma=self.sigma,
            rho=float(self.beta_dpo_cfg.rho),
            eps=float(self.beta_dpo_cfg.eps),
        )

        # Step 4: Compute adaptive beta
        gap_selected_mean = gap[selected_idx].mean()
        beta_used = beta_dpo_beta_update(
            beta_0=float(self.beta_dpo_cfg.beta_0),
            alpha=float(self.beta_dpo_cfg.alpha),
            gap_selected=gap_selected_mean,
            threshold=self.M_0,
            min_beta=float(self.beta_dpo_cfg.min_beta),
        )

        # Step 5: Compute loss with mask
        loss_ten, chosen_rewards, rejected_rewards = beta_dpo_loss(
            policy_chosen_log_prob,
            policy_rejected_log_prob,
            ref_chosen_log_prob,
            ref_rejected_log_prob,
            beta_used=beta_used,
        )

        # Masked loss (only selected samples contribute)
        denom = mask.sum().clamp_min(1.0)
        loss = (loss_ten * mask).sum() / denom

        # Compute margin for logging
        model_margin = margin_compute(
            policy_chosen_log_prob=policy_chosen_log_prob,
            policy_rejected_log_prob=policy_rejected_log_prob,
            ref_chosen_log_prob=ref_chosen_log_prob,
            ref_rejected_log_prob=ref_rejected_log_prob,
        )

        # Logging
        with torch.no_grad():
            beta_val = float(beta_used.item())
            
            # Margin logging
            self._maybe_log_margins(model_margin, beta_val)

            # Trainer logging
            log_payload = {
                "beta_dpo/beta_used": beta_val,
                "beta_dpo/M0": float(self.M_0.item()),
                "beta_dpo/sigma": float(self.sigma.item()),
                "beta_dpo/keep_ratio": float(mask.mean().item()),
                "beta_dpo/gap_mean": float(gap.mean().item()),
                "beta_dpo/gap_selected_mean": float(gap_selected_mean.item()),
                "dpo/margin_mean": float(model_margin.mean().item()),
                "dpo/loss": float(loss.detach().float().item()),
            }

            try:
                self.log(log_payload)
            except Exception:
                pass

        if return_outputs:
            return loss, {"chosen": chosen_rewards, "rejected": rejected_rewards}
        return loss

    def evaluate(self, eval_dataset=None, **kwargs):
        """Evaluate with optional WandB logging."""
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        if self.args.report_to and "wandb" in self.args.report_to:
            try:
                self.log({"eval/loss": float(metrics["eval_loss"])})
            except Exception:
                pass
        return metrics
