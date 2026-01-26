"""Dynamic Beta DPO Trainer with risk-based beta adjustment."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from trl import DPOTrainer

from ..losses.dpo_loss import compute_log_prob, dpo_loss as dpo_loss_fn
from ..losses.margin import margin_compute, empirical_over_threshold_proportion, risk_test
from ..losses.beta_update import update_beta
from ..quantile.accumulator import WarmupQuantileAccumulator, EMAUpdate
from ..utils.logging import compute_and_log_model_margin


@dataclass
class DynamicBetaDPOConfig:
    """Configuration for dynamic beta DPO training.
    
    Attributes:
        delta: Risk threshold for proportion test.
        momentum: EMA momentum for threshold updates.
        beta_0: Initial beta value.
        alpha: Learning rate for beta updates.
        gamma: Scaling factor for tanh in beta update.
        beta_min: Minimum allowed beta value.
        beta_max: Maximum allowed beta value.
        warmup_steps: Number of steps for warmup phase.
        log_margins: Whether to log margins to disk.
        log_dir: Directory for margin logs.
        jsonl_name: Name of the JSONL summary file.
        save_per_rank: If True, each rank writes to separate folder.
        jsonl_sample_size: Max samples to store in JSONL (truncates if >0).
    """
    # Risk control
    delta: float = 0.1
    momentum: float = 0.05

    # Beta update
    beta_0: float = 0.1
    alpha: float = 0.005
    gamma: float = 2.0
    beta_min: float = 0.0
    beta_max: float = 2.0

    # Warmup
    warmup_steps: int = 120

    # Margin logging
    log_margins: bool = True
    log_dir: str = "logs/margins"
    jsonl_name: str = "margins.jsonl"
    save_per_rank: bool = False
    jsonl_sample_size: int = 32


class DynamicBetaDPOTrainer(DPOTrainer):
    """DPO Trainer with dynamic beta adjustment based on risk control.
    
    Features:
    - Warmup phase: Estimates threshold tau_0 by quantile over warmup margins
    - Training phase: Tau updated by EMA of per-batch quantiles
    - Risk test: p_hat = P(M >= tau) compared to delta
    - Beta update: beta <- beta * exp(alpha * tanh(gamma * u_k))
    """

    def __init__(self, *args, dynamic_cfg: DynamicBetaDPOConfig, **kwargs):
        super().__init__(*args, **kwargs)

        self.dynamic_cfg = dynamic_cfg

        # Beta adjustment state
        self.beta = float(dynamic_cfg.beta_0)
        self._warmup_done = False
        self._warmup_count = 0
        self.warmup_threshold = WarmupQuantileAccumulator(q=(1 - self.dynamic_cfg.delta))
        self._ema: Optional[EMAUpdate] = None
        self.tau = 0.0

        # Bookkeeping for logging
        self._last_stats: Dict[str, Any] = {}

        # Rank/process info (Accelerate)
        self._rank = self._get_rank()

        # Margin logging paths
        if self.dynamic_cfg.log_margins:
            base = self.dynamic_cfg.log_dir
            if self.dynamic_cfg.save_per_rank:
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

    def _maybe_log_margins(self, model_margin: torch.Tensor) -> None:
        """Log margins to disk if enabled."""
        if not self.dynamic_cfg.log_margins:
            return
        if (not self.dynamic_cfg.save_per_rank) and self._rank != 0:
            return

        epoch = getattr(self.state, "epoch", None)
        epoch_i = int(epoch) if epoch is not None else 0
        epoch_dir = os.path.join(self._margin_base_dir, f"epoch_{epoch_i:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        jsonl_path = os.path.join(epoch_dir, self.dynamic_cfg.jsonl_name)
        step = int(getattr(self.state, "global_step", 0))

        if self.dynamic_cfg.jsonl_sample_size and self.dynamic_cfg.jsonl_sample_size > 0:
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
                "npy": npy_path,
                "sample": [float(x) for x in m[: self.dynamic_cfg.jsonl_sample_size]],
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            compute_and_log_model_margin(
                model_margin=model_margin,
                epoch_dir=epoch_dir,
                epoch=epoch_i,
                step=step,
                jsonl_path=jsonl_path,
            )

    def _concat_prompt_completion(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_input_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> tuple:
        """Concatenate prompt and completion sequences.

        Returns:
            Tuple of (input_ids, attention_mask, prompt_seq_len) where prompt_seq_len
            is the sequence length of the prompt tensor (used to mask the entire prompt region).
        """
        # Get prompt sequence length (mask entire prompt region, including padding)
        prompt_seq_len = prompt_input_ids.size(1)

        # Concatenate along sequence dimension
        input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)

        return input_ids, attention_mask, prompt_seq_len

    def _build_labels_with_prompt_length(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_seq_len: int,
    ) -> torch.Tensor:
        """Build labels masking prompt tokens based on prompt sequence length."""
        labels = input_ids.clone()
        # Mask entire prompt region (same for all samples in batch)
        labels[:, :prompt_seq_len] = -100
        # Also mask padding tokens
        labels[attention_mask == 0] = -100
        return labels

    def _build_labels_from_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build token-level labels for logprob computation.

        Masks prompt tokens to -100 (if prompt_attention_mask is provided),
        and also masks padding tokens.
        """
        labels = input_ids.clone()
        if prompt_attention_mask is not None:
            prompt_len = prompt_attention_mask.long().sum(dim=1)
            for i, pl in enumerate(prompt_len.tolist()):
                pl = min(pl, labels.size(1))
                labels[i, :pl] = -100
        labels[attention_mask == 0] = -100
        return labels

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Compute DPO loss with dynamic beta adjustment."""
        # Check if we need to concatenate prompt + completion (TRL v0.26+ format)
        if "prompt_input_ids" in inputs:
            # Concatenate prompt + completion for chosen
            chosen_input_ids, chosen_attention_mask, chosen_prompt_len = self._concat_prompt_completion(
                prompt_input_ids=inputs["prompt_input_ids"],
                prompt_attention_mask=inputs["prompt_attention_mask"],
                completion_input_ids=inputs["chosen_input_ids"],
                completion_attention_mask=inputs["chosen_attention_mask"],
            )
            # Concatenate prompt + completion for rejected
            rejected_input_ids, rejected_attention_mask, rejected_prompt_len = self._concat_prompt_completion(
                prompt_input_ids=inputs["prompt_input_ids"],
                prompt_attention_mask=inputs["prompt_attention_mask"],
                completion_input_ids=inputs["rejected_input_ids"],
                completion_attention_mask=inputs["rejected_attention_mask"],
            )
            # Build labels with proper prompt masking
            chosen_labels = self._build_labels_with_prompt_length(
                chosen_input_ids, chosen_attention_mask, chosen_prompt_len
            )
            rejected_labels = self._build_labels_with_prompt_length(
                rejected_input_ids, rejected_attention_mask, rejected_prompt_len
            )
        else:
            # Use inputs directly (already concatenated format)
            chosen_input_ids = inputs["chosen_input_ids"]
            chosen_attention_mask = inputs["chosen_attention_mask"]
            rejected_input_ids = inputs["rejected_input_ids"]
            rejected_attention_mask = inputs["rejected_attention_mask"]

            # Build labels
            if "chosen_labels" in inputs and "rejected_labels" in inputs:
                chosen_labels = inputs["chosen_labels"]
                rejected_labels = inputs["rejected_labels"]
            else:
                prompt_attn = inputs.get("prompt_attention_mask", None)
                chosen_labels = self._build_labels_from_prompt(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    prompt_attention_mask=prompt_attn,
                )
                rejected_labels = self._build_labels_from_prompt(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                    prompt_attention_mask=prompt_attn,
                )

        # Policy model forward passes
        policy_chosen_out = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
        ).logits

        policy_rejected_out = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
        ).logits

        # Reference model forward passes (no gradients)
        with torch.no_grad():
            ref_chosen_out = self.ref_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
            ).logits

            ref_rejected_out = self.ref_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
            ).logits

        # Compute log probabilities
        policy_chosen_log_prob = compute_log_prob(logits=policy_chosen_out, labels=chosen_labels)
        policy_rejected_log_prob = compute_log_prob(logits=policy_rejected_out, labels=rejected_labels)
        ref_chosen_log_prob = compute_log_prob(logits=ref_chosen_out, labels=chosen_labels)
        ref_rejected_log_prob = compute_log_prob(logits=ref_rejected_out, labels=rejected_labels)

        # Compute DPO loss
        loss_ten, chosen_rewards, rejected_rewards = dpo_loss_fn(
            policy_chosen_log_prob=policy_chosen_log_prob,
            policy_rejected_log_prob=policy_rejected_log_prob,
            ref_chosen_log_prob=ref_chosen_log_prob,
            ref_rejected_log_prob=ref_rejected_log_prob,
            beta=float(self.beta),
        )
        loss = loss_ten.mean()

        # Compute margin
        model_margin = margin_compute(
            policy_chosen_log_prob=policy_chosen_log_prob,
            policy_rejected_log_prob=policy_rejected_log_prob,
            ref_chosen_log_prob=ref_chosen_log_prob,
            ref_rejected_log_prob=ref_rejected_log_prob,
        )

        # Dynamic beta adjustment
        with torch.no_grad():
            self._warmup_count += 1

            if (not self._warmup_done) and (self._warmup_count <= self.dynamic_cfg.warmup_steps):
                self.warmup_threshold.update(model_margin)

                if self._warmup_count == self.dynamic_cfg.warmup_steps:
                    tau0 = self.warmup_threshold.finalize()
                    self._ema = EMAUpdate(
                        tau_0=tau0,
                        q=1.0 - float(self.dynamic_cfg.delta),
                        momentum=float(self.dynamic_cfg.momentum),
                    )
                    self.tau = float(tau0)
                    self._warmup_done = True
            else:
                # Update tau by EMA of batch quantile
                if self._ema is not None:
                    self.tau = float(self._ema.update_tau(model_margin))

                # Compute p_hat + risk
                if self.tau is not None:
                    p_hat = empirical_over_threshold_proportion(model_margin, self.tau)
                else:
                    p_hat = 0.0

                fail = risk_test(p_hat, float(self.dynamic_cfg.delta))

                # Update beta
                beta_new, u_k, s_k, alpha = update_beta(
                    beta=float(self.beta),
                    p_hat=float(p_hat),
                    delta=float(self.dynamic_cfg.delta),
                    alpha=float(self.dynamic_cfg.alpha),
                    n=int(model_margin.numel()),
                    gamma=float(self.dynamic_cfg.gamma),
                    beta_min=float(self.dynamic_cfg.beta_min),
                    beta_max=float(self.dynamic_cfg.beta_max),
                )
                self.beta = float(beta_new)

                self._last_stats = {
                    "p_hat": float(p_hat),
                    "tau": float(self.tau) if self.tau is not None else None,
                    "fail": int(fail),
                    "u_k": float(u_k),
                    "s_k": float(s_k),
                    "alpha": float(alpha),
                }

            # Margin logging
            self._maybe_log_margins(model_margin)

            # Trainer logging
            log_payload = {
                "dpo/beta": float(self.beta),
                "dpo/margin_mean": float(model_margin.mean().item()),
                "dpo/loss": float(loss.detach().float().item()),
            }
            if self._warmup_done and self._last_stats:
                log_payload.update({
                    "risk/p_hat": self._last_stats.get("p_hat", 0.0),
                    "risk/tau": self._last_stats.get("tau", 0.0),
                    "risk/fail": self._last_stats.get("fail", 0),
                    "risk/u_k": self._last_stats.get("u_k", 0.0),
                    "risk/s_k": self._last_stats.get("s_k", 0.0),
                })

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
