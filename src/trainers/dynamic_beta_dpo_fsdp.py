"""Dynamic Beta DPO Trainer with risk-based beta adjustment, FSDP-compatible."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from trl import DPOTrainer

from ..losses.dpo_loss import compute_log_prob, dpo_loss as dpo_loss_fn
from ..losses.margin import (
    margin_compute,
    empirical_over_threshold_proportion,
    risk_test,
)
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


class DynamicBetaDPOTrainerFSDP(DPOTrainer):
    """DPO Trainer with dynamic beta adjustment based on risk control.

    Features:
    - FSDP-compatible state synchronization for beta and tau.
    - Warmup phase: Estimates threshold tau_0 by quantile over warmup margins
    - Training phase: Tau updated by EMA of per-batch quantiles
    - Risk test: p_hat = P(M >= tau) compared to delta
    - Beta update: beta <- beta * exp(alpha * tanh(gamma * u_k))
    """

    def __init__(self, *args, dynamic_cfg: DynamicBetaDPOConfig, **kwargs):
        super().__init__(*args, **kwargs)

        self.dynamic_cfg = dynamic_cfg

        # Beta adjustment state as tensors (for FSDP broadcast)
        self.beta = torch.tensor(dynamic_cfg.beta_0, device=self.accelerator.device)
        self.tau = torch.tensor(0.0, device=self.accelerator.device)
        self._warmup_done_tensor = torch.tensor(
            [0], device=self.accelerator.device
        )  # 0=False, 1=True
        self._warmup_count = 0

        # State managed only on main process
        if self.accelerator.is_main_process:
            self.warmup_threshold = WarmupQuantileAccumulator(
                q=(1 - self.dynamic_cfg.delta)
            )
            self._ema: Optional[EMAUpdate] = None
        else:
            self.warmup_threshold = None
            self._ema = None

        self._last_stats: Dict[str, Any] = {}
        self._rank = self.accelerator.process_index

        # Margin logging paths
        if self.dynamic_cfg.log_margins:
            base = self.dynamic_cfg.log_dir
            if self.dynamic_cfg.save_per_rank:
                base = os.path.join(base, f"rank_{self._rank}")
            os.makedirs(base, exist_ok=True)
            self._margin_base_dir = base
        else:
            self._margin_base_dir = None

    @property
    def _warmup_done(self) -> bool:
        return bool(self._warmup_done_tensor.item())

    @_warmup_done.setter
    def _warmup_done(self, value: bool) -> None:
        self._warmup_done_tensor.fill_(1 if value else 0)

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

    def _maybe_log_margins(self, model_margin: torch.Tensor) -> None:
        """Log margins to disk if enabled (main process only for gathered margins)."""
        if not self.dynamic_cfg.log_margins:
            return
        if not self.accelerator.is_main_process:
            return

        epoch = getattr(self.state, "epoch", None)
        epoch_i = int(epoch) if epoch is not None else 0
        epoch_dir = os.path.join(self._margin_base_dir, f"epoch_{epoch_i:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        jsonl_path = os.path.join(epoch_dir, self.dynamic_cfg.jsonl_name)
        step = int(getattr(self.state, "global_step", 0))

        m = model_margin.detach().float().cpu().numpy()

        if (
            self.dynamic_cfg.jsonl_sample_size
            and self.dynamic_cfg.jsonl_sample_size > 0
        ):
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

    def _sync_distributed_state(self) -> None:
        """Broadcast beta, tau, and warmup_done from rank 0 to all ranks."""
        if not dist.is_initialized():
            return
        dist.broadcast(self.beta, src=0)
        dist.broadcast(self.tau, src=0)
        dist.broadcast(self._warmup_done_tensor, src=0)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Compute DPO loss with dynamic beta adjustment, synchronized for FSDP."""

        # === Policy model forward passes ===
        policy_chosen_out = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        ).logits

        policy_rejected_out = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        ).logits

        # === Reference model forward passes (no gradients) ===
        with torch.no_grad():
            ref_chosen_out = self.ref_model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
            ).logits

            ref_rejected_out = self.ref_model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
            ).logits

        # === Build labels ===
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

        # === Compute log probabilities ===
        policy_chosen_logps = compute_log_prob(
            logits=policy_chosen_out, labels=chosen_labels
        )
        policy_rejected_logps = compute_log_prob(
            logits=policy_rejected_out, labels=rejected_labels
        )
        ref_chosen_logps = compute_log_prob(logits=ref_chosen_out, labels=chosen_labels)
        ref_rejected_logps = compute_log_prob(
            logits=ref_rejected_out, labels=rejected_labels
        )

        # === Compute DPO loss ===
        loss_ten, chosen_rewards, rejected_rewards = dpo_loss_fn(
            policy_chosen_log_prob=policy_chosen_logps,
            policy_rejected_log_prob=policy_rejected_logps,
            ref_chosen_log_prob=ref_chosen_logps,
            ref_rejected_log_prob=ref_rejected_logps,
            beta=self.beta.item(),
        )
        loss = loss_ten.mean()

        # === Compute margin on local batch ===
        model_margin = margin_compute(
            policy_chosen_log_prob=policy_chosen_logps,
            policy_rejected_log_prob=policy_rejected_logps,
            ref_chosen_log_prob=ref_chosen_logps,
            ref_rejected_log_prob=ref_rejected_logps,
        )

        # === FSDP: Gather margins and update state on main process ===
        with torch.no_grad():
            # Gather all margins to main process
            if dist.is_initialized():
                gathered_margins = self.accelerator.gather_for_metrics(model_margin)
            else:
                gathered_margins = model_margin

            self._warmup_count += 1

            # State updates on main process only
            if self.accelerator.is_main_process:
                if (not self._warmup_done) and (
                    self._warmup_count <= self.dynamic_cfg.warmup_steps
                ):
                    self.warmup_threshold.update(gathered_margins)

                    if self._warmup_count == self.dynamic_cfg.warmup_steps:
                        tau0 = self.warmup_threshold.finalize()
                        self._ema = EMAUpdate(
                            tau_0=tau0,
                            q=1.0 - float(self.dynamic_cfg.delta),
                            momentum=float(self.dynamic_cfg.momentum),
                        )
                        self.tau.fill_(float(tau0))
                        self._warmup_done = True
                else:
                    # Update tau by EMA
                    if self._ema is not None:
                        new_tau = self._ema.update_tau(gathered_margins)
                        self.tau.fill_(float(new_tau))

                    # Compute risk metrics
                    p_hat = empirical_over_threshold_proportion(
                        gathered_margins, self.tau.item()
                    )
                    fail = risk_test(p_hat, float(self.dynamic_cfg.delta))

                    # Update beta
                    beta_new, u_k, s_k, alpha = update_beta(
                        beta=self.beta.item(),
                        p_hat=float(p_hat),
                        delta=float(self.dynamic_cfg.delta),
                        alpha=float(self.dynamic_cfg.alpha),
                        n=int(gathered_margins.numel()),
                        gamma=float(self.dynamic_cfg.gamma),
                        beta_min=float(self.dynamic_cfg.beta_min),
                        beta_max=float(self.dynamic_cfg.beta_max),
                    )
                    self.beta.fill_(float(beta_new))

                    self._last_stats = {
                        "p_hat": float(p_hat),
                        "tau": self.tau.item(),
                        "fail": int(fail),
                        "u_k": float(u_k),
                        "s_k": float(s_k),
                        "alpha": float(alpha),
                    }

                # Log margins on main process
                self._maybe_log_margins(gathered_margins)

            # Broadcast updated state to all ranks
            self._sync_distributed_state()

            # Logging (self.log handles rank-specific logic)
            log_payload = {
                "dpo/beta": self.beta.item(),
                "dpo/margin_mean": float(gathered_margins.mean().item()),
                "dpo/loss": float(loss.detach().item()),
            }
            if self._warmup_done and self._last_stats:
                log_payload.update(
                    {
                        "risk/p_hat": self._last_stats.get("p_hat", 0.0),
                        "risk/tau": self._last_stats.get("tau", 0.0),
                        "risk/fail": self._last_stats.get("fail", 0),
                        "risk/u_k": self._last_stats.get("u_k", 0.0),
                        "risk/s_k": self._last_stats.get("s_k", 0.0),
                    }
                )

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
