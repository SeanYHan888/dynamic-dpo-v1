import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
from trl import DPOTrainer

from ..losses.beta_dpo import (
    beta_dpo_beta_update,
    beta_dpo_data_filter,
    beta_dpo_dpo_loss,
    compute_beta_dpo_margin,
    compute_beta_dpo_threshold,
)
from ..losses.beta_update import update_beta
from ..losses.dpo_loss import compute_log_prob, dpo_loss as dpo_loss_tensor
from ..losses.margin import (
    empirical_over_threshold_proportion,
    margin_compute,
    risk_test,
)
from ..quantile.accumulator import EMAUpdate, WarmupQuantileAccumulator
from ..utils.logging import compute_and_log_model_margin


@dataclass
class DynamicBetaDPOConfig:
    """
    Parameters for our training steps
    """

    # loss control
    mode_loss: str = "risk"

    # risk control
    delta: float = 0.1
    momentum: float = 0.05

    # beta update
    beta_0: float = 0.1
    alpha: float = 0.005
    gamma: float = 2.0
    beta_min: float = 0.0
    beta_max: float = 2.0

    # warmup
    warmup_steps: int = 120

    # margin logging
    log_margins: bool = True
    log_dir: str = "logs/margins"
    jsonl_name: str = "margins.jsonl"
    # if True, every rank writes its own folder; otherwise only rank0 writes
    save_per_rank: bool = False
    # if >0, truncate sample stored in jsonl to this length (recommended)
    jsonl_sample_size: int = 32

    # beta dpo
    bdpo_m: float = 0.9
    bdpo_rho: float = 0.8
    bdpo_a: float = 0.6
    bdpo_min_beta: float = 1e-3
    bdpo_eps: float = 1e-6


class DynamicBetaDPOTrainer(DPOTrainer):
    """
    Dynamic-beta DPO trainer:
      - warmup: estimate threshold tau0 by quantile over warmup margins
      - training: tau updated by EMA of per-batch quantiles
      - risk: p_hat = P(M >= tau) compared to delta
      - update range: uk = (p_hat - tau) / math.sqrt(delta * (1 - delta) / n)
      - beta update: beta <- beta * exp(alpha * tanh(gamma * u_k))

    """

    def __init__(self, *args, dynamic_cfg: DynamicBetaDPOConfig, **kwargs):
        super().__init__(*args, **kwargs)

        self.dynamic_cfg = dynamic_cfg

        # beta adjustment state
        self.beta = float(dynamic_cfg.beta_0)
        self._warmup_done = False
        self._warmup_count = 0
        self.warmup_threshold = WarmupQuantileAccumulator(q=1 - self.dynamic_cfg.delta)
        self._ema: EMAUpdate | None = None
        self.tau = 0

        # beta dpo
        self.bdpo_M0: torch.Tensor | None = None
        self.bdpo_sigma: torch.Tensor | None = None

        # bookkeeping for logging
        self._last_stats: Dict[str, Any] = {}

        # rank / process info (Accelerate)
        self._rank = self._get_rank()

        # margin logging paths
        if self.dynamic_cfg.log_margins:
            base = self.dynamic_cfg.log_dir
            if self.dynamic_cfg.save_per_rank:
                base = os.path.join(base, f"rank_{self._rank}")
            os.makedirs(base, exist_ok=True)
            self._margin_base_dir = base
        else:
            self._margin_base_dir = None

    def _get_rank(self) -> int:
        # TRL uses accelerate internally; prefer accelerator if available
        acc = getattr(self, "accelerator", None)
        if acc is not None:
            try:
                return int(acc.process_index)
            except Exception:
                pass
        return int(os.environ.get("RANK", "0"))

    def _maybe_log_margins(self, model_margin: torch.Tensor):
        if not self.dynamic_cfg.log_margins:
            return
        # only rank0 writes by default (much safer for IO)
        if (not self.dynamic_cfg.save_per_rank) and self._rank != 0:
            return

        # create epoch dir (we use state.epoch if available)
        epoch = getattr(self.state, "epoch", None)
        epoch_i = int(epoch) if epoch is not None else 0
        epoch_dir = os.path.join(self._margin_base_dir, f"epoch_{epoch_i:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # choose jsonl path
        jsonl_path = os.path.join(epoch_dir, self.dynamic_cfg.jsonl_name)

        # step index (global step)
        step = int(getattr(self.state, "global_step", 0))

        # write using your existing helper, but truncate jsonl sample if requested
        if self.dynamic_cfg.jsonl_sample_size and self.dynamic_cfg.jsonl_sample_size > 0:
            import numpy as np

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
                JSONL_PATH=jsonl_path,
            )

    def _build_labels_from_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None,
    ):
        """
        Build token-level labels for logprob computation.
        Mask prompt tokens to -100 (if prompt_attention_mask is provided), and also mask padding tokens.
        """
        labels = input_ids.clone()
        if prompt_attention_mask is not None:
            # prompt_attention_mask: [B, L] (or [B, Lp] depending on TRL)
            # We treat "1" positions as prompt tokens that should be masked out.
            prompt_len = prompt_attention_mask.long().sum(dim=1)  # [B]
            for i, pl in enumerate(prompt_len.tolist()):
                pl = min(pl, labels.size(1))
                labels[i, :pl] = -100

        labels[attention_mask == 0] = -100
        return labels

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # policy chosen and response output (needs grads)
        policy_chosen_out = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        ).logits

        policy_rejected_out = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        ).logits

        # ref model chosen and response output (without grads)
        with torch.no_grad():
            ref_chosen_out = self.ref_model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
            ).logits

            ref_rejected_out = self.ref_model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
            ).logits

        # compute log_prob
        # Case A: custom collate_fn already provides *_labels
        if "chosen_labels" in inputs and "rejected_labels" in inputs:
            chosen_labels = inputs["chosen_labels"]
            rejected_labels = inputs["rejected_labels"]

        # Case B: TRL default pipeline provides prompt_* and chosen/rejected ids but no labels
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

        policy_chosen_log_prob = compute_log_prob(logits=policy_chosen_out, labels=chosen_labels)
        policy_rejected_log_prob = compute_log_prob(logits=policy_rejected_out, labels=rejected_labels)
        ref_chosen_log_prob = compute_log_prob(logits=ref_chosen_out, labels=chosen_labels)
        ref_rejected_log_prob = compute_log_prob(logits=ref_rejected_out, labels=rejected_labels)

        mode = str(getattr(self.dynamic_cfg, "mode_loss", "risk")).lower()

        if mode == "beta_dpo":
            gap = compute_beta_dpo_margin(
                policy_chosen_log_prob,
                policy_rejected_log_prob,
                ref_chosen_log_prob,
                ref_rejected_log_prob,
            )

            self.bdpo_M0, self.bdpo_sigma, _, _ = compute_beta_dpo_threshold(
                M_0=self.bdpo_M0,
                sigma=self.bdpo_sigma,
                m=float(self.dynamic_cfg.bdpo_m),
                gap=gap,
                eps=float(self.dynamic_cfg.bdpo_eps),
            )

            mask, selected_idx, _ = beta_dpo_data_filter(
                gap=gap,
                M_0=self.bdpo_M0,
                sigma=self.bdpo_sigma,
                rho=float(self.dynamic_cfg.bdpo_rho),
                eps=float(self.dynamic_cfg.bdpo_eps),
            )

            gap_selected_mean = gap[selected_idx].mean()
            beta_used = beta_dpo_beta_update(
                beta_0=float(self.dynamic_cfg.beta_0),
                alpha=float(self.dynamic_cfg.bdpo_a),
                gap_selected=gap_selected_mean,
                threshold=self.bdpo_M0,
                min_beta=float(self.dynamic_cfg.bdpo_min_beta),
            )

            loss_ten, chosen_rewards, rejected_rewards = beta_dpo_dpo_loss(
                policy_chosen_log_prob,
                policy_rejected_log_prob,
                ref_chosen_log_prob,
                ref_rejected_log_prob,
                beta_used=beta_used,
            )

            denom = mask.sum().clamp_min(1.0)
            loss = (loss_ten * mask).sum() / denom
        else:
            # compute dpo loss tensor
            loss_ten, chosen_rewards, rejected_rewards = dpo_loss_tensor(
                policy_chosen_log_prob=policy_chosen_log_prob,
                policy_rejected_log_prob=policy_rejected_log_prob,
                ref_chosen_log_prob=ref_chosen_log_prob,
                ref_rejected_log_prob=ref_rejected_log_prob,
                beta=float(self.beta),
            )

            loss = loss_ten.mean()

        # margin
        model_margin = margin_compute(
            policy_chosen_log_prob=policy_chosen_log_prob,
            policy_rejected_log_prob=policy_rejected_log_prob,
            ref_chosen_log_prob=ref_chosen_log_prob,
            ref_rejected_log_prob=ref_rejected_log_prob,
        )

        # dynamic beta adjustment
        with torch.no_grad():
            if mode != "beta_dpo":
                self._warmup_count += 1

                if (not self._warmup_done) and (
                    self._warmup_count <= self.dynamic_cfg.warmup_steps
                ):
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
                    # update tau by EMA of batch quantile
                    if self._ema is not None:
                        self.tau = float(self._ema.update_tau(model_margin))

                    # compute p_hat + risk
                    if self.tau is not None:
                        p_hat = empirical_over_threshold_proportion(model_margin, self.tau)
                    else:
                        p_hat = 0.0

                    fail = risk_test(p_hat, float(self.dynamic_cfg.delta))

                    # update beta (always update; fail can be used to gate if you want)
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

            # margin logging (optional)
            self._maybe_log_margins(model_margin)

            # trainer logging (goes to wandb/tensorboard if enabled in args)
            log_payload = {
                "dpo/beta": float(self.beta),
                "dpo/margin_mean": float(model_margin.mean().item()),
                "dpo/loss": float(loss.detach().float().item()),
            }
            if mode == "beta_dpo":
                log_payload.update(
                    {
                        "beta_dpo/beta_used": float(beta_used.item()),
                        "beta_dpo/M0": float(self.bdpo_M0.item()),
                        "beta_dpo/sigma": float(self.bdpo_sigma.item()),
                        "beta_dpo/keep_ratio": float(mask.mean().item()),
                    }
                )
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
                # logging should never break training
                pass

        if return_outputs:
            return loss, {"chosen": chosen_rewards, "rejected": rejected_rewards}
        return loss

    # log eval loss
    def evaluate(self, eval_dataset=None, **kwargs):
        # call HF / TRL default evaluation loop
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        if self.args.report_to and "wandb" in self.args.report_to:
            try:
                self.log({"eval/loss": float(metrics["eval_loss"])})
            except Exception:
                pass

        return metrics
