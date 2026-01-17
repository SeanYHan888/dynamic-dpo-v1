"""Beta DPO loss functions and utilities.

This module implements the Beta DPO algorithm from:
"Beta-DPO: Direct Preference Optimization with Dynamic β"

The algorithm uses adaptive data filtering and per-batch beta adjustment
based on the gap between policy and reference model log probabilities.
"""

import torch
import torch.nn.functional as F


def compute_beta_dpo_margin(
    policy_chosen_log_prob: torch.Tensor,
    policy_rejected_log_prob: torch.Tensor,
    ref_chosen_log_prob: torch.Tensor,
    ref_rejected_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Compute the gap (margin) for Beta DPO.
    
    This computes the difference between the policy-reference gaps
    for chosen and rejected responses.
    
    Args:
        policy_chosen_log_prob: Log prob of chosen under policy.
        policy_rejected_log_prob: Log prob of rejected under policy.
        ref_chosen_log_prob: Log prob of chosen under reference.
        ref_rejected_log_prob: Log prob of rejected under reference.
        
    Returns:
        Detached gap tensor of shape (batch_size,).
    """
    chosen_gap = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_gap = policy_rejected_log_prob - ref_rejected_log_prob
    gap = (chosen_gap - rejected_gap).detach()
    return gap


def compute_beta_dpo_threshold(
    M_0: torch.Tensor | None,
    sigma: torch.Tensor | None,
    m: float,
    gap: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute and update the threshold for Beta DPO using EMA.
    
    Maintains running estimates of mean (M_0) and std (sigma) of the gap.
    
    Args:
        M_0: Current threshold mean (None for first batch).
        sigma: Current threshold std (None for first batch).
        m: EMA momentum factor (higher = slower update).
        gap: Current batch gap values.
        eps: Small constant for numerical stability.
        
    Returns:
        Tuple of (new_M_0, new_sigma, batch_mean, batch_std).
    """
    gap = gap.detach()
    batch_mean = gap.mean()
    batch_std = gap.std(unbiased=False).clamp_min(eps)

    # Initialize from first batch
    if (M_0 is None) or (sigma is None):
        M_0_new = batch_mean.detach()
        sigma_new = batch_std.detach()
    # EMA update
    else:
        M_0_new = (m * M_0 + (1.0 - m) * batch_mean).detach()
        sigma_new = (m * sigma + (1.0 - m) * batch_std).detach().clamp_min(eps)

    return M_0_new, sigma_new, batch_mean.detach(), batch_std.detach()


def beta_dpo_data_filter(
    gap: torch.Tensor,
    M_0: torch.Tensor,
    sigma: torch.Tensor,
    rho: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter data points using Gaussian-weighted sampling.
    
    Data points closer to the threshold M_0 are more likely to be selected.
    
    Args:
        gap: Batch gap values.
        M_0: Current threshold mean.
        sigma: Current threshold std.
        rho: Fraction of data to keep (0-1).
        eps: Small constant for numerical stability.
        
    Returns:
        Tuple of (mask, selected_indices, weights).
    """
    # Compute Gaussian weights
    z = (gap - M_0) / (sigma + eps)
    weights = torch.exp(-0.5 * z.pow(2))

    B = gap.numel()
    k = max(1, int(B * rho))

    # Numerical safety
    if weights.sum().item() <= 0:
        weights = torch.ones_like(weights)

    # Sample without replacement
    selected_idx = torch.multinomial(weights, k, replacement=False)

    mask = torch.zeros_like(gap)
    mask[selected_idx] = 1.0

    return mask.detach(), selected_idx, weights.detach()


def beta_dpo_beta_update(
    beta_0: float,
    alpha: float,
    gap_selected: torch.Tensor,
    threshold: torch.Tensor,
    min_beta: float = 1e-3,
) -> torch.Tensor:
    """Compute adaptive beta for Beta DPO.
    
    Beta is adjusted based on how far the selected samples' gap
    is from the threshold.
    
    Args:
        beta_0: Base beta value.
        alpha: Scaling factor for gap deviation.
        gap_selected: Mean gap of selected samples.
        threshold: Current threshold (M_0).
        min_beta: Minimum allowed beta value.
        
    Returns:
        Computed beta value (detached tensor).
    """
    beta_used = gap_selected.new_tensor(beta_0) * (1.0 + alpha * (gap_selected - threshold))
    beta_used = beta_used.clamp(min=min_beta)
    return beta_used.detach()


def beta_dpo_loss(
    policy_chosen_log_prob: torch.Tensor,
    policy_rejected_log_prob: torch.Tensor,
    ref_chosen_log_prob: torch.Tensor,
    ref_rejected_log_prob: torch.Tensor,
    beta_used: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Beta DPO loss.
    
    Similar to standard DPO loss but uses an adaptive beta.
    
    Args:
        policy_chosen_log_prob: Log prob of chosen under policy.
        policy_rejected_log_prob: Log prob of rejected under policy.
        ref_chosen_log_prob: Log prob of chosen under reference.
        ref_rejected_log_prob: Log prob of rejected under reference.
        beta_used: Adaptive beta value for this batch.
        
    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards).
    """
    chosen_gap = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_gap = policy_rejected_log_prob - ref_rejected_log_prob

    loss = -F.logsigmoid(beta_used.detach() * (chosen_gap - rejected_gap))

    chosen_rewards = (beta_used * chosen_gap).detach()
    rejected_rewards = (beta_used * rejected_gap).detach()

    return loss, chosen_rewards, rejected_rewards
