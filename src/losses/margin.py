"""Margin computation and risk testing utilities."""

import torch


def margin_compute(
    policy_chosen_log_prob: torch.Tensor,
    policy_rejected_log_prob: torch.Tensor,
    ref_chosen_log_prob: torch.Tensor,
    ref_rejected_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Compute the model margin.
    
    The margin measures the difference between policy and reference model
    preferences for chosen vs rejected responses.
    
    Args:
        policy_chosen_log_prob: Log probability of chosen response under policy.
        policy_rejected_log_prob: Log probability of rejected response under policy.
        ref_chosen_log_prob: Log probability of chosen response under reference.
        ref_rejected_log_prob: Log probability of rejected response under reference.
        
    Returns:
        Detached model margin tensor.
    """
    policy_diff = policy_chosen_log_prob - policy_rejected_log_prob
    ref_diff = ref_chosen_log_prob - ref_rejected_log_prob
    model_margin = (policy_diff - ref_diff).detach()
    return model_margin


def empirical_over_threshold_proportion(
    margins: torch.Tensor, threshold: float
) -> float:
    """Calculate the proportion of margins above a threshold.
    
    Args:
        margins: Tensor of margin values.
        threshold: Threshold value to compare against.
        
    Returns:
        Proportion of margins >= threshold.
    """
    return (margins >= threshold).float().mean().item()


def risk_test(p_hat: float, delta: float) -> bool:
    """Test if the empirical proportion exceeds the risk threshold.
    
    Args:
        p_hat: Empirical proportion of margins above threshold.
        delta: Risk threshold.
        
    Returns:
        True if p_hat > delta, indicating risk test passes.
    """
    return p_hat > delta
