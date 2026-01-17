"""Beta update equations for dynamic DPO."""

import math


def update_beta(
    beta: float,
    p_hat: float,
    delta: float,
    alpha: float,
    n: int,
    gamma: float,
    beta_min: float,
    beta_max: float,
) -> tuple[float, float, float, float]:
    """Update beta using the risk-based update equation.
    
    Args:
        beta: Current beta value.
        p_hat: Empirical proportion of margins above threshold.
        delta: Risk threshold.
        alpha: Learning rate for beta updates.
        n: Sample size (batch size).
        gamma: Scaling factor for tanh.
        beta_min: Minimum allowed beta value.
        beta_max: Maximum allowed beta value.
        
    Returns:
        Tuple of (new_beta, u_k, s_k, alpha).
    """
    u_k = (p_hat - delta) / math.sqrt((delta * (1 - delta)) / n)
    s_k = math.tanh(gamma * u_k)
    beta_new = beta * math.exp(alpha * s_k)
    beta_new = max(beta_min, min(beta_new, beta_max))
    return beta_new, u_k, s_k, alpha
