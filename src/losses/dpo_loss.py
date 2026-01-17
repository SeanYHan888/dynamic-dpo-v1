"""Core DPO loss functions and log probability computation."""

import torch
import torch.nn.functional as F


def compute_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate the log probability of labels given logits.
    
    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        labels: Token labels of shape (batch, seq_len). Use -100 for masked tokens.
        
    Returns:
        Sum of log probabilities per sequence, shape (batch,).
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    per_token_log_prob = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    return (per_token_log_prob * loss_mask).sum(-1)


def dpo_loss(
    policy_chosen_log_prob: torch.Tensor,
    policy_rejected_log_prob: torch.Tensor,
    ref_chosen_log_prob: torch.Tensor,
    ref_rejected_log_prob: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the DPO loss.
    
    The loss.mean() should be applied in the training step.
    
    Args:
        policy_chosen_log_prob: Log probability of chosen response under policy.
        policy_rejected_log_prob: Log probability of rejected response under policy.
        ref_chosen_log_prob: Log probability of chosen response under reference.
        ref_rejected_log_prob: Log probability of rejected response under reference.
        beta: Temperature parameter for DPO.
        
    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards).
    """
    chosen_log_prob = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_log_prob = policy_rejected_log_prob - ref_rejected_log_prob

    loss = -F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))

    chosen_rewards = (beta * chosen_log_prob).detach()
    rejected_rewards = (beta * rejected_log_prob).detach()

    return loss, chosen_rewards, rejected_rewards
