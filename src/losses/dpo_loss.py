import torch
import torch.nn.functional as F


# calculate the log probability
def compute_log_prob(logits, labels):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()

    loss_mask = labels != -100
    labels[labels == -100] = 0

    per_token_log_prob = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    return (per_token_log_prob * loss_mask).sum(-1)


# calculate the dpo loss
# we will do the loss.mean() in the training step
def dpo_loss(
    policy_chosen_log_prob,
    policy_rejected_log_prob,
    ref_chosen_log_prob,
    ref_rejected_log_prob,
    beta,
):
    chosen_log_prob = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_log_prob = policy_rejected_log_prob - ref_rejected_log_prob

    # compute the loss
    loss = -F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))

    chosen_rewards = (beta * chosen_log_prob).detach()
    rejected_rewards = (beta * rejected_log_prob).detach()

    return loss, chosen_rewards, rejected_rewards
