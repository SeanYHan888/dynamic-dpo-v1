import torch
import torch.nn.functional as F


# beta dpo margin
def compute_beta_dpo_margin(
    policy_chosen_log_prob,
    policy_rejected_log_prob,
    ref_chosen_log_prob,
    ref_rejected_log_prob,
):
    chosen_gap = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_gap = policy_rejected_log_prob - ref_rejected_log_prob
    gap = (chosen_gap - rejected_gap).detach()
    return gap


# threshold compute
def compute_beta_dpo_threshold(M_0, sigma, m, gap, eps):
    gap = gap.detach()
    batch_mean = gap.mean()
    batch_std = gap.std(unbiased=False).clamp_min(eps)

    # init from first batch
    if (M_0 is None) or (sigma is None):
        M_0_new = batch_mean.detach()
        sigma_new = batch_std.detach()
    # EMA update
    else:
        M_0_new = (m * M_0 + (1.0 - m) * batch_mean).detach()
        sigma_new = (m * sigma + (1.0 - m) * batch_std).detach().clamp_min(eps)

    return M_0_new, sigma_new, batch_mean.detach(), batch_std.detach()


# data filtering
def beta_dpo_data_filter(gap, M_0, sigma, rho, eps):
    # compute Gaussian weights
    z = (gap - M_0) / (sigma + eps)
    weights = torch.exp(-0.5 * z.pow(2))

    B = gap.numel()
    k = max(1, int(B * rho))

    # numerical safety
    if weights.sum().item() <= 0:
        weights = torch.ones_like(weights)

    # sample without replacement
    selected_idx = torch.multinomial(weights, k, replacement=False)

    mask = torch.zeros_like(gap)
    mask[selected_idx] = 1.0

    return mask.detach(), selected_idx, weights.detach()


# beta update
def beta_dpo_beta_update(beta_0, alpha, gap_selected, threshold, min_beta):
    beta_used = gap_selected.new_tensor(beta_0) * (1.0 + alpha * (gap_selected - threshold))
    beta_used = beta_used.clamp(min=min_beta)
    return beta_used.detach()


# compute beta_dpo_loss
def beta_dpo_dpo_loss(
    policy_chosen_log_prob,
    policy_rejected_log_prob,
    ref_chosen_log_prob,
    ref_rejected_log_prob,
    beta_used,
):
    chosen_gap = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_gap = policy_rejected_log_prob - ref_rejected_log_prob

    # compute the loss
    loss = -F.logsigmoid(beta_used.detach() * (chosen_gap - rejected_gap))

    chosen_rewards = (beta_used * chosen_gap).detach()
    rejected_rewards = (beta_used * rejected_gap).detach()

    return loss, chosen_rewards, rejected_rewards
