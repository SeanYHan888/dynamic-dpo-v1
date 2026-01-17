import torch


# compute the margin
def margin_compute(
    policy_chosen_log_prob,
    policy_rejected_log_prob,
    ref_chosen_log_prob,
    ref_rejected_log_prob,
):
    policy_diff = policy_chosen_log_prob - policy_rejected_log_prob
    ref_diff = ref_chosen_log_prob - ref_rejected_log_prob
    model_margin = (policy_diff - ref_diff).detach()

    return model_margin


# risk proportion test
def empirical_over_threshold_proportion(margins: torch.Tensor, threshold):
    return (margins >= threshold).float().mean().item()


def risk_test(p_hat, delta):
    return p_hat > delta
