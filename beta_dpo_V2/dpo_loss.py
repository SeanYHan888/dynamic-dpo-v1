import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import json


# calculate the log probability
def compute_log_prob(logits, labels):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    per_token_log_prob = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

   
    return (per_token_log_prob * loss_mask).sum(-1)

# compute the margin 
def margin_compute(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob):
    # record the M that we need to calculate
    # we can detach it in training step
    policy_diff = policy_chosen_log_prob - policy_rejected_log_prob
    ref_diff =  ref_chosen_log_prob - ref_rejected_log_prob
    model_margin = (policy_diff - ref_diff).detach()

    return model_margin

# risk proportation test
def empirical_over_threshold_proportion(margins: torch.Tensor, threshold):
    return (margins >= threshold).float().mean().item()

def risk_test(p_hat, delta):
    return p_hat > delta

# beta update equation
def update_beta(beta, p_hat, delta, alpha, n, gamma, beta_min, beta_max):
    u_k = (p_hat - delta) / math.sqrt((delta * (1 - delta)) / n)
    s_k = math.tanh(gamma * u_k)
    beta_new = beta * math.exp(alpha * s_k)
    beta_new = max(beta_min, min(beta_new, beta_max))
    return beta_new, u_k, s_k, alpha

# compute and log model margin
def compute_and_log_model_margin(model_margin, epoch_dir, epoch, step, JSONL_PATH):
    # full array
    # using numpy to process, so only on cpu
    m = model_margin.detach().float().cpu().numpy()  
                
    # 1) save full margins as .npy (raw, lossless)
    # step: batch index
    npy_path = os.path.join(epoch_dir, f"step_{step:05d}.npy")
    np.save(npy_path, m)

    # 2) write a readable per-batch record to ONE jsonl file
    # summary stats
    # quantiles
    p10, p50, p90 = np.percentile(m, [10, 50, 90])
                
    record = {
        "epoch": int(epoch),
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
        "sample": [float(x) for x in m[:]],
    }

    # save in the jsonl file       
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# calculate the dpo loss
# we will do the loss.mean() in the training step
def dpo_loss(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta):
    
    chosen_log_prob = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_log_prob = policy_rejected_log_prob - ref_rejected_log_prob

    # compute the loss
    loss = - F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))

    chosen_rewards = (beta * chosen_log_prob).detach()
    rejected_rewards = (beta * rejected_log_prob).detach()

    return loss, chosen_rewards, rejected_rewards

# ---------------------------------beta dpo--------------------------------------------
# beta dpo margin
def compute_beta_dpo_margin(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob):
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
def beta_dpo_dpo_loss(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta_used):
    
    chosen_gap = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_gap = policy_rejected_log_prob - ref_rejected_log_prob

    # compute the loss
    loss = - F.logsigmoid(beta_used.detach() * (chosen_gap - rejected_gap))

    chosen_rewards = (beta_used * chosen_gap).detach()
    rejected_rewards = (beta_used * rejected_gap).detach()

    return loss, chosen_rewards, rejected_rewards