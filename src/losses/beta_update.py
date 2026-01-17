import math


# beta update equation
def update_beta(beta, p_hat, delta, alpha, n, gamma, beta_min, beta_max):
    u_k = (p_hat - delta) / math.sqrt((delta * (1 - delta)) / n)
    s_k = math.tanh(gamma * u_k)
    beta_new = beta * math.exp(alpha * s_k)
    beta_new = max(beta_min, min(beta_new, beta_max))
    return beta_new, u_k, s_k, alpha
