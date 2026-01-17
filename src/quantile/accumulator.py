"""Quantile accumulator and EMA update for dynamic beta DPO."""

import torch


class WarmupQuantileAccumulator:
    """Accumulate margins during warmup and estimate initial quantile threshold (tau_0).
    
    We store warmup margins (optionally winsorized) and compute:
        tau_0 = quantile(margins, q)
    where typically q = 1 - delta, so that P(M >= tau) ~ delta.
    """
    
    def __init__(self, q: float):
        """Initialize the accumulator.
        
        Args:
            q: Quantile to compute (e.g., 0.9 for 90th percentile).
        """
        self.q = q
        self._buf: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, batch_margins: torch.Tensor) -> None:
        """Add batch margins to the accumulator.
        
        Args:
            batch_margins: Tensor of shape (batch_size,) for this warmup step.
        """
        t = batch_margins.detach().float().view(-1)
        if t.numel() == 0:
            return
        self._buf.append(t.cpu())

    def finalize(self) -> float:
        """Compute the final quantile threshold from accumulated margins.
        
        Returns:
            The quantile threshold tau_0.
        """
        if len(self._buf) == 0:
            return 0.0
        all_m = torch.cat(self._buf, dim=0)
        tau0 = torch.quantile(all_m, self.q).item()
        return float(tau0)


class EMAUpdate:
    """Exponential moving average update for threshold tau."""
    
    def __init__(self, tau_0: float, q: float, momentum: float):
        """Initialize the EMA updater.
        
        Args:
            tau_0: Initial threshold value.
            q: Quantile to compute on each batch.
            momentum: EMA momentum (lambda in the equation).
        """
        self.tau = tau_0
        self.q = q
        self.lam = momentum

    def update_tau(self, batch_margins: torch.Tensor) -> float:
        """Update threshold using EMA on batch quantile.
        
        Args:
            batch_margins: Tensor of margin values for the current batch.
            
        Returns:
            Updated tau value.
        """
        t = batch_margins.detach().float().view(-1)
        if t.numel() == 0:
            return self.tau
        batch_tau = torch.quantile(t, self.q).item()
        self.tau = (1.0 - self.lam) * self.tau + self.lam * batch_tau
        return self.tau
