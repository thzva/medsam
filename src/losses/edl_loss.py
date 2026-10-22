"""Evidential Deep Learning loss functions.

References:
    Sensoy et al., "Evidential Deep Learning to Quantify Classification
    Uncertainty", NeurIPS 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL divergence between Dir(α) and Dir(1, …, 1).

    Acts as a regulariser that pushes evidence toward zero for
    incorrect classes.

    Args:
        alpha: (B, K) Dirichlet concentration parameters.
        num_classes: K.
    Returns:
        kl: (B,) per-sample KL divergence.
    """
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_ones = ones.sum(dim=-1, keepdim=True)

    ln_alpha = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    ln_ones = torch.lgamma(S_ones) - torch.lgamma(ones).sum(dim=-1, keepdim=True)

    dg_term = (alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))

    kl = (ln_alpha - ln_ones + dg_term.sum(dim=-1, keepdim=True)).squeeze(-1)
    return kl


def edl_mse_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """EDL loss based on MSE of expected Dirichlet probabilities.

    Args:
        alpha: (B, K) Dirichlet parameters.
        target: (B, K) one-hot or multi-hot labels.
        num_classes: K.
        kl_weight: weight for KL regulariser.
    Returns:
        scalar loss.
    """
    S = alpha.sum(dim=-1, keepdim=True)
    p_hat = alpha / S  # expected probability

    # MSE between expected prob and target
    mse = (target - p_hat) ** 2
    # Variance of Dirichlet
    var = alpha * (S - alpha) / (S * S * (S + 1))
    loss = (mse + var).sum(dim=-1).mean()

    if kl_weight > 0:
        # Remove evidence for the true class before computing KL
        alpha_tilde = target + (1.0 - target) * (alpha - 1.0) + 1.0
        kl = kl_divergence(alpha_tilde, num_classes).mean()
        loss = loss + kl_weight * kl

    return loss


def edl_log_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """EDL loss based on negative log-likelihood of Dirichlet.

    Args:
        alpha: (B, K)
        target: (B, K) one-hot / multi-hot
        num_classes: K
        kl_weight: weight for KL regulariser
    Returns:
        scalar loss
    """
    S = alpha.sum(dim=-1, keepdim=True)
    nll = (target * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()

    if kl_weight > 0:
        alpha_tilde = target + (1.0 - target) * (alpha - 1.0) + 1.0
        kl = kl_divergence(alpha_tilde, num_classes).mean()
        nll = nll + kl_weight * kl

    return nll


class EDLLoss(nn.Module):
    """Unified EDL loss with annealing KL regulariser.

    During training the KL weight is linearly annealed from 0 to
    ``kl_weight`` over ``annealing_epochs`` epochs so the model can
    first learn to fit the data before being regularised.
    """

    def __init__(
        self,
        num_classes: int,
        loss_type: str = "log",       # "mse" or "log"
        kl_weight: float = 0.01,
        annealing_epochs: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        self.loss_fn = edl_log_loss if loss_type == "log" else edl_mse_loss

    def forward(
        self,
        opinion: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            opinion: dict with at least 'alpha' key (B, K).
            target: (B, K) one-hot / multi-hot  **or**  (B,) class indices.
            epoch: current epoch for KL annealing.
        Returns:
            dict with 'total', 'nll', 'kl' losses.
        """
        alpha = opinion["alpha"]

        # Convert class indices to one-hot if needed
        if target.dim() == 1 or (target.dim() == 2 and target.shape[-1] == 1):
            target = target.view(-1).long()
            target = F.one_hot(target, self.num_classes).float()

        # Annealed KL weight
        anneal = min(1.0, epoch / max(self.annealing_epochs, 1))
        current_kl_w = self.kl_weight * anneal

        # Main loss
        nll = self.loss_fn(alpha, target, self.num_classes, kl_weight=0.0)

        # KL regulariser
        alpha_tilde = target + (1.0 - target) * (alpha - 1.0) + 1.0
        kl = kl_divergence(alpha_tilde, self.num_classes).mean()

        total = nll + current_kl_w * kl

        return {"total": total, "nll": nll, "kl": kl}
