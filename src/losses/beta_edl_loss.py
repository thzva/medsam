"""Multi-label Beta evidential loss.

Reviewer M1/R1-1: the task is multi-label, so the objective must decompose
over labels.  For label c with target y_c in {0,1} the binary frame gives a
one-hot vector t_c = [y_c, 1 - y_c] over {present, absent}, and the evidential
negative log-likelihood of the Beta opinion is

    L_c = sum_j t_cj * ( psi(S_c) - psi(alpha_cj) )

which is the evidential analogue of binary cross-entropy: taking the
expectation of -log p under Beta(alpha_c, beta_c) reproduces BCE exactly when
the Dirichlet strength S_c -> inf.  Per-class weights w_c inversely
proportional to prevalence are applied to L_c, so the stated objective
("inverse-frequency weighted binary cross-entropy") is literally what is
optimised -- this removes the Methods/Equation inconsistency (D2).

The KL term regularises evidence for the *wrong* outcome of each label
towards the vacuous prior Beta(1,1).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def beta_kl_divergence(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,1) ) per label.  alpha: (B, C, 2) -> (B, C)."""
    ones = torch.ones_like(alpha)
    S_a = alpha.sum(-1, keepdim=True)
    S_1 = ones.sum(-1, keepdim=True)
    ln_a = torch.lgamma(S_a) - torch.lgamma(alpha).sum(-1, keepdim=True)
    ln_1 = torch.lgamma(S_1) - torch.lgamma(ones).sum(-1, keepdim=True)
    dg = (alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_a))
    return (ln_a - ln_1 + dg.sum(-1, keepdim=True)).squeeze(-1)


def _targets_to_binary_frame(target: torch.Tensor) -> torch.Tensor:
    """(B, C) multi-hot -> (B, C, 2) one-hot per binary frame."""
    y = target.float().unsqueeze(-1)
    return torch.cat([y, 1.0 - y], dim=-1)


def beta_edl_log_loss(alpha, t, weight=None):
    S = alpha.sum(-1, keepdim=True)
    per_label = (t * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)   # (B, C)
    if weight is not None:
        per_label = per_label * weight.unsqueeze(0)
    return per_label


def beta_edl_mse_loss(alpha, t, weight=None):
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    var = alpha * (S - alpha) / (S * S * (S + 1.0))
    per_label = (((t - p) ** 2) + var).sum(-1)                            # (B, C)
    if weight is not None:
        per_label = per_label * weight.unsqueeze(0)
    return per_label


class BetaEDLLoss(nn.Module):
    """Inverse-frequency weighted multi-label evidential loss with KL annealing."""

    def __init__(
        self,
        num_classes: int,
        loss_type: str = "log",
        kl_weight: float = 0.01,
        annealing_epochs: int = 10,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        self.loss_fn = beta_edl_log_loss if loss_type == "log" else beta_edl_mse_loss
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        self.register_buffer("class_weights", class_weights.float())

    def forward(self, opinion: Dict[str, torch.Tensor], target: torch.Tensor,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        alpha = opinion["alpha"].float()               # (B, C, 2)
        if alpha.dim() != 3:
            raise ValueError(f"BetaEDLLoss expects (B,C,2) alpha, got {tuple(alpha.shape)}")
        t = _targets_to_binary_frame(target.to(alpha.device))

        nll_per_label = self.loss_fn(alpha, t, self.class_weights)
        nll = nll_per_label.sum(-1).mean()             # sum over labels, mean over batch

        # KL on evidence supporting the wrong outcome of each label
        alpha_tilde = t + (1.0 - t) * (alpha - 1.0) + 1.0
        kl_per_label = beta_kl_divergence(alpha_tilde) * self.class_weights.unsqueeze(0)
        kl = kl_per_label.sum(-1).mean()

        anneal = min(1.0, epoch / max(self.annealing_epochs, 1)) if self.annealing_epochs > 0 else 1.0
        total = nll + self.kl_weight * anneal * kl
        return {"total": total, "nll": nll, "kl": kl}


def inverse_frequency_weights(labels: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    """w_c proportional to 1/prevalence_c, normalised to mean 1 and clipped."""
    pos = labels.float().sum(0).clamp(min=1.0)
    w = labels.shape[0] / pos
    w = w / w.mean()
    return w.clamp(max=clip)


def labels_from_loader(loader) -> Optional[torch.Tensor]:
    """Extract the (N, C) label matrix from a DataLoader whose dataset may be
    wrapped in one or more torch.utils.data.Subset layers."""
    from torch.utils.data import Subset
    ds = loader.dataset
    idx = None
    while isinstance(ds, Subset):
        idx = ds.indices if idx is None else [ds.indices[i] for i in idx]
        ds = ds.dataset
    labels = getattr(ds, "labels", None)
    if labels is None:
        return None
    return labels if idx is None else labels[torch.as_tensor(idx)]
