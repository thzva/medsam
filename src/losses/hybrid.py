"""Criterion that dispatches on the opinion's parameterisation.

The revision trains three families of model in one queue:
  * per-label Beta opinions        alpha shape (B, C, 2)  -> BetaEDLLoss
  * categorical Dirichlet opinions alpha shape (B, C)     -> EDLLoss
  * deterministic BCE baselines    alpha shape (B, C)     -> weighted BCE
Dispatching on the runtime shape rather than on a config flag keeps the queue
robust: a mode cannot silently be trained with the wrong objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .edl_loss import EDLLoss
from .beta_edl_loss import BetaEDLLoss


class HybridCriterion(nn.Module):
    def __init__(self, num_classes: int, loss_type: str = "log",
                 kl_weight: float = 0.01, annealing_epochs: int = 10,
                 class_weights: Optional[torch.Tensor] = None,
                 deterministic_bce: bool = False):
        super().__init__()
        self.deterministic_bce = deterministic_bce
        self.beta = BetaEDLLoss(num_classes, loss_type, kl_weight,
                                annealing_epochs, class_weights)
        self.dirichlet = EDLLoss(num_classes, loss_type, kl_weight, annealing_epochs)
        w = torch.ones(num_classes) if class_weights is None else class_weights.float()
        self.register_buffer("class_weights", w)

    def forward(self, opinion: Dict[str, torch.Tensor], target: torch.Tensor,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        alpha = opinion["alpha"]
        if alpha.dim() == 3:
            return self.beta(opinion, target, epoch)
        if self.deterministic_bce:
            # Inverse-frequency weighted BCE on per-label logits.
            # binary_cross_entropy is unsafe under autocast; the _with_logits
            # form is not, and it is numerically stable, so we always take the
            # logits path and force fp32.
            if "logits" not in opinion:
                raise KeyError("deterministic_bce requires 'logits' in the opinion dict")
            logits = opinion["logits"].float()
            bce = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none")
            loss = (bce * self.class_weights.unsqueeze(0)).sum(-1).mean()
            z = torch.zeros((), device=loss.device)
            return {"total": loss, "nll": loss, "kl": z}
        return self.dirichlet(opinion, target, epoch)
