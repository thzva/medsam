"""Standard cross-entropy loss baseline for comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CrossEntropyBaseline(nn.Module):
    """CE / BCE loss wrapper that works with the same opinion-dict interface."""

    def __init__(self, num_classes: int, multi_label: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.multi_label = multi_label

    def forward(
        self,
        opinion: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Uses the belief vector as a proxy for class probabilities.
        """
        # Use belief as logits (before softmax)
        alpha = opinion["alpha"]
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S  # expected Dirichlet probability

        if self.multi_label:
            # Convert class indices to one-hot if needed
            if target.dim() == 1:
                target = F.one_hot(target, self.num_classes).float()
            loss = F.binary_cross_entropy(probs, target.float())
        else:
            if target.dim() == 2:
                target = target.argmax(dim=-1)
            loss = F.cross_entropy(probs, target)

        return {"total": loss, "nll": loss, "kl": torch.tensor(0.0, device=loss.device)}
