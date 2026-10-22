"""Evidential Uncertainty Head – maps features to Dirichlet parameters.

Based on Subjective Logic:
    evidence  e = f(features)          (non-negative)
    alpha     α = e + 1
    S         = Σ α_k
    belief    b_k = e_k / S
    uncertainty u = K / S
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class EvidentialHead(nn.Module):
    """Projects feature vector to Dirichlet concentration parameters α."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        activation: str = "softplus",
    ):
        """
        Args:
            in_dim: input feature dimension.
            num_classes: K, number of classes.
            activation: non-negative activation for evidence.
                        one of 'softplus', 'relu', 'exp'.
        """
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes)

        act_map = {
            "softplus": F.softplus,
            "relu": F.relu,
            "exp": torch.exp,
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation: {activation}")
        self.act = act_map[activation]

    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, in_dim)
        Returns:
            dict with keys:
                evidence  (B, K)  – non-negative evidence
                alpha     (B, K)  – Dirichlet concentration  α = e + 1
                belief    (B, K)  – belief mass  b_k = e_k / S
                uncertainty (B,)  – overall uncertainty  u = K / S
                S         (B,)    – Dirichlet strength
        """
        logits = self.fc(features)                    # (B, K)
        evidence = self.act(logits)                   # (B, K) ≥ 0
        alpha = evidence + 1.0                        # (B, K)
        S = alpha.sum(dim=-1, keepdim=True)           # (B, 1)
        belief = evidence / S                         # (B, K)
        uncertainty = self.num_classes / S.squeeze(-1) # (B,)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "belief": belief,
            "uncertainty": uncertainty,
            "S": S.squeeze(-1),
        }
