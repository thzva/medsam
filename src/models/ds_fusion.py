"""Dempster-Shafer Evidence Fusion.

Implements Dempster's Rule of Combination for merging belief masses
from multiple evidential experts.
"""

import torch
import torch.nn as nn
from typing import Dict


def dempster_combination(
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Combine two Dirichlet opinions via Dempster's rule.

    Given two sets of Dirichlet parameters α₁, α₂ (each ≥ 1), we derive
    belief masses and uncertainty, then fuse them analytically.

    The closed-form for combining two Dirichlet-based subjective opinions
    (Jøsang, 2016) is:

        For each class k:
            b_fused_k = (b1_k * b2_k + b1_k * u2 + u1 * b2_k) / (1 - C)
        u_fused   = (u1 * u2) / (1 - C)

    where C = Σ_{i≠j} b1_i * b2_j  is the conflict mass.

    Args:
        alpha1: (B, K) Dirichlet parameters from expert 1.
        alpha2: (B, K) Dirichlet parameters from expert 2.

    Returns:
        dict with fused alpha, belief, uncertainty, evidence, S.
    """
    K = alpha1.shape[-1]

    # Derive belief and uncertainty from each expert
    S1 = alpha1.sum(dim=-1, keepdim=True)          # (B, 1)
    S2 = alpha2.sum(dim=-1, keepdim=True)
    e1 = alpha1 - 1.0                               # evidence
    e2 = alpha2 - 1.0
    b1 = e1 / S1                                     # (B, K)
    b2 = e2 / S2
    u1 = K / S1                                      # (B, 1)
    u2 = K / S2

    # Conflict mass  C = Σ_{i≠j} b1_i * b2_j
    # = (Σ b1_i)(Σ b2_j) - Σ_k b1_k * b2_k
    bb = (b1 * b2).sum(dim=-1, keepdim=True)         # Σ b1_k b2_k
    C = (b1.sum(dim=-1, keepdim=True) *
         b2.sum(dim=-1, keepdim=True)) - bb           # (B, 1)
    C = C.clamp(min=0.0, max=0.9999)                  # numerical safety

    norm = 1.0 - C                                     # (B, 1)

    # Fused belief
    b_fused = (b1 * b2 + b1 * u2 + u1 * b2) / norm   # (B, K)
    u_fused = (u1 * u2) / norm                         # (B, 1)

    # Back to Dirichlet parameters
    # b_k = e_k / S  and  u = K / S  =>  S = K / u
    S_fused = K / u_fused.clamp(min=1e-8)              # (B, 1)
    e_fused = b_fused * S_fused                        # (B, K)
    alpha_fused = e_fused + 1.0                        # (B, K)

    return {
        "evidence": e_fused,
        "alpha": alpha_fused,
        "belief": b_fused,
        "uncertainty": u_fused.squeeze(-1),
        "S": S_fused.squeeze(-1),
        "conflict": C.squeeze(-1),
    }


class DSFusion(nn.Module):
    """Wrapper module for Dempster-Shafer fusion of two evidential experts."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        opinion1: Dict[str, torch.Tensor],
        opinion2: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            opinion1: output dict from EvidentialHead (expert 1).
            opinion2: output dict from EvidentialHead (expert 2).
        Returns:
            Fused opinion dict.
        """
        return dempster_combination(opinion1["alpha"], opinion2["alpha"])


# ── Alternative fusion baselines (for ablation) ──────────────────────── #

class AverageFusion(nn.Module):
    """Simple probability averaging baseline."""

    def __init__(self):
        super().__init__()

    def forward(self, opinion1, opinion2):
        avg_alpha = (opinion1["alpha"] + opinion2["alpha"]) / 2.0
        K = avg_alpha.shape[-1]
        S = avg_alpha.sum(dim=-1, keepdim=True)
        e = avg_alpha - 1.0
        b = e / S
        u = K / S.squeeze(-1)
        return {"evidence": e, "alpha": avg_alpha, "belief": b,
                "uncertainty": u, "S": S.squeeze(-1)}


class GatingFusion(nn.Module):
    """Learnable gating fusion baseline."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, opinion1, opinion2, feat_global=None, feat_local=None):
        if feat_global is None or feat_local is None:
            # Fallback to simple average
            return AverageFusion()(opinion1, opinion2)
        cat = torch.cat([feat_global, feat_local], dim=-1)
        w = self.gate(cat)  # (B, 1)
        alpha = w * opinion1["alpha"] + (1 - w) * opinion2["alpha"]
        K = alpha.shape[-1]
        S = alpha.sum(dim=-1, keepdim=True)
        e = alpha - 1.0
        b = e / S
        u = K / S.squeeze(-1)
        return {"evidence": e, "alpha": alpha, "belief": b,
                "uncertainty": u, "S": S.squeeze(-1)}
