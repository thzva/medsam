"""Per-label Beta evidential heads for multi-label classification.

Addresses reviewer comment M1: a single categorical Dirichlet over C labels
forces sum_c p_c = 1, which is not a valid probability model when several
findings co-occur in the same radiograph.  Here each label c owns its own
binary frame of discernment Theta_c = {present, absent} with a Beta
(equivalently two-class Dirichlet) opinion:

    e_c = (e_c^+, e_c^-) >= 0            evidence
    alpha_c = e_c^+ + 1, beta_c = e_c^- + 1
    S_c = alpha_c + beta_c               Dirichlet strength
    b_c^+ = e_c^+ / S_c                  belief  (present)
    b_c^- = e_c^- / S_c                  disbelief (absent)
    u_c   = 2 / S_c                      vacuity, b^+ + b^- + u = 1
    p_c   = alpha_c / S_c = b_c^+ + u_c/2   projected probability (base rate 1/2)

Labels are independent frames, so p_c are unconstrained in [0,1] and
co-occurrence is representable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Numerical guards (all fusion/head math is done in float32; see D1 in
# revision/ISSUES.md -- the original FP16 Dempster combination produced NaN
# uncertainty on 22-36% of OOD samples, which were then silently dropped).
_U_MIN = 1e-6      # smallest representable vacuity  -> S <= 2e6
_U_MAX = 1.0
_EVIDENCE_MAX = 1e6


class BetaEvidentialHead(nn.Module):
    """Projects a feature vector to C independent Beta opinions."""

    def __init__(self, in_dim: int, num_classes: int, activation: str = "softplus"):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, 2 * num_classes)
        act_map = {"softplus": F.softplus, "relu": F.relu, "exp": torch.exp}
        if activation not in act_map:
            raise ValueError(f"Unknown activation: {activation}")
        self.act = act_map[activation]

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, in_dim)
        Returns:
            evidence   (B, C, 2)  non-negative evidence [positive, negative]
            alpha      (B, C, 2)  Beta parameters  (alpha_c, beta_c)
            belief     (B, C, 2)  [b^+, b^-]
            u          (B, C)     per-label vacuity
            prob       (B, C)     projected probability of "present"
            uncertainty(B,)       mean vacuity over labels (scalar OOD score)
            S          (B, C)
        """
        logits = self.fc(features).float()                     # fp32 from here on
        logits = logits.view(-1, self.num_classes, 2)          # (B, C, 2)
        evidence = self.act(logits).clamp(max=_EVIDENCE_MAX)   # (B, C, 2)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1)                                  # (B, C)
        belief = evidence / S.unsqueeze(-1)
        u = (2.0 / S).clamp(_U_MIN, _U_MAX)                    # (B, C)
        prob = belief[..., 0] + 0.5 * u                        # projected prob
        return {
            "evidence": evidence,
            "alpha": alpha,
            "belief": belief,
            "u": u,
            "prob": prob,
            "uncertainty": u.mean(dim=-1),
            "S": S,
        }


# ─────────────────────────── fusion rules ─────────────────────────────── #
# All operate on per-label binary opinions and return the same dict schema.

def _from_belief(b_pos: torch.Tensor, b_neg: torch.Tensor, u: torch.Tensor,
                 conflict: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Rebuild a full opinion dict from fused belief masses."""
    u = u.clamp(_U_MIN, _U_MAX)
    # Renormalise so that b+ + b- + u == 1 exactly (guards accumulated error).
    tot = b_pos + b_neg + u
    b_pos = b_pos / tot
    b_neg = b_neg / tot
    u = (u / tot).clamp(_U_MIN, _U_MAX)

    S = 2.0 / u                                     # (B, C)
    belief = torch.stack([b_pos, b_neg], dim=-1)    # (B, C, 2)
    evidence = belief * S.unsqueeze(-1)
    alpha = evidence + 1.0
    prob = b_pos + 0.5 * u
    return {
        "evidence": evidence,
        "alpha": alpha,
        "belief": belief,
        "u": u,
        "prob": prob,
        "uncertainty": u.mean(dim=-1),
        "S": S,
        "conflict": conflict,
    }


def _unpack(op: Dict[str, torch.Tensor]):
    b = op["belief"].float()
    return b[..., 0], b[..., 1], op["u"].float()


def discount(op: Dict[str, torch.Tensor], r: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Shafer's discounting: b~ = r*b, u~ = 1 - r + r*u  (r in [0,1]).

    Transfers (1-r) of every class-specific mass to ignorance, so an
    unreliable source can no longer manufacture consensus (R2 comment 3).
    """
    b_pos, b_neg, u = _unpack(op)
    r = r.to(b_pos.dtype)
    return _from_belief(r * b_pos, r * b_neg, 1.0 - r + r * u,
                        torch.zeros_like(u))


def beta_dempster(op1, op2) -> Dict[str, torch.Tensor]:
    """Dempster's rule on each binary frame (normalised, conflict removed)."""
    b1p, b1n, u1 = _unpack(op1)
    b2p, b2n, u2 = _unpack(op2)
    K = (b1p * b2n + b1n * b2p).clamp(0.0, 1.0 - 1e-6)   # conflict mass
    norm = 1.0 - K
    b_pos = (b1p * b2p + b1p * u2 + u1 * b2p) / norm
    b_neg = (b1n * b2n + b1n * u2 + u1 * b2n) / norm
    u = (u1 * u2) / norm
    return _from_belief(b_pos, b_neg, u, K)


def beta_yager(op1, op2) -> Dict[str, torch.Tensor]:
    """Yager's rule: conflict is assigned to ignorance instead of normalised
    away.  Conflict-robust alternative requested by R1 comment 4."""
    b1p, b1n, u1 = _unpack(op1)
    b2p, b2n, u2 = _unpack(op2)
    K = (b1p * b2n + b1n * b2p).clamp(0.0, 1.0)
    b_pos = b1p * b2p + b1p * u2 + u1 * b2p
    b_neg = b1n * b2n + b1n * u2 + u1 * b2n
    u = u1 * u2 + K
    return _from_belief(b_pos, b_neg, u, K)


def beta_cautious(op1, op2) -> Dict[str, torch.Tensor]:
    """Mean-BBA (Murphy) combination: average the two mass functions.

    Idempotent-ish and does not double-count shared evidence, which is the
    failure mode flagged for dependent sources (R2 comment 3 / R1 comment 4).
    """
    b1p, b1n, u1 = _unpack(op1)
    b2p, b2n, u2 = _unpack(op2)
    K = (b1p * b2n + b1n * b2p).clamp(0.0, 1.0)
    return _from_belief(0.5 * (b1p + b2p), 0.5 * (b1n + b2n),
                        0.5 * (u1 + u2), K)


def beta_average(op1, op2) -> Dict[str, torch.Tensor]:
    """Evidence averaging baseline (alpha-space mean)."""
    a = 0.5 * (op1["alpha"].float() + op2["alpha"].float())
    S = a.sum(dim=-1)
    e = a - 1.0
    b = e / S.unsqueeze(-1)
    u = (2.0 / S).clamp(_U_MIN, _U_MAX)
    return _from_belief(b[..., 0], b[..., 1], u, torch.zeros_like(u))


class BetaFusion(nn.Module):
    """Dispatch wrapper. `method` selects the combination rule."""

    _RULES = {
        "dempster_shafer": beta_dempster,
        "yager": beta_yager,
        "cautious": beta_cautious,
        "average": beta_average,
    }

    def __init__(self, method: str = "dempster_shafer", num_classes: int = 15,
                 embed_dim: int = 0, learn_discount: bool = False):
        super().__init__()
        self.method = method
        self.learn_discount = learn_discount
        if method == "gating":
            self.gate = nn.Sequential(nn.Linear(embed_dim, num_classes), nn.Sigmoid())
        elif method not in self._RULES:
            raise ValueError(f"Unknown fusion method: {method}")
        if learn_discount:
            # One reliability logit per (branch, label); r = sigmoid(.)
            self.discount_logit = nn.Parameter(torch.full((2, num_classes), 2.0))

    def forward(self, op1, op2, feat_global=None, feat_local=None,
                discount_r=None):
        """discount_r: optional (2, C) tensor of externally estimated
        reliabilities, used for post-hoc discounting at evaluation time."""
        if self.method == "gating":
            if feat_global is None or feat_local is None:
                return beta_average(op1, op2)
            w = self.gate(torch.cat([feat_global, feat_local], dim=-1)).float()
            a = (w.unsqueeze(-1) * op1["alpha"].float()
                 + (1 - w).unsqueeze(-1) * op2["alpha"].float())
            S = a.sum(dim=-1)
            b = (a - 1.0) / S.unsqueeze(-1)
            u = (2.0 / S).clamp(_U_MIN, _U_MAX)
            return _from_belief(b[..., 0], b[..., 1], u, torch.zeros_like(u))

        if discount_r is None and self.learn_discount:
            discount_r = torch.sigmoid(self.discount_logit)
        if discount_r is not None:
            r = discount_r.to(op1["u"].device).float()
            op1 = discount(op1, r[0].unsqueeze(0))
            op2 = discount(op2, r[1].unsqueeze(0))
        return self._RULES[self.method](op1, op2)
