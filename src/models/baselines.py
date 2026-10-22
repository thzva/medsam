"""Baseline models for comparison experiments.

Categories from the proposal:
  (A) Single-stream: ResNet-50 (CE), ViT (CE), DenseNet-121
  (B) Calibration-only: ResNet + Temperature Scaling, ViT + Dirichlet Cal
  (C) UQ baselines: MC Dropout, Deep Ensemble, EDL (single-scale)
  (D) Multi-scale: Global+Local + Avg prob / logit / Gating
  (E) Ours: TMS-EDL (in tms_edl.py)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional

from .evidential_head import EvidentialHead

logger = logging.getLogger(__name__)


def _safe_create(backbone, pretrained, **kwargs):
    try:
        return timm.create_model(backbone, pretrained=pretrained, **kwargs)
    except Exception as e:
        logger.warning(f"Pretrained weights unavailable for {backbone}: {e}")
        logger.warning("Using random initialisation.")
        return timm.create_model(backbone, pretrained=False, **kwargs)


# ═══════════════════════════════════════════════════════════════════════ #
# (A) Single-stream baselines
# ═══════════════════════════════════════════════════════════════════════ #

class SingleStreamModel(nn.Module):
    """Single backbone + classification head (CE/BCE loss)."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = _safe_create(backbone, pretrained, num_classes=0)
        feat_dim = self.encoder.num_features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, hr=None) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        feat = self.dropout(feat)
        logits = self.head(feat).float()
        # Deterministic multi-label baseline: one independent Bernoulli per
        # label.  The previous wrapper normalised alpha over the label axis,
        # which imposes exactly the mutually-exclusive assumption reviewer M1
        # objects to -- using it here would cripple the baseline with the very
        # flaw the paper is correcting.
        prob = torch.sigmoid(logits)
        p = prob.clamp(1e-6, 1 - 1e-6)
        # Uncertainty score = mean binary predictive entropy over labels.
        ent = -(p * p.log() + (1 - p) * (1 - p).log())
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        opinion = {"evidence": evidence, "alpha": alpha, "logits": logits,
                   "prob": prob, "uncertainty": ent.mean(dim=-1), "S": S.squeeze(-1)}
        return {"global": opinion, "local": opinion, "fused": opinion,
                "attn_weights": None, "logits": logits}


# ═══════════════════════════════════════════════════════════════════════ #
# (B) Temperature Scaling
# ═══════════════════════════════════════════════════════════════════════ #

class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling wrapper."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.base_model(x)
        if "logits" in out:
            logits = out["logits"] / self.temperature
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            S = alpha.sum(dim=-1, keepdim=True)
            belief = evidence / S
            K = alpha.shape[-1]
            uncertainty = K / S.squeeze(-1)
            scaled = {"evidence": evidence, "alpha": alpha, "belief": belief,
                      "uncertainty": uncertainty, "S": S.squeeze(-1)}
            out["fused"] = scaled
        return out

    def calibrate(self, val_loader, device):
        """Optimize temperature on validation set using NLL."""
        self.base_model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                out = self.base_model(images)
                logits_list.append(out["logits"])
                labels_list.append(labels)
        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list)

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def closure():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature
            if all_labels.dim() == 2:
                loss = F.binary_cross_entropy_with_logits(scaled, all_labels)
            else:
                loss = F.cross_entropy(scaled, all_labels)
            loss.backward()
            return loss
        optimizer.step(closure)
        print(f"Calibrated temperature: {self.temperature.item():.4f}")


# ═══════════════════════════════════════════════════════════════════════ #
# (C) MC Dropout
# ═══════════════════════════════════════════════════════════════════════ #

class MCDropoutModel(nn.Module):
    """Single-stream model with MC Dropout for uncertainty."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 15,
        dropout: float = 0.2,
        n_samples: int = 10,
    ):
        super().__init__()
        self.base = SingleStreamModel(backbone, pretrained, num_classes, dropout)
        self.n_samples = n_samples
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, hr=None) -> Dict[str, torch.Tensor]:
        if self.training:
            return self.base(x)

        # MC sampling at test time, over INDEPENDENT per-label Bernoullis.
        self.base.train()  # enable dropout
        samples = []
        for _ in range(self.n_samples):
            samples.append(torch.sigmoid(self.base(x)["logits"].float()))
        self.base.eval()

        stacked = torch.stack(samples, dim=0).clamp(1e-6, 1 - 1e-6)   # (T, B, C)
        mean_probs = stacked.mean(dim=0)                              # (B, C)
        bern_ent = lambda q: -(q * q.log() + (1 - q) * (1 - q).log())
        # BALD / mutual information: total predictive entropy minus the mean
        # aleatoric entropy.  This is the epistemic part, and it is the right
        # comparator for an evidential vacuity score (reviewer M9).
        total = bern_ent(mean_probs)                                  # (B, C)
        aleatoric = bern_ent(stacked).mean(dim=0)                     # (B, C)
        mutual_info = (total - aleatoric).clamp(min=0.0)

        logits = torch.log(mean_probs) - torch.log1p(-mean_probs)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        opinion = {"evidence": evidence, "alpha": alpha, "logits": logits,
                   "prob": mean_probs, "uncertainty": mutual_info.mean(dim=-1),
                   "predictive_entropy": total.mean(dim=-1),
                   "S": alpha.sum(dim=-1)}
        return {"global": opinion, "local": opinion, "fused": opinion,
                "attn_weights": None, "logits": logits}


# ═══════════════════════════════════════════════════════════════════════ #
# (C) EDL Single-Scale
# ═══════════════════════════════════════════════════════════════════════ #

class EDLSingleScale(nn.Module):
    """Single backbone + EDL head (no multi-scale, no fusion)."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 15,
        activation: str = "softplus",
    ):
        super().__init__()
        self.encoder = _safe_create(backbone, pretrained, num_classes=0)
        feat_dim = self.encoder.num_features
        self.edl_head = EvidentialHead(feat_dim, num_classes, activation)

    def forward(self, x: torch.Tensor, hr=None) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        opinion = self.edl_head(feat)
        return {"global": opinion, "local": opinion, "fused": opinion,
                "attn_weights": None}


# ═══════════════════════════════════════════════════════════════════════ #
# (D) Multi-scale + average probability
# ═══════════════════════════════════════════════════════════════════════ #

class MultiScaleAvgProb(nn.Module):
    """Global (ViT) + Local (ResNet) with simple probability averaging."""

    def __init__(
        self,
        num_classes: int = 15,
        global_backbone: str = "vit_base_patch16_224",
        local_backbone: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.global_enc = _safe_create(global_backbone, pretrained, num_classes=0)
        self.local_enc = _safe_create(local_backbone, pretrained, num_classes=0)
        self.global_head = nn.Linear(self.global_enc.num_features, num_classes)
        self.local_head = nn.Linear(self.local_enc.num_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, hr=None) -> Dict[str, torch.Tensor]:
        g_feat = self.global_enc(x)
        l_feat = self.local_enc(x)
        g_logits = self.global_head(g_feat)
        l_logits = self.local_head(l_feat)

        # Average probabilities
        g_prob = torch.softmax(g_logits, dim=-1)
        l_prob = torch.softmax(l_logits, dim=-1)
        avg_prob = (g_prob + l_prob) / 2.0

        # Wrap as opinion
        def _to_opinion(logits):
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            S = alpha.sum(dim=-1, keepdim=True)
            belief = evidence / S
            u = self.num_classes / S.squeeze(-1)
            return {"evidence": evidence, "alpha": alpha, "belief": belief,
                    "uncertainty": u, "S": S.squeeze(-1)}

        op_g = _to_opinion(g_logits)
        op_l = _to_opinion(l_logits)

        # Fused from averaged logits
        avg_logits = (g_logits + l_logits) / 2.0
        op_fused = _to_opinion(avg_logits)

        return {"global": op_g, "local": op_l, "fused": op_fused,
                "attn_weights": None, "logits": avg_logits}


# ═══════════════════════════════════════════════════════════════════════ #
# (C) Deep Ensemble (wrapper for multiple models)
# ═══════════════════════════════════════════════════════════════════════ #

class DeepEnsemble(nn.Module):
    """Ensemble of independently trained single-stream models."""

    def __init__(self, models: list, num_classes: int = 15):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, hr=None) -> Dict[str, torch.Tensor]:
        all_probs = []
        for m in self.models:
            out = m(x)
            alpha = out["fused"]["alpha"]
            S = alpha.sum(dim=-1, keepdim=True)
            all_probs.append(alpha / S)

        stacked = torch.stack(all_probs, dim=0)  # (M, B, K)
        mean_probs = stacked.mean(dim=0)
        entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=-1)

        evidence = F.softplus(mean_probs)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        belief = evidence / S
        opinion = {"evidence": evidence, "alpha": alpha, "belief": belief,
                   "uncertainty": entropy, "S": S.squeeze(-1)}
        return {"global": opinion, "local": opinion, "fused": opinion,
                "attn_weights": None}


# ═══════════════════════════════════════════════════════════════════════ #
# Multi-label Beta single-stream (M1-compliant single-scale ablation)
# ═══════════════════════════════════════════════════════════════════════ #

class BetaSingleScale(nn.Module):
    """Single backbone + per-label Beta evidential head.

    Used for the stream_global_only / stream_local_only ablations so that the
    single-stream comparison uses the same (valid) multi-label evidential
    parameterisation as the fused model.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 num_classes: int = 15, activation: str = "softplus"):
        super().__init__()
        from .beta_evidential import BetaEvidentialHead
        self.encoder = _safe_create(backbone, pretrained, num_classes=0)
        self.head = BetaEvidentialHead(self.encoder.num_features, num_classes, activation)
        self.num_classes = num_classes

    def forward(self, x, hr=None):
        op = self.head(self.encoder(x))
        # Expose the same three-key schema the trainer expects.
        return {"global": op, "local": op, "fused": op, "attn_weights": None}
