"""TMS-EDL: Trustworthy Multi-Scale Evidential Deep Learning.

Full model that wires together:
  1. Global Expert  (ViT)
  2. Local Expert   (ResNet + attention crop)
  3. Evidential Heads  (Dirichlet parameterisation)
  4. D-S Fusion    (Dempster's Rule of Combination)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .global_expert import GlobalExpert
from .local_expert import LocalExpert
from .evidential_head import EvidentialHead
from .ds_fusion import DSFusion, AverageFusion, GatingFusion
from .beta_evidential import BetaEvidentialHead, BetaFusion


class TMSEDL(nn.Module):
    """Trustworthy Multi-Scale Evidential Deep Learning framework."""

    def __init__(
        self,
        num_classes: int,
        # Global expert
        global_backbone: str = "vit_base_patch16_224",
        global_pretrained: bool = True,
        global_embed_dim: int = 768,
        # Local expert
        local_backbone: str = "resnet50",
        local_pretrained: bool = True,
        local_embed_dim: int = 2048,
        use_attention_crop: bool = True,
        crop_size: int = 224,
        top_k_patches: int = 5,
        crop_frac: float = 0.4,
        nms_radius: int = 6,
        mil_pool: str = "gated",
        # Evidential
        evidential_activation: str = "softplus",
        # Fusion
        fusion_method: str = "dempster_shafer",
        # Evidential parameterisation: "dirichlet" (original, single categorical
        # Dirichlet over C labels) or "beta" (per-label binary frames, M1 fix)
        evidential_mode: str = "beta",
        learn_discount: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.evidential_mode = evidential_mode

        # ── Experts ───────────────────────────────────────────────
        self.global_expert = GlobalExpert(
            backbone=global_backbone,
            pretrained=global_pretrained,
            embed_dim=global_embed_dim,
        )
        self.local_expert = LocalExpert(
            backbone=local_backbone,
            pretrained=local_pretrained,
            embed_dim=local_embed_dim,
            use_attention_crop=use_attention_crop,
            crop_size=crop_size,
            top_k_patches=top_k_patches,
            crop_frac=crop_frac,
            nms_radius=nms_radius,
            mil_pool=mil_pool,
        )

        # ── Evidential heads ──────────────────────────────────────
        Head = BetaEvidentialHead if evidential_mode == "beta" else EvidentialHead
        self.global_edl_head = Head(
            in_dim=global_embed_dim,
            num_classes=num_classes,
            activation=evidential_activation,
        )
        self.local_edl_head = Head(
            in_dim=local_embed_dim,
            num_classes=num_classes,
            activation=evidential_activation,
        )

        # ── Fusion ────────────────────────────────────────────────
        if evidential_mode == "beta":
            self.fusion = BetaFusion(
                method=fusion_method,
                num_classes=num_classes,
                embed_dim=global_embed_dim + local_embed_dim,
                learn_discount=learn_discount,
            )
        elif fusion_method == "dempster_shafer":
            self.fusion = DSFusion()
        elif fusion_method == "average":
            self.fusion = AverageFusion()
        elif fusion_method == "gating":
            self.fusion = GatingFusion(embed_dim=global_embed_dim + local_embed_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, discount_r: Optional[torch.Tensor] = None,
                hr: Optional[torch.Tensor] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, 3, H, W) input image.
        Returns:
            dict with keys:
                "global"  – opinion dict from global expert
                "local"   – opinion dict from local expert
                "fused"   – opinion dict after D-S fusion
                "attn_weights" – (B, top_k) or None
        """
        # 1. Extract features
        feat_global = self.global_expert(x)                    # (B, D_g)
        feat_local, attn_w = self.local_expert(x, hr=hr)      # (B, D_l)

        # 2. Evidential heads
        opinion_global = self.global_edl_head(feat_global)
        opinion_local = self.local_edl_head(feat_local)

        # 3. Fusion
        if self.evidential_mode == "beta":
            opinion_fused = self.fusion(
                opinion_global, opinion_local,
                feat_global=feat_global, feat_local=feat_local,
                discount_r=discount_r,
            )
        elif self.fusion_method == "gating":
            opinion_fused = self.fusion(
                opinion_global, opinion_local,
                feat_global=feat_global, feat_local=feat_local,
            )
        else:
            opinion_fused = self.fusion(opinion_global, opinion_local)

        return {
            "global": opinion_global,
            "local": opinion_local,
            "fused": opinion_fused,
            "attn_weights": attn_w,
        }
