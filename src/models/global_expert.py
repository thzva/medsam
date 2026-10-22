"""Global Expert – Vision Transformer for whole-image context."""

import logging
import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class GlobalExpert(nn.Module):
    """ViT-based global expert that captures long-range dependencies."""

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        embed_dim: int = 768,
    ):
        super().__init__()
        try:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights for {backbone}: {e}")
            logger.warning("Falling back to random initialisation.")
            self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        # timm returns features of shape (B, embed_dim) when num_classes=0
        actual_dim = self.encoder.num_features
        self.proj = nn.Identity() if actual_dim == embed_dim else nn.Linear(actual_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) full-resolution image.
        Returns:
            features: (B, embed_dim)
        """
        feat = self.encoder(x)          # (B, actual_dim)
        feat = self.proj(feat)           # (B, embed_dim)
        return feat
