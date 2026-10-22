"""Local Expert – CNN with attention-based ROI cropping for fine-grained details."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logger = logging.getLogger(__name__)


class AttentionCrop(nn.Module):
    """Lightweight attention module that selects top-k informative patches.

    Given a feature map from a shallow CNN pass, it computes a spatial
    attention map, selects the top-k regions, crops them from the original
    image, and returns the stacked patches.
    """

    def __init__(self, in_channels: int = 64, top_k: int = 5, crop_size: int = 224,
                 crop_frac: float = 0.4, out_size: int = 224,
                 nms_radius: int = 6):
        # nms_radius = 0 disables suppression (plain top-k), which isolates the
        # effect of the MIL head from the effect of spatial diversification.
        super().__init__()
        self.top_k = top_k
        self.crop_size = crop_size      # kept for config compatibility
        self.crop_frac = crop_frac      # crop extent as a fraction of the image
        self.out_size = out_size        # resolution handed to the CNN
        self.nms_radius = nms_radius    # suppression radius on the attention grid
        # D7: the previous version ended this stack in nn.Sigmoid() and then took
        # a softmax over those (0,1) values.  Softmax needs logits: with sigmoid
        # outputs the six weights can never leave [0.095, 0.259], and with the
        # near-saturated values the network actually produced they were exactly
        # uniform (normalised entropy 1.0000), so the MIL attention never
        # weighted anything.  The conv now emits logits and the softmax is taken
        # over them, with a learnable temperature.
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, feature_map: torch.Tensor, original_image: torch.Tensor,
                hr_image: torch.Tensor = None):
        """
        Args:
            feature_map: (B, C, h, w) from a shallow feature extractor.
            original_image: (B, 3, H, W) the 224x224 model input.
            hr_image: (B, 3, H2, W2) higher-resolution source to crop from.
                D8: cropping from `original_image` cannot yield detail the
                global stream does not already have -- the radiographs are
                resized 1024 -> 224 before the model, so a 160px crop of the
                224px tensor upsampled back to 224 carries 0.22x the global
                stream's high-frequency energy.  When an HR tensor is supplied
                the same normalised regions are sampled from it instead.
        Returns:
            patches:  (B, top_k, 3, out_size, out_size)
            topk_idx:   (B, top_k) selected grid positions
            sel_scores: (B, top_k) selector logits at those positions; the only
                        differentiable path back to attn_conv
        """
        src = hr_image if hr_image is not None else original_image
        B = src.shape[0]
        h, w = feature_map.shape[2], feature_map.shape[3]

        attn_logits = self.attn_conv(feature_map).view(B, -1)      # (B, h*w) logits

        # Greedy non-maximum suppression over the attention grid.  Plain topk
        # on a smooth map returns six adjacent cells, so the resulting crops
        # overlapped heavily (mean pairwise correlation 0.52) and carried the
        # same evidence -- exactly the double counting that makes Dempster
        # fusion collapse.  Suppressing a radius around each pick forces the
        # patches to come from genuinely different regions.
        if self.nms_radius <= 0:
            topk_idx = attn_logits.topk(self.top_k, dim=1).indices
            rows_, cols_ = topk_idx // w, topk_idx % w
            # Gather the scores AT the chosen positions.  topk/argmax are not
            # differentiable, so without this the selector conv is severed from
            # the loss: it received no gradient at all and stayed bit-identical
            # to its random initialisation for the whole of training, which made
            # patch selection effectively random.
            sel_scores = attn_logits.gather(1, topk_idx)
            return self._crop(src, rows_, cols_, h, w, B), topk_idx, sel_scores
        scores = attn_logits.clone()
        idx_list = []
        for _ in range(self.top_k):
            best = scores.argmax(dim=1)                              # (B,)
            idx_list.append(best)
            br, bc = best // w, best % w
            rr = torch.arange(h, device=scores.device).view(1, h, 1)
            cc = torch.arange(w, device=scores.device).view(1, 1, w)
            near = ((rr - br.view(B, 1, 1)).abs() <= self.nms_radius) & \
                   ((cc - bc.view(B, 1, 1)).abs() <= self.nms_radius)
            scores = scores.masked_fill(near.view(B, -1), float("-inf"))
        topk_idx = torch.stack(idx_list, dim=1)                      # (B, k)

        rows = topk_idx // w
        cols = topk_idx % w
        sel_scores = attn_logits.gather(1, topk_idx)
        return self._crop(src, rows, cols, h, w, B), topk_idx, sel_scores

    def _crop(self, src, rows, cols, h, w, B):
        cy = (rows.float() + 0.5) / h * 2.0 - 1.0
        cx = (cols.float() + 0.5) / w * 2.0 - 1.0

        # Half-extent of the crop in normalised coordinates.  crop_frac is the
        # fraction of the image each crop covers, so the half-extent is
        # crop_frac (since the full range [-1, 1] has extent 2).
        half = self.crop_frac
        g = torch.linspace(-half, half, self.out_size, device=src.device)
        grid_y, grid_x = torch.meshgrid(g, g, indexing="ij")

        grid_x = cx[:, :, None, None] + grid_x[None, None, :, :]
        grid_y = cy[:, :, None, None] + grid_y[None, None, :, :]
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.view(B * self.top_k, self.out_size, self.out_size, 2)

        src_exp = src.unsqueeze(1).expand(-1, self.top_k, -1, -1, -1)
        src_exp = src_exp.reshape(B * self.top_k, *src.shape[1:])

        patches = F.grid_sample(src_exp, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        return patches.view(B, self.top_k, 3, self.out_size, self.out_size)
        return patches, topk_idx


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling over instance embeddings (Ilse et al., 2018).

        a_k = softmax_k ( w^T [ tanh(V h_k) * sigmoid(U h_k) ] )
        z   = sum_k a_k h_k

    The weights are a function of the ENCODED patch features h_k, which is what
    makes them informative.  The previous implementation took a softmax over
    the top-k values of the selection heat-map instead; because top-k already
    returns the six largest cells of a smooth map, those values are necessarily
    close together and the softmax was mathematically pinned near uniform
    (measured normalised entropy 0.9999 after training, i.e. a plain mean).
    """

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)
        # Mixing weight for the selector's own score.  Initialised small so the
        # head starts out purely content-based and learns how much to trust the
        # selector, while still giving attn_conv a gradient from step one.
        self.score_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, sel_scores: torch.Tensor = None):
        """h: (B, K, D), sel_scores: (B, K) -> pooled (B, D), weights (B, K)"""
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        if sel_scores is not None:
            a = a + self.score_gain * sel_scores.to(a.dtype)
        a = F.softmax(a.float(), dim=1)
        return (h * a.unsqueeze(-1)).sum(dim=1), a


class LocalExpert(nn.Module):
    """CNN-based local expert with optional attention-based ROI cropping."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 2048,
        use_attention_crop: bool = True,
        crop_size: int = 224,
        top_k_patches: int = 5,
        crop_frac: float = 0.4,
        mil_hidden: int = 256,
        nms_radius: int = 6,
        mil_pool: str = "gated",
    ):
        super().__init__()
        self.use_attention_crop = use_attention_crop
        # "gated": gated-attention MIL (default).  "legacy_uniform": reproduces the
        # original pooling, softmax over sigmoid(selector logits), whose weights are
        # pinned near uniform (normalised entropy 0.9999) but still pass a gradient
        # to the selector.  Used only to re-run the uniform-MIL ablation row.
        assert mil_pool in ("gated", "legacy_uniform")
        self.mil_pool = mil_pool
        self.top_k = top_k_patches

        # Full CNN encoder (features only)
        try:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights for {backbone}: {e}")
            logger.warning("Falling back to random initialisation.")
            self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        actual_dim = self.encoder.num_features
        self.proj = nn.Identity() if actual_dim == embed_dim else nn.Linear(actual_dim, embed_dim)
        self.embed_dim = embed_dim

        if use_attention_crop:
            # Shallow feature extractor for attention map
            self.shallow = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
            self.attn_crop = AttentionCrop(
                in_channels=64, top_k=top_k_patches, crop_size=crop_size,
                crop_frac=crop_frac, out_size=224, nms_radius=nms_radius,
            )
            self.mil = GatedAttentionMIL(embed_dim, mil_hidden)

    def forward(self, x: torch.Tensor, hr: torch.Tensor = None):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, embed_dim)  aggregated local features.
            attn_weights: (B, top_k) or None.
        """
        if self.use_attention_crop:
            shallow_feat = self.shallow(x)                       # (B, 64, h, w)
            patches, _sel_idx, sel_scores = self.attn_crop(shallow_feat, x, hr_image=hr)
            B, K = patches.shape[:2]
            patches_flat = patches.view(B * K, *patches.shape[2:])   # (B*k, 3, cs, cs)
            # Crops are already sampled at 224; no upsampling needed.
            feats = self.encoder(patches_flat)                       # (B*k, D)
            feats = self.proj(feats)
            feats = feats.view(B, K, -1)                             # (B, k, D)
            if self.mil_pool == "legacy_uniform":
                attn_weights = F.softmax(torch.sigmoid(sel_scores.float()), dim=1)
                pooled = (feats * attn_weights.unsqueeze(-1).to(feats.dtype)).sum(dim=1)
                return pooled, attn_weights
            pooled, attn_weights = self.mil(feats, sel_scores)       # gated MIL
            return pooled, attn_weights
        else:
            feats = self.encoder(x)
            feats = self.proj(feats)
            return feats, None
