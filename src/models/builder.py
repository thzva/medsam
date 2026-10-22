"""Build model from config dict."""

from typing import Dict
from .tms_edl import TMSEDL


def build_model(cfg: Dict) -> TMSEDL:
    """Instantiate TMS-EDL from a config dictionary."""
    mcfg = cfg["model"]
    g = mcfg["global_expert"]
    l = mcfg["local_expert"]
    e = mcfg["evidential"]
    f = mcfg["fusion"]

    ds_name = cfg["data"]["dataset"]
    if ds_name == "nih_chestxray":
        num_classes = cfg["data"]["nih"]["num_classes"]
    elif ds_name == "camelyon16":
        num_classes = cfg["data"]["camelyon16"]["num_classes"]
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    model = TMSEDL(
        num_classes=num_classes,
        global_backbone=g["backbone"],
        global_pretrained=g["pretrained"],
        global_embed_dim=g["embed_dim"],
        local_backbone=l["backbone"],
        local_pretrained=l["pretrained"],
        local_embed_dim=l["embed_dim"],
        use_attention_crop=l.get("use_attention_crop", True),
        crop_size=l.get("crop_size", 224),
        top_k_patches=l.get("top_k_patches", 5),
        crop_frac=l.get("crop_frac", 0.4),
        nms_radius=l.get("nms_radius", 6),
        mil_pool=l.get("mil_pool", "gated"),
        evidential_activation=e.get("activation", "softplus"),
        fusion_method=f.get("method", "dempster_shafer"),
        evidential_mode=e.get("mode", "beta"),
        learn_discount=f.get("learn_discount", False),
    )
    return model
