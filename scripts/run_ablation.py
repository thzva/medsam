"""Ablation study script for TMS-EDL.

Ablation experiments:
  1. Fusion method:  D-S  vs  Average  vs  Gating
  2. Expert streams: Global-only  vs  Local-only  vs  Both
  3. EDL activation: softplus  vs  relu  vs  exp
  4. KL annealing:   with  vs  without
  5. Attention crop:  with  vs  without
  6. Loss variant:    EDL-log  vs  EDL-mse  vs  CE
"""

import os
import sys
import argparse
import logging
import json
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.cuda.amp import autocast

from src.training.utils import set_seed, load_config, save_config
from src.data.builder import build_dataloaders
from src.models.builder import build_model
from src.models.baselines import SingleStreamModel, EDLSingleScale
from src.models.tms_edl import TMSEDL
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Ablation definitions ──────────────────────────────────────────────── #

def get_num_classes(cfg):
    ds = cfg["data"]["dataset"]
    if ds == "nih_chestxray":
        return cfg["data"]["nih"]["num_classes"]
    return cfg["data"]["camelyon16"]["num_classes"]


def ablation_fusion_method(base_cfg):
    """Ablation 1: Compare fusion methods."""
    variants = {}
    for method in ["dempster_shafer", "average", "gating"]:
        c = copy.deepcopy(base_cfg)
        c["model"]["fusion"]["method"] = method
        variants[f"fusion_{method}"] = c
    return variants


def ablation_expert_streams(base_cfg):
    """Ablation 2: Global-only, Local-only, Both."""
    num_classes = get_num_classes(base_cfg)
    variants = {}

    # Both (full model) — use base config as-is
    variants["stream_both"] = copy.deepcopy(base_cfg)

    # Global-only and Local-only are handled via special model construction
    # We mark them with a flag so the runner knows
    c_global = copy.deepcopy(base_cfg)
    c_global["_ablation_stream"] = "global_only"
    variants["stream_global_only"] = c_global

    c_local = copy.deepcopy(base_cfg)
    c_local["_ablation_stream"] = "local_only"
    variants["stream_local_only"] = c_local

    return variants


def ablation_edl_activation(base_cfg):
    """Ablation 3: EDL activation function."""
    variants = {}
    for act in ["softplus", "relu", "exp"]:
        c = copy.deepcopy(base_cfg)
        c["model"]["evidential"]["activation"] = act
        variants[f"activation_{act}"] = c
    return variants


def ablation_kl_annealing(base_cfg):
    """Ablation 4: KL annealing vs no annealing."""
    variants = {}

    c_with = copy.deepcopy(base_cfg)
    c_with["training"]["loss"]["annealing_epochs"] = 10
    variants["kl_with_annealing"] = c_with

    c_without = copy.deepcopy(base_cfg)
    c_without["training"]["loss"]["annealing_epochs"] = 0
    variants["kl_no_annealing"] = c_without

    c_no_kl = copy.deepcopy(base_cfg)
    c_no_kl["training"]["loss"]["kl_weight"] = 0.0
    variants["kl_disabled"] = c_no_kl

    return variants


def ablation_attention_crop(base_cfg):
    """Ablation 5: With vs without attention-based ROI cropping."""
    variants = {}

    c_with = copy.deepcopy(base_cfg)
    c_with["model"]["local_expert"]["use_attention_crop"] = True
    variants["attn_crop_on"] = c_with

    c_without = copy.deepcopy(base_cfg)
    c_without["model"]["local_expert"]["use_attention_crop"] = False
    variants["attn_crop_off"] = c_without

    return variants


def ablation_loss_variant(base_cfg):
    """Ablation 6: EDL-log vs EDL-mse vs CE."""
    variants = {}

    c_log = copy.deepcopy(base_cfg)
    c_log["training"]["loss"]["type"] = "log"
    variants["loss_edl_log"] = c_log

    c_mse = copy.deepcopy(base_cfg)
    c_mse["training"]["loss"]["type"] = "mse"
    variants["loss_edl_mse"] = c_mse

    c_ce = copy.deepcopy(base_cfg)
    c_ce["training"]["loss"]["type"] = "ce"
    variants["loss_ce"] = c_ce

    return variants


ALL_ABLATIONS = {
    "fusion": ablation_fusion_method,
    "streams": ablation_expert_streams,
    "activation": ablation_edl_activation,
    "kl": ablation_kl_annealing,
    "attn_crop": ablation_attention_crop,
    "loss": ablation_loss_variant,
}


# ── Runner ─────────────────────────────────────────────────────────────── #

def build_ablation_model(cfg, device):
    """Build model, handling special ablation flags."""
    num_classes = get_num_classes(cfg)
    stream = cfg.pop("_ablation_stream", None)

    if stream == "global_only":
        # Single-stream ViT with EDL head
        model = EDLSingleScale(
            backbone=cfg["model"]["global_expert"]["backbone"],
            pretrained=cfg["model"]["global_expert"]["pretrained"],
            num_classes=num_classes,
            activation=cfg["model"]["evidential"]["activation"],
        )
    elif stream == "local_only":
        model = EDLSingleScale(
            backbone=cfg["model"]["local_expert"]["backbone"],
            pretrained=cfg["model"]["local_expert"]["pretrained"],
            num_classes=num_classes,
            activation=cfg["model"]["evidential"]["activation"],
        )
    else:
        model = build_model(cfg)

    return model.to(device)


@torch.no_grad()
def evaluate_model(model, loader, device, multi_label=False):
    """Quick evaluation returning metrics dict."""
    model.eval()
    all_probs, all_targets, all_unc = [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        with autocast(enabled=True):
            outputs = model(images)
        alpha = outputs["fused"]["alpha"]
        S = alpha.sum(dim=-1, keepdim=True)
        probs = (alpha / S).cpu().numpy()
        unc = outputs["fused"]["uncertainty"].cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.numpy())
        all_unc.append(unc)

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    uncertainty = np.concatenate(all_unc)

    metrics = compute_all_metrics(probs, targets, multi_label=multi_label)
    metrics["mean_uncertainty"] = float(uncertainty.mean())
    return metrics


def run_single_ablation(name, cfg, train_loader, val_loader, test_loader,
                        device, output_dir, multi_label):
    """Train and evaluate a single ablation variant."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"Ablation: {name}")
    logger.info(f"{'─'*60}")

    run_dir = os.path.join(output_dir, name)
    run_cfg = copy.deepcopy(cfg)
    run_cfg["logging"]["output_dir"] = run_dir

    model = build_ablation_model(run_cfg, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, run_cfg, device)
    trainer.train()

    # Evaluate on test set
    metrics = evaluate_model(model, test_loader, device, multi_label)
    logger.info(f"  Test results: {json.dumps(metrics, indent=2)}")

    # Save
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_config(run_cfg, os.path.join(run_dir, "config.yaml"))

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Which ablations to run (default: all). "
                             f"Choices: {list(ALL_ABLATIONS.keys())}")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/ablations")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    ds = cfg["data"]["dataset"]
    multi_label = (ds == "nih_chestxray" and cfg["data"]["nih"].get("multi_label", True))

    ablations_to_run = args.ablations or list(ALL_ABLATIONS.keys())
    all_results = {}

    for abl_name in ablations_to_run:
        if abl_name not in ALL_ABLATIONS:
            logger.warning(f"Unknown ablation: {abl_name}, skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION GROUP: {abl_name}")
        logger.info(f"{'='*60}")

        variants = ALL_ABLATIONS[abl_name](cfg)
        group_results = {}

        for variant_name, variant_cfg in variants.items():
            set_seed(cfg.get("seed", 42))
            metrics = run_single_ablation(
                variant_name, variant_cfg,
                train_loader, val_loader, test_loader,
                device, os.path.join(args.output_dir, abl_name),
                multi_label,
            )
            group_results[variant_name] = metrics

        all_results[abl_name] = group_results

    # ── Summary table ──────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Ablation':<30s} {'AUC':>8s} {'ACC':>8s} {'ECE':>8s}")
    logger.info("-" * 60)

    for group, variants in all_results.items():
        for name, m in variants.items():
            logger.info(f"{name:<30s} {m.get('auc',0):>8.4f} "
                        f"{m.get('acc',0):>8.4f} {m.get('ece',1):>8.4f}")
        logger.info("-" * 60)

    # Save full results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAll ablation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
