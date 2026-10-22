"""Evaluation script for TMS-EDL and baselines."""

import os
import sys
import argparse
import logging
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.cuda.amp import autocast

from src.training.utils import set_seed, load_config
from src.models.builder import build_model
from src.data.builder import build_dataloaders
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.calibration import reliability_diagram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TMS-EDL")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--ood_dir", type=str, default=None,
                        help="Path to OOD image folder for OOD detection test")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, use_amp=True):
    """Run inference and collect predictions."""
    model.eval()
    all_probs, all_targets, all_uncertainty = [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        with autocast(enabled=use_amp):
            outputs = model(images)

        alpha = outputs["fused"]["alpha"]
        S = alpha.sum(dim=-1, keepdim=True)
        probs = (alpha / S).cpu().numpy()
        unc = outputs["fused"]["uncertainty"].cpu().numpy()

        all_probs.append(probs)
        all_targets.append(labels.numpy())
        all_uncertainty.append(unc)

    return (np.concatenate(all_probs),
            np.concatenate(all_targets),
            np.concatenate(all_uncertainty))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # Build data
    _, _, test_loader = build_dataloaders(cfg)
    logger.info(f"Test set size: {len(test_loader.dataset)}")

    # Build & load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Determine multi_label
    ds = cfg["data"]["dataset"]
    multi_label = (ds == "nih_chestxray" and cfg["data"]["nih"].get("multi_label", True))

    # Run inference
    probs, targets, uncertainty = run_inference(model, test_loader, device)

    # OOD detection
    ood_uncertainty = None
    if args.ood_dir and os.path.isdir(args.ood_dir):
        from src.data.ood_dataset import OODDataset
        from torch.utils.data import DataLoader
        ood_ds = OODDataset(args.ood_dir, image_size=cfg["data"].get("image_size", 224))
        ood_loader = DataLoader(ood_ds, batch_size=cfg["training"]["batch_size"],
                                num_workers=cfg["data"].get("num_workers", 4))
        _, _, ood_uncertainty = run_inference(model, ood_loader, device)
        logger.info(f"OOD samples: {len(ood_ds)}")

    # Compute metrics
    metrics = compute_all_metrics(
        probs, targets,
        uncertainty=uncertainty,
        ood_uncertainty=ood_uncertainty,
        multi_label=multi_label,
    )

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Reliability diagram
    reliability_diagram(
        probs, targets,
        save_path=os.path.join(args.output_dir, "reliability_diagram.png"),
        title="TMS-EDL Reliability Diagram",
    )
    logger.info(f"Results saved to {args.output_dir}")

    if args.save_predictions:
        np.savez(
            os.path.join(args.output_dir, "predictions.npz"),
            probs=probs, targets=targets, uncertainty=uncertainty,
        )


if __name__ == "__main__":
    main()
