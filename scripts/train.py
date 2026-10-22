"""Main training script for TMS-EDL."""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.training.utils import set_seed, load_config, save_config
from src.models.builder import build_model
from src.data.builder import build_dataloaders
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TMS-EDL")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda / cpu)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["optimizer"]["lr"] = args.lr
    if args.output_dir is not None:
        cfg["logging"]["output_dir"] = args.output_dir

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build data
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
                f"Test: {len(test_loader.dataset)}")

    # Build model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,} total, {n_train:,} trainable")

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from {args.resume}")

    # Save config
    out_dir = cfg["logging"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # Train
    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    history = trainer.train()

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
