"""Train all baseline models for comparison.

Baselines:
  (A) ResNet-50 (CE), ViT-B/16 (CE), DenseNet-121 (CE)
  (B) ResNet-50 + Temperature Scaling
  (C) MC Dropout, Deep Ensemble, EDL single-scale
  (D) Global+Local + Avg prob
"""

import os
import sys
import argparse
import logging
import json
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.training.utils import set_seed, load_config, save_config
from src.data.builder import build_dataloaders
from src.training.trainer import Trainer
from src.models.baselines import (
    SingleStreamModel,
    TemperatureScaling,
    MCDropoutModel,
    EDLSingleScale,
    MultiScaleAvgProb,
    DeepEnsemble,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


BASELINE_CONFIGS = {
    # (A) Single-stream
    "resnet50_ce": {
        "class": SingleStreamModel,
        "kwargs": {"backbone": "resnet50", "pretrained": True},
    },
    "vit_b16_ce": {
        "class": SingleStreamModel,
        "kwargs": {"backbone": "vit_base_patch16_224", "pretrained": True},
    },
    "densenet121_ce": {
        "class": SingleStreamModel,
        "kwargs": {"backbone": "densenet121", "pretrained": True},
    },
    # (B) Temperature Scaling (post-hoc on ResNet-50)
    "resnet50_temp_scaling": {
        "class": "temp_scaling",
        "base": "resnet50",
    },
    # (C) MC Dropout
    "resnet50_mc_dropout": {
        "class": MCDropoutModel,
        "kwargs": {"backbone": "resnet50", "pretrained": True, "dropout": 0.2, "n_samples": 10},
    },
    # (C) EDL single-scale
    "edl_single_resnet50": {
        "class": EDLSingleScale,
        "kwargs": {"backbone": "resnet50", "pretrained": True, "activation": "softplus"},
    },
    # (D) Multi-scale average probability
    "multiscale_avg_prob": {
        "class": MultiScaleAvgProb,
        "kwargs": {"global_backbone": "vit_base_patch16_224", "local_backbone": "resnet50",
                   "pretrained": True},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--baselines", nargs="+", default=None,
                        help="Specific baselines to train (default: all)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ensemble_size", type=int, default=3,
                        help="Number of models for Deep Ensemble")
    return parser.parse_args()


def get_num_classes(cfg):
    ds = cfg["data"]["dataset"]
    if ds == "nih_chestxray":
        return cfg["data"]["nih"]["num_classes"]
    return cfg["data"]["camelyon16"]["num_classes"]


def train_single_baseline(name, bcfg, cfg, train_loader, val_loader, device, output_dir):
    """Train a single baseline model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training baseline: {name}")
    logger.info(f"{'='*60}")

    num_classes = get_num_classes(cfg)
    run_dir = os.path.join(output_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    run_cfg = copy.deepcopy(cfg)
    run_cfg["logging"]["output_dir"] = run_dir

    if bcfg["class"] == "temp_scaling":
        # First train base model, then calibrate
        base_model = SingleStreamModel(
            backbone=bcfg["base"], pretrained=True, num_classes=num_classes,
        )
        base_model = base_model.to(device)
        trainer = Trainer(base_model, train_loader, val_loader, run_cfg, device)
        trainer.train()

        # Calibrate temperature
        model = TemperatureScaling(base_model)
        model.calibrate(val_loader, device)
        torch.save({"model_state_dict": model.state_dict()},
                    os.path.join(run_dir, "best_model.pth"))
    else:
        kwargs = bcfg["kwargs"].copy()
        kwargs["num_classes"] = num_classes
        model = bcfg["class"](**kwargs)
        model = model.to(device)
        trainer = Trainer(model, train_loader, val_loader, run_cfg, device)
        trainer.train()

    logger.info(f"Baseline {name} training complete.")
    return model


def train_deep_ensemble(cfg, train_loader, val_loader, device, output_dir, n_models=3):
    """Train Deep Ensemble (multiple independent models)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Deep Ensemble ({n_models} models)")
    logger.info(f"{'='*60}")

    num_classes = get_num_classes(cfg)
    run_dir = os.path.join(output_dir, "deep_ensemble")
    os.makedirs(run_dir, exist_ok=True)

    models = []
    for i in range(n_models):
        logger.info(f"  Ensemble member {i+1}/{n_models}")
        set_seed(cfg.get("seed", 42) + i)  # different seed per member
        member_dir = os.path.join(run_dir, f"member_{i}")
        run_cfg = copy.deepcopy(cfg)
        run_cfg["logging"]["output_dir"] = member_dir

        model = SingleStreamModel(
            backbone="resnet50", pretrained=True, num_classes=num_classes,
        )
        model = model.to(device)
        trainer = Trainer(model, train_loader, val_loader, run_cfg, device)
        trainer.train()
        models.append(model)

    ensemble = DeepEnsemble(models, num_classes=num_classes)
    torch.save({"model_state_dict": ensemble.state_dict()},
               os.path.join(run_dir, "best_model.pth"))
    logger.info("Deep Ensemble training complete.")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    baselines_to_train = args.baselines or list(BASELINE_CONFIGS.keys())

    for name in baselines_to_train:
        if name == "deep_ensemble":
            train_deep_ensemble(cfg, train_loader, val_loader, device,
                                args.output_dir, args.ensemble_size)
        elif name in BASELINE_CONFIGS:
            train_single_baseline(
                name, BASELINE_CONFIGS[name], cfg,
                train_loader, val_loader, device, args.output_dir,
            )
        else:
            logger.warning(f"Unknown baseline: {name}, skipping.")

    # Also train deep ensemble if "all" was requested
    if args.baselines is None:
        train_deep_ensemble(cfg, train_loader, val_loader, device,
                            args.output_dir, args.ensemble_size)

    logger.info("\nAll baselines trained.")


if __name__ == "__main__":
    main()
