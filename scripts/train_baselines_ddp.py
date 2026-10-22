"""Train all baseline models with DDP support.

Usage:
    torchrun --nproc_per_node=3 scripts/train_baselines_ddp.py --config config/default.yaml
    torchrun --nproc_per_node=3 scripts/train_baselines_ddp.py --config config/default.yaml \
        --baselines resnet50_ce densenet121_ce edl_single_resnet50 deep_ensemble
"""

import os
import sys
import argparse
import logging
import json
import copy
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast

from src.training.utils import set_seed, load_config
from src.data.builder import build_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics
from src.models.baselines import (
    SingleStreamModel,
    MCDropoutModel,
    EDLSingleScale,
    MultiScaleAvgProb,
    DeepEnsemble,
)


def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_num_classes(cfg):
    ds = cfg["data"]["dataset"]
    if ds == "nih_chestxray":
        return cfg["data"]["nih"]["num_classes"]
    return cfg["data"]["camelyon16"]["num_classes"]


def get_multi_label(cfg):
    ds = cfg["data"]["dataset"]
    return ds == "nih_chestxray" and cfg["data"]["nih"].get("multi_label", True)


@torch.no_grad()
def evaluate_model(model, loader, device, multi_label=False):
    model.eval()
    all_probs, all_targets, all_unc = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        with autocast(device_type="cuda", enabled=True):
            outputs = model(images)
        if "fused" in outputs:
            alpha = outputs["fused"]["alpha"]
            unc = outputs["fused"]["uncertainty"]
        else:
            alpha = outputs["alpha"]
            unc = outputs["uncertainty"]
        S = alpha.sum(dim=-1, keepdim=True)
        all_probs.append((alpha / S).cpu().numpy())
        all_targets.append(labels.numpy())
        all_unc.append(unc.cpu().numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    unc = np.concatenate(all_unc)
    metrics = compute_all_metrics(probs, targets, multi_label=multi_label)
    metrics["mean_uncertainty"] = float(unc.mean())
    return metrics


def train_and_eval_baseline(name, model, cfg, train_loader, val_loader, test_loader,
                             device, multi_label, run_dir, rank, world_size):
    run_cfg = copy.deepcopy(cfg)
    run_cfg["logging"]["output_dir"] = str(run_dir)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, train_loader, val_loader, run_cfg, device,
                      rank=rank, world_size=world_size)
    trainer.train()

    if rank == 0:
        best_ckpt = Path(run_dir) / "best_model.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
            trainer._raw_model.load_state_dict(sd)

        metrics = evaluate_model(trainer._raw_model, test_loader, device, multi_label)
        with open(Path(run_dir) / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--output_dir", type=str, default="outputs/baselines")
    p.add_argument("--baselines", nargs="+", default=None,
                   help="Subset to run. Default: all. Choices: "
                        "resnet50_ce vit_b16_ce densenet121_ce "
                        "resnet50_mc_dropout edl_single_resnet50 "
                        "multiscale_avg_prob deep_ensemble")
    p.add_argument("--ensemble_size", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [rank{rank}] [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    cfg["seed"] = args.seed
    set_seed(args.seed + rank)

    num_classes = get_num_classes(cfg)
    multi_label = get_multi_label(cfg)
    output_dir = Path(args.output_dir)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, rank=rank, world_size=world_size)
    if rank == 0:
        logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    ALL_BASELINES = [
        "resnet50_ce",
        "densenet121_ce",
        "vit_b16_ce",
        "resnet50_mc_dropout",
        "edl_single_resnet50",
        "multiscale_avg_prob",
        "deep_ensemble",
    ]
    to_run = args.baselines or ALL_BASELINES

    results = {}

    for name in to_run:
        if rank == 0:
            logger.info(f"\n{'='*60}\n  Baseline: {name}\n{'='*60}")

        set_seed(args.seed + rank)
        run_dir = output_dir / name

        if name == "resnet50_ce":
            model = SingleStreamModel("resnet50", pretrained=True, num_classes=num_classes).to(device)
        elif name == "densenet121_ce":
            model = SingleStreamModel("densenet121", pretrained=True, num_classes=num_classes).to(device)
        elif name == "vit_b16_ce":
            model = SingleStreamModel("vit_base_patch16_224", pretrained=True, num_classes=num_classes).to(device)
        elif name == "resnet50_mc_dropout":
            model = MCDropoutModel("resnet50", pretrained=True, num_classes=num_classes, dropout=0.2).to(device)
        elif name == "edl_single_resnet50":
            model = EDLSingleScale("resnet50", pretrained=True, num_classes=num_classes).to(device)
        elif name == "multiscale_avg_prob":
            model = MultiScaleAvgProb(num_classes=num_classes, pretrained=True).to(device)
        elif name == "deep_ensemble":
            # Train ensemble_size independent ResNet-50 models
            member_metrics_list = []
            member_models = []
            for i in range(args.ensemble_size):
                if rank == 0:
                    logger.info(f"  Ensemble member {i+1}/{args.ensemble_size}")
                set_seed(args.seed + i + rank)
                m = SingleStreamModel("resnet50", pretrained=True, num_classes=num_classes).to(device)
                member_dir = output_dir / "deep_ensemble" / f"member_{i}"
                member_dir.mkdir(parents=True, exist_ok=True)
                m_metrics = train_and_eval_baseline(
                    f"deep_ensemble_m{i}", m, cfg,
                    train_loader, val_loader, test_loader,
                    device, multi_label, member_dir, rank, world_size,
                )
                member_models.append(m)
                if rank == 0 and m_metrics:
                    member_metrics_list.append(m_metrics)

            if rank == 0:
                ensemble = DeepEnsemble(member_models, num_classes=num_classes).to(device)
                metrics = evaluate_model(ensemble, test_loader, device, multi_label)
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(run_dir / "test_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                results[name] = metrics
                logger.info(f"  deep_ensemble: AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  ECE={metrics['ece']:.4f}")
            continue
        else:
            if rank == 0:
                logger.warning(f"Unknown baseline {name}, skipping.")
            continue

        metrics = train_and_eval_baseline(
            name, model, cfg, train_loader, val_loader, test_loader,
            device, multi_label, run_dir, rank, world_size,
        )

        if rank == 0 and metrics:
            results[name] = metrics
            logger.info(f"  {name}: AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  ECE={metrics['ece']:.4f}")

        if is_ddp:
            dist.barrier()

    # Print summary on rank 0
    if rank == 0 and results:
        logger.info("\n" + "="*60)
        logger.info("BASELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Method':<30} {'AUC':>7}  {'ACC':>7}  {'ECE':>7}  {'Uncertainty':>12}")
        logger.info("-"*65)
        for name, m in results.items():
            logger.info(f"{name:<30} {m['auc']:>7.4f}  {m['acc']:>7.4f}  {m['ece']:>7.4f}  {m.get('mean_uncertainty',0):>12.4f}")

        # Save combined report
        with open(output_dir / "baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved to {output_dir / 'baseline_results.json'}")

    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
