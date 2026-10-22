"""DDP training entry point for TMS-EDL.

Launch with torchrun:
    torchrun --nproc_per_node=3 scripts/train_ddp.py --config config/default.yaml

Single-GPU fallback still works:
    python scripts/train_ddp.py --config config/default.yaml
"""
import os
# Bound the intra-op thread pools BEFORE torch is imported.  Unset, each
# process opens one OpenMP thread per core (80 here); with several training
# and evaluation jobs running concurrently that produced ~1.5M context
# switches per second and a load average of 400+ while the CPU sat 62% idle.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import os
import sys
import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
torch.set_num_threads(4)
import torch.distributed as dist
from torch.amp import autocast

from src.training.utils import set_seed, load_config
from src.data.builder import build_dataloaders
from src.models.builder import build_model
from src.models.baselines import EDLSingleScale, SingleStreamModel, MCDropoutModel, MultiScaleAvgProb, DeepEnsemble, BetaSingleScale
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics


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
        hr = batch.get("image_hr")
        hr = hr.to(device, non_blocking=True) if hr is not None else None
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(images, hr=hr) if hr is not None else model(images)
        from src.models.opinion_utils import opinion_prob, opinion_uncertainty
        op = outputs["fused"] if "fused" in outputs else outputs
        all_probs.append(opinion_prob(op).float().cpu().numpy())
        unc = opinion_uncertainty(op)
        all_targets.append(labels.numpy())
        all_unc.append(unc.float().cpu().numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    unc = np.concatenate(all_unc)
    metrics = compute_all_metrics(probs, targets, multi_label=multi_label)
    metrics["mean_uncertainty"] = float(unc.mean())
    return metrics


ABLATION_MODES = [
    "fusion_dempster_shafer", "fusion_average", "fusion_gating",
    "stream_global_only", "stream_local_only",
    "kl_disabled", "kl_no_annealing",
    "attn_crop_off", "loss_edl_mse",
    # ── added for the revision ──────────────────────────────────────────
    "fusion_yager",        # R1-4: conflict-robust alternative to Dempster
    "fusion_cautious",     # R1-4/R2-3: mean-BBA, does not double-count
    "fusion_discounted",   # R2-3: learned source discounting r_s per branch
    "mil_no_nms",          # isolates the MIL head from NMS diversification
    "evid_dirichlet",      # M1 control: the original single categorical Dirichlet
    "mil_uniform",         # TVC: legacy near-uniform MIL pooling under the current code
]

BASELINE_MODES = [
    "resnet50_ce", "densenet121_ce", "vit_b16_ce",
    "resnet50_mc_dropout", "edl_single_resnet50", "multiscale_avg_prob",
]


def build_model_for_mode(mode, cfg, device):
    """Build model and (optionally) modified config for a given run mode."""
    num_classes = get_num_classes(cfg)
    cfg = copy.deepcopy(cfg)
    beta = cfg["model"].get("evidential", {}).get("mode", "beta") == "beta"

    # ── Ablations ──────────────────────────────────────────────────────
    if mode == "fusion_dempster_shafer":
        cfg["model"]["fusion"]["method"] = "dempster_shafer"
        return build_model(cfg).to(device), cfg
    elif mode == "fusion_average":
        cfg["model"]["fusion"]["method"] = "average"
        return build_model(cfg).to(device), cfg
    elif mode == "fusion_gating":
        cfg["model"]["fusion"]["method"] = "gating"
        return build_model(cfg).to(device), cfg
    elif mode == "stream_global_only":
        Single = BetaSingleScale if beta else EDLSingleScale
        m = Single(cfg["model"]["global_expert"]["backbone"], True, num_classes).to(device)
        return m, cfg
    elif mode == "stream_local_only":
        Single = BetaSingleScale if beta else EDLSingleScale
        m = Single(cfg["model"]["local_expert"]["backbone"], True, num_classes).to(device)
        return m, cfg
    elif mode in ("fusion_yager", "fusion_cautious"):
        cfg["model"]["fusion"]["method"] = mode.replace("fusion_", "")
        return build_model(cfg).to(device), cfg
    elif mode == "fusion_discounted":
        cfg["model"]["fusion"]["method"] = "dempster_shafer"
        cfg["model"]["fusion"]["learn_discount"] = True
        return build_model(cfg).to(device), cfg
    elif mode == "mil_no_nms":
        cfg["model"]["local_expert"]["nms_radius"] = 0   # MIL head, plain top-k
        return build_model(cfg).to(device), cfg
    elif mode == "mil_uniform":
        cfg["model"]["local_expert"]["mil_pool"] = "legacy_uniform"
        cfg["model"]["local_expert"]["nms_radius"] = 0   # the legacy selector had no NMS
        return build_model(cfg).to(device), cfg
    elif mode == "evid_dirichlet":
        cfg["model"].setdefault("evidential", {})["mode"] = "dirichlet"
        return build_model(cfg).to(device), cfg
    elif mode == "kl_disabled":
        cfg["training"]["loss"]["kl_weight"] = 0.0
        return build_model(cfg).to(device), cfg
    elif mode == "kl_no_annealing":
        cfg["training"]["loss"]["annealing_epochs"] = 0
        return build_model(cfg).to(device), cfg
    elif mode == "attn_crop_off":
        cfg["model"]["local_expert"]["use_attention_crop"] = False
        return build_model(cfg).to(device), cfg
    elif mode == "loss_edl_mse":
        cfg["training"]["loss"]["type"] = "mse"
        return build_model(cfg).to(device), cfg
    # ── Baselines ──────────────────────────────────────────────────────
    elif mode == "resnet50_ce":
        cfg["training"]["loss"]["type"] = "bce"   # genuine deterministic BCE baseline (R1-6)
        return SingleStreamModel("resnet50", True, num_classes).to(device), cfg
    elif mode == "densenet121_ce":
        cfg["training"]["loss"]["type"] = "bce"   # genuine deterministic BCE baseline (R1-6)
        return SingleStreamModel("densenet121", True, num_classes).to(device), cfg
    elif mode == "vit_b16_ce":
        cfg["training"]["loss"]["type"] = "bce"   # genuine deterministic BCE baseline (R1-6)
        return SingleStreamModel("vit_base_patch16_224", True, num_classes).to(device), cfg
    elif mode == "resnet50_mc_dropout":
        cfg["training"]["loss"]["type"] = "bce"
        return MCDropoutModel("resnet50", True, num_classes, dropout=0.2).to(device), cfg
    elif mode == "edl_single_resnet50":
        return EDLSingleScale("resnet50", True, num_classes).to(device), cfg
    elif mode == "multiscale_avg_prob":
        return MultiScaleAvgProb(num_classes=num_classes, pretrained=True).to(device), cfg
    # ── Main model ─────────────────────────────────────────────────────
    else:  # "tms_edl" or default
        return build_model(cfg).to(device), cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output dir (default: outputs/<mode>)")
    p.add_argument("--mode", type=str, default="tms_edl",
                   help="What to train: tms_edl | ablation_<name> | baseline_<name> | "
                        "all_ablations | all_baselines")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one_mode(mode, cfg, train_loader, val_loader, test_loader,
                 device, multi_label, base_output_dir, rank, world_size, logger):
    """Train + evaluate a single mode. Returns metrics dict or None (non-rank-0)."""
    # Determine output dir
    tag = f"seed{cfg.get('seed', 42)}"
    if mode == "tms_edl":
        output_dir = Path(base_output_dir) / "tms_edl" / tag
    elif mode in ABLATION_MODES:
        output_dir = Path(base_output_dir) / "ablations" / f"ablation_{mode}" / tag
    else:
        output_dir = Path(base_output_dir) / "baselines" / mode / tag

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n{'='*60}\n  Mode: {mode}  ->  {output_dir}\n{'='*60}")

    model, run_cfg = build_model_for_mode(mode, cfg, device)
    run_cfg["logging"]["output_dir"] = str(output_dir)

    trainer = Trainer(model, train_loader, val_loader, run_cfg, device,
                      rank=rank, world_size=world_size)
    trainer.train()

    if rank == 0:
        best_ckpt = output_dir / "best_model.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
            trainer._raw_model.load_state_dict(sd)
            logger.info(f"  Loaded best epoch={ckpt['epoch']} val_auc={ckpt['best_val_metric']:.4f}")

        metrics = evaluate_model(trainer._raw_model, test_loader, device, multi_label)
        logger.info(f"  Test AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  ECE={metrics['ece']:.4f}")
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    return None


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
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    cfg["seed"] = args.seed
    set_seed(args.seed + rank)

    multi_label = get_multi_label(cfg)
    num_classes = get_num_classes(cfg)
    base_output_dir = args.output_dir or "outputs"

    if rank == 0:
        logger.info(f"World size: {world_size} | Device: {device} | Mode: {args.mode}")
        logger.info(f"Dataset: {cfg['data']['dataset']} | Classes: {num_classes}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg, rank=rank, world_size=world_size)
    if rank == 0:
        logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Decide which modes to run
    mode = args.mode
    if mode == "all_ablations":
        modes_to_run = ABLATION_MODES
    elif mode == "all_baselines":
        modes_to_run = BASELINE_MODES
    else:
        modes_to_run = [mode]

    all_results = {}
    for m in modes_to_run:
        set_seed(args.seed + rank)
        metrics = run_one_mode(
            m, cfg, train_loader, val_loader, test_loader,
            device, multi_label, base_output_dir, rank, world_size, logger,
        )
        if rank == 0 and metrics:
            all_results[m] = metrics
        if is_ddp:
            dist.barrier()

    if rank == 0 and len(all_results) > 1:
        logger.info("\n" + "="*65)
        logger.info(f"{'Mode':<35} {'AUC':>7}  {'ACC':>7}  {'ECE':>7}")
        logger.info("-"*65)
        for m, met in all_results.items():
            logger.info(f"{m:<35} {met['auc']:>7.4f}  {met['acc']:>7.4f}  {met['ece']:>7.4f}")

    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
