"""
One-click script: train TMS-EDL + ablations + evaluate + summarize.

Usage:
    python scripts/run_full_experiment.py --config config/default.yaml
    python scripts/run_full_experiment.py --config config/default.yaml --epochs 30 --skip_ablations
"""

import os
import sys
import argparse
import json
import time
import copy
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.amp import autocast

from src.training.utils import set_seed, load_config, save_config
from src.data.builder import build_dataloaders
from src.models.builder import build_model
from src.models.baselines import EDLSingleScale
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.calibration import reliability_diagram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════ #
#  Helpers
# ═══════════════════════════════════════════════════════════════════════ #

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
        alpha = outputs["fused"]["alpha"]
        S = alpha.sum(dim=-1, keepdim=True)
        all_probs.append((alpha / S).cpu().numpy())
        all_targets.append(labels.numpy())
        all_unc.append(outputs["fused"]["uncertainty"].cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    unc = np.concatenate(all_unc)
    metrics = compute_all_metrics(probs, targets, multi_label=multi_label)
    metrics["mean_uncertainty"] = float(unc.mean())
    return metrics


def train_and_eval(model, cfg, train_loader, val_loader, test_loader,
                   device, multi_label, run_dir):
    """Train a model, evaluate on test set, return metrics."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["logging"]["output_dir"] = run_dir
    os.makedirs(run_dir, exist_ok=True)

    trainer = Trainer(model, train_loader, val_loader, run_cfg, device)
    history = trainer.train()

    # Load best checkpoint for evaluation (not last epoch)
    best_ckpt_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        sd = ckpt["model_state_dict"]
        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
        logger.info(f"Loaded best model from epoch {ckpt['epoch']} (val={ckpt['best_val_metric']:.4f})")

    metrics = evaluate_model(model, test_loader, device, multi_label)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, history


# ═══════════════════════════════════════════════════════════════════════ #
#  Ablation variants (simplified — key ones only)
# ═══════════════════════════════════════════════════════════════════════ #

def build_ablation_variants(cfg):
    """Return dict of {name: modified_cfg} for ablation experiments."""
    variants = {}

    # 1. Fusion method
    for method in ["dempster_shafer", "average", "gating"]:
        c = copy.deepcopy(cfg)
        c["model"]["fusion"]["method"] = method
        variants[f"ablation_fusion_{method}"] = c

    # 2. Expert streams
    variants["ablation_stream_global_only"] = copy.deepcopy(cfg)
    variants["ablation_stream_global_only"]["_stream"] = "global_only"
    variants["ablation_stream_local_only"] = copy.deepcopy(cfg)
    variants["ablation_stream_local_only"]["_stream"] = "local_only"

    # 3. KL annealing
    c_no_kl = copy.deepcopy(cfg)
    c_no_kl["training"]["loss"]["kl_weight"] = 0.0
    variants["ablation_kl_disabled"] = c_no_kl

    c_no_anneal = copy.deepcopy(cfg)
    c_no_anneal["training"]["loss"]["annealing_epochs"] = 0
    variants["ablation_kl_no_annealing"] = c_no_anneal

    # 4. Attention crop off
    c_no_crop = copy.deepcopy(cfg)
    c_no_crop["model"]["local_expert"]["use_attention_crop"] = False
    variants["ablation_attn_crop_off"] = c_no_crop

    # 5. Loss variant: EDL-mse
    c_mse = copy.deepcopy(cfg)
    c_mse["training"]["loss"]["type"] = "mse"
    variants["ablation_loss_edl_mse"] = c_mse

    return variants


def build_model_for_ablation(cfg, device):
    num_classes = get_num_classes(cfg)
    stream = cfg.pop("_stream", None)
    if stream == "global_only":
        return EDLSingleScale(
            backbone=cfg["model"]["global_expert"]["backbone"],
            pretrained=cfg["model"]["global_expert"]["pretrained"],
            num_classes=num_classes,
        ).to(device)
    elif stream == "local_only":
        return EDLSingleScale(
            backbone=cfg["model"]["local_expert"]["backbone"],
            pretrained=cfg["model"]["local_expert"]["pretrained"],
            num_classes=num_classes,
        ).to(device)
    else:
        return build_model(cfg).to(device)


# ═══════════════════════════════════════════════════════════════════════ #
#  Main
# ═══════════════════════════════════════════════════════════════════════ #

def parse_args():
    p = argparse.ArgumentParser(description="Run full TMS-EDL experiment pipeline")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--report_file", type=str, default="outputs/experiment_report.txt",
                   help="Path to the final summary report file")
    p.add_argument("--skip_ablations", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    cfg["seed"] = args.seed

    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    multi_label = get_multi_label(cfg)
    num_classes = get_num_classes(cfg)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {cfg['data']['dataset']} | Classes: {num_classes}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | "
                f"Test: {len(test_loader.dataset)}")

    all_results = {}
    timings = {}
    t_total_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # 1. Train & evaluate TMS-EDL (Ours)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  [1] Training TMS-EDL (Ours)")
    logger.info("=" * 60)
    set_seed(cfg["seed"])
    t0 = time.time()
    model_ours = build_model(cfg).to(device)
    metrics_ours, _ = train_and_eval(
        model_ours, cfg, train_loader, val_loader, test_loader,
        device, multi_label, os.path.join(output_dir, "tms_edl"),
    )
    timings["tms_edl"] = time.time() - t0
    all_results["TMS-EDL (Ours)"] = metrics_ours
    logger.info(f"  TMS-EDL done: AUC={metrics_ours['auc']:.4f} ACC={metrics_ours['acc']:.4f} "
                f"ECE={metrics_ours['ece']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 2. Ablation studies
    # ══════════════════════════════════════════════════════════════════
    ablation_results = {}
    if not args.skip_ablations:
        logger.info("\n" + "=" * 60)
        logger.info("  [2] Running Ablation Studies")
        logger.info("=" * 60)

        variants = build_ablation_variants(cfg)
        for vname, vcfg in variants.items():
            logger.info(f"\n  ── {vname} ──")
            set_seed(cfg["seed"])
            t0 = time.time()
            model_a = build_model_for_ablation(vcfg, device)
            metrics_a, _ = train_and_eval(
                model_a, vcfg, train_loader, val_loader, test_loader,
                device, multi_label, os.path.join(output_dir, "ablations", vname),
            )
            timings[vname] = time.time() - t0
            ablation_results[vname] = metrics_a
            logger.info(f"  {vname}: AUC={metrics_a['auc']:.4f} ACC={metrics_a['acc']:.4f} "
                        f"ECE={metrics_a['ece']:.4f}")
            del model_a
            torch.cuda.empty_cache()

    total_time = time.time() - t_total_start

    # ══════════════════════════════════════════════════════════════════
    # 4. Generate summary report
    # ══════════════════════════════════════════════════════════════════
    report_path = args.report_file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("  TMS-EDL Experiment Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Dataset: {cfg['data']['dataset']}")
    lines.append(f"  Epochs: {cfg['training']['epochs']}")
    lines.append(f"  Device: {device}")
    lines.append(f"  Total time: {total_time/60:.1f} min")
    lines.append("=" * 80)

    # ── Main results table ────────────────────────────────────────────
    lines.append("")
    lines.append("┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("│                        MAIN RESULTS TABLE                               │")
    lines.append("├──────────────────────────┬─────────┬─────────┬─────────┬────────────────┤")
    lines.append("│ Method                   │   AUC   │   ACC   │   ECE   │ Uncertainty    │")
    lines.append("├──────────────────────────┼─────────┼─────────┼─────────┼────────────────┤")

    for name, m in all_results.items():
        auc = f"{m['auc']:.4f}"
        acc = f"{m['acc']:.4f}"
        ece = f"{m['ece']:.4f}"
        unc = f"{m.get('mean_uncertainty', 0):.4f}"
        marker = " ***" if "Ours" in name else ""
        lines.append(f"│ {name:<24s} │ {auc:>7s} │ {acc:>7s} │ {ece:>7s} │ {unc:>7s}{marker:<7s} │")

    lines.append("└──────────────────────────┴─────────┴─────────┴─────────┴────────────────┘")
    lines.append("  *** = Our method (TMS-EDL)")

    # ── Ablation table ────────────────────────────────────────────────
    if ablation_results:
        lines.append("")
        lines.append("┌─────────────────────────────────────────────────────────────────────────┐")
        lines.append("│                        ABLATION STUDY RESULTS                           │")
        lines.append("├──────────────────────────────────┬─────────┬─────────┬─────────────────┤")
        lines.append("│ Variant                          │   AUC   │   ACC   │   ECE           │")
        lines.append("├──────────────────────────────────┼─────────┼─────────┼─────────────────┤")

        for vname, m in ablation_results.items():
            short = vname.replace("ablation_", "")
            auc = f"{m['auc']:.4f}"
            acc = f"{m['acc']:.4f}"
            ece = f"{m['ece']:.4f}"
            lines.append(f"│ {short:<32s} │ {auc:>7s} │ {acc:>7s} │ {ece:>7s}         │")

        lines.append("└──────────────────────────────────┴─────────┴─────────┴─────────────────┘")

    # ── Timing ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("TRAINING TIME:")
    for name, t in timings.items():
        lines.append(f"  {name:<35s}  {t/60:>6.1f} min")
    lines.append(f"  {'TOTAL':<35s}  {total_time/60:>6.1f} min")

    # ── Key findings ──────────────────────────────────────────────────
    lines.append("")
    lines.append("KEY FINDINGS:")
    if all_results:
        best_auc_name = max(all_results, key=lambda k: all_results[k]["auc"])
        best_ece_name = min(all_results, key=lambda k: all_results[k]["ece"])
        lines.append(f"  Best AUC:  {best_auc_name} ({all_results[best_auc_name]['auc']:.4f})")
        lines.append(f"  Best ECE:  {best_ece_name} ({all_results[best_ece_name]['ece']:.4f})")

        ours = all_results.get("TMS-EDL (Ours)")
        if ours:
            lines.append(f"  TMS-EDL:   AUC={ours['auc']:.4f}  ACC={ours['acc']:.4f}  ECE={ours['ece']:.4f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  Output directory: " + os.path.abspath(output_dir))
    lines.append("  This report:      " + os.path.abspath(report_path))
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    # Write report
    with open(report_path, "w") as f:
        f.write(report_text)

    # Also save raw JSON
    json_path = report_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "main_results": all_results,
            "ablation_results": ablation_results,
            "timings": timings,
            "config": cfg,
            "total_time_sec": total_time,
        }, f, indent=2)

    # Print to console
    print("\n" + report_text)
    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"JSON saved to:   {json_path}")


if __name__ == "__main__":
    main()
