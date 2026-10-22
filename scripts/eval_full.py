"""Full post-training evaluation: per-class AUC + OOD AUROC.

Loads a TMS-EDL (or ablation) checkpoint, runs it on:
  1. NIH ChestX-ray test set     -> per-class AUC, overall AUC/ACC/ECE/u
  2. CT liver OOD set            -> OOD AUROC
  3. ImageNet OOD set            -> OOD AUROC

Output: JSON next to the checkpoint named `eval_full.json`.

Single-GPU eval. Pass --device to pick GPU; default is cuda:3 (avoid training GPUs).
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.training.utils import load_config
from src.data.builder import build_dataloaders
from src.data.ood_dataset import OODDataset
from src.data.nih_chestxray import LABEL_NAMES
from src.evaluation.metrics import compute_all_metrics, compute_ood_auroc

# Reuse model factory from train_ddp
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_ddp import build_model_for_mode, ABLATION_MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_probs_unc(model, loader, device, use_amp=True):
    """Return (probs[N,C], uncertainty[N], targets[N,C] or None)."""
    model.eval()
    all_probs, all_unc, all_targets = [], [], []
    has_targets = True
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch.get("label")
        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
        if "fused" in outputs:
            alpha = outputs["fused"]["alpha"]
            unc = outputs["fused"]["uncertainty"]
        else:
            alpha = outputs["alpha"]
            unc = outputs["uncertainty"]
        S = alpha.sum(dim=-1, keepdim=True)
        all_probs.append((alpha / S).cpu().numpy())
        all_unc.append(unc.cpu().numpy())
        if labels is None or (hasattr(labels, "ndim") and labels.dim() == 1 and (labels < 0).all()):
            has_targets = False
        else:
            all_targets.append(labels.numpy())
    probs = np.concatenate(all_probs, axis=0)
    unc = np.concatenate(all_unc, axis=0)
    targets = np.concatenate(all_targets, axis=0) if has_targets else None
    return probs, unc, targets


def evaluate_checkpoint(mode, checkpoint_path, cfg, device, ood_dirs):
    """Evaluate one checkpoint. Returns the full metrics dict."""
    logger.info(f"\n{'='*60}\n  Mode: {mode}\n  Ckpt: {checkpoint_path}\n{'='*60}")

    # Build the model in the same way training did
    model, _ = build_model_for_mode(mode, cfg, device)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    logger.info(f"  Loaded epoch={ckpt.get('epoch','?')} val_auc={ckpt.get('best_val_metric',float('nan')):.4f}")

    # Build NIH test loader
    _, _, test_loader = build_dataloaders(cfg, rank=-1, world_size=1)

    # In-distribution eval
    probs, unc_id, targets = collect_probs_unc(model, test_loader, device)
    metrics = compute_all_metrics(
        probs, targets,
        multi_label=True,
        n_bins=cfg["evaluation"].get("ece_bins", 15),
        return_per_class=True,
    )
    metrics["mean_uncertainty_id"] = float(unc_id.mean())
    metrics["best_epoch"] = int(ckpt.get("epoch", -1))
    metrics["val_auc_at_best"] = float(ckpt.get("best_val_metric", float("nan")))

    # Per-class AUC: replace integer keys with disease names
    if "per_class_auc" in metrics:
        named = {LABEL_NAMES[k]: v for k, v in metrics["per_class_auc"].items()}
        metrics["per_class_auc"] = named

    logger.info(
        f"  ID:  AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  "
        f"ECE={metrics['ece']:.4f}  u_mean={metrics['mean_uncertainty_id']:.4f}"
    )

    # OOD eval
    img_size = cfg["data"].get("image_size", 224)
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    ood_results = {}
    unc_id_finite = unc_id[np.isfinite(unc_id)]
    for name, path in ood_dirs.items():
        if not Path(path).exists():
            logger.warning(f"  OOD '{name}' missing path: {path}, skipping")
            continue
        try:
            ds = OODDataset(path, image_size=img_size, max_samples=1000)
            if len(ds) == 0:
                logger.warning(f"  OOD '{name}' empty, skipping")
                continue
            loader = DataLoader(ds, batch_size=bs, num_workers=nw, pin_memory=True)
            _, unc_ood, _ = collect_probs_unc(model, loader, device)
            n_raw = len(unc_ood)
            finite_mask = np.isfinite(unc_ood)
            n_finite = int(finite_mask.sum())
            n_nan = n_raw - n_finite
            unc_ood_finite = unc_ood[finite_mask]
            if n_finite == 0:
                logger.warning(f"  OOD '{name}': all {n_raw} samples produced NaN uncertainty, skipping AUROC")
                ood_results[name] = {
                    "n_samples": n_raw, "n_nan": n_nan, "error": "all_nan",
                }
                continue
            ood_auroc = compute_ood_auroc(unc_id_finite, unc_ood_finite)
            ood_results[name] = {
                "n_samples": n_raw,
                "n_nan": n_nan,
                "mean_uncertainty": float(unc_ood_finite.mean()),
                "ood_auroc_vs_id": ood_auroc,
            }
            logger.info(
                f"  OOD '{name}': N={n_raw} ({n_nan} NaN dropped)  "
                f"u_mean={unc_ood_finite.mean():.4f}  AUROC={ood_auroc:.4f}"
            )
        except Exception as e:
            logger.warning(f"  OOD '{name}' failed: {e}")
            ood_results[name] = {"error": str(e)}

    metrics["ood"] = ood_results
    return metrics


def discover_modes(outputs_dir):
    """Find all (mode, checkpoint_path) pairs ready to evaluate."""
    outputs = Path(outputs_dir)
    pairs = []
    # Main TMS-EDL
    main_ckpt = outputs / "tms_edl" / "best_model.pth"
    if main_ckpt.exists():
        pairs.append(("tms_edl", main_ckpt))
    # Ablations
    abl_root = outputs / "ablations"
    if abl_root.exists():
        for d in sorted(abl_root.glob("ablation_*")):
            mode = d.name.replace("ablation_", "")
            ckpt = d / "best_model.pth"
            if ckpt.exists():
                pairs.append((mode, ckpt))
    return pairs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--mode", default=None,
                    help="If set, only evaluate this mode. Default: all available.")
    ap.add_argument("--checkpoint", default=None,
                    help="If set, evaluate this checkpoint (requires --mode).")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--ood_ct", default="data/ood_ct_liver")
    ap.add_argument("--ood_imagenet", default="data/ood_imagenet")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ood_dirs = {"ct_liver": args.ood_ct, "imagenet": args.ood_imagenet}

    if args.checkpoint and args.mode:
        pairs = [(args.mode, Path(args.checkpoint))]
    elif args.mode:
        # Locate checkpoint by convention
        base = Path(args.outputs_dir)
        path = base / "tms_edl" / "best_model.pth" if args.mode == "tms_edl" \
            else base / "ablations" / f"ablation_{args.mode}" / "best_model.pth"
        pairs = [(args.mode, path)]
    else:
        pairs = discover_modes(args.outputs_dir)

    summary = {}
    for mode, ckpt in pairs:
        try:
            metrics = evaluate_checkpoint(mode, ckpt, cfg, device, ood_dirs)
            out_path = ckpt.parent / "eval_full.json"
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"  -> wrote {out_path}")
            summary[mode] = {
                "auc": metrics["auc"], "acc": metrics["acc"], "ece": metrics["ece"],
                "best_epoch": metrics["best_epoch"],
                "ood_ct_liver_auroc": metrics["ood"].get("ct_liver", {}).get("ood_auroc_vs_id"),
                "ood_imagenet_auroc": metrics["ood"].get("imagenet", {}).get("ood_auroc_vs_id"),
            }
        except Exception as e:
            logger.exception(f"FAILED on {mode}: {e}")
            summary[mode] = {"error": str(e)}

    out_summary = Path(args.outputs_dir) / "eval_full_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary: {out_summary}")


if __name__ == "__main__":
    main()
