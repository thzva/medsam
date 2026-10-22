"""Re-evaluate all saved best checkpoints and regenerate experiment report.

Loads best_model.pth from each run directory, evaluates on the test set,
and writes a corrected experiment_report.txt without any retraining.

Usage:
    python scripts/re_evaluate.py --config config/default.yaml
"""

import os
import sys
import json
import copy
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.amp import autocast

from src.training.utils import load_config
from src.data.builder import build_dataloaders
from src.models.builder import build_model
from src.models.baselines import EDLSingleScale
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_num_classes(cfg):
    ds = cfg["data"]["dataset"]
    if ds == "nih_chestxray":
        return cfg["data"]["nih"]["num_classes"]
    return cfg["data"]["camelyon16"]["num_classes"]


def get_multi_label(cfg):
    ds = cfg["data"]["dataset"]
    return ds == "nih_chestxray" and cfg["data"]["nih"].get("multi_label", True)


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, model, test_loader, device, multi_label):
    """Load checkpoint into model and evaluate on test set."""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.eval()

    best_epoch = ckpt.get("epoch", "?")
    best_val = ckpt.get("best_val_metric", float("nan"))
    logger.info(f"  Loaded epoch={best_epoch}, val_auc={best_val:.4f}  <- {ckpt_path}")

    all_probs, all_targets, all_unc = [], [], []
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)

        # Support both single-stream (EDLSingleScale) and dual-stream (TMSEDL)
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
    unc_arr = np.concatenate(all_unc)
    metrics = compute_all_metrics(probs, targets, multi_label=multi_label)
    metrics["mean_uncertainty"] = float(unc_arr.mean())
    metrics["best_epoch"] = best_epoch
    metrics["best_val_auc"] = float(best_val)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--report_file", type=str, default="outputs/experiment_report_corrected.txt")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda"))
    multi_label = get_multi_label(cfg)
    num_classes = get_num_classes(cfg)
    output_dir = Path(args.output_dir)

    logger.info("Building test loader…")
    _, _, test_loader = build_dataloaders(cfg)
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    all_results = {}  # name -> metrics dict

    # ── 1. TMS-EDL main model ─────────────────────────────────────────
    ckpt_path = output_dir / "tms_edl" / "best_model.pth"
    if ckpt_path.exists():
        logger.info("\n[TMS-EDL (Ours)]")
        model = build_model(cfg).to(device)
        metrics = evaluate_checkpoint(ckpt_path, model, test_loader, device, multi_label)
        all_results["TMS-EDL (Ours)"] = metrics
        logger.info(f"  AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  ECE={metrics['ece']:.4f}")
        # Update stored test_metrics.json with corrected values
        with open(output_dir / "tms_edl" / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        del model
        torch.cuda.empty_cache()
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}, skipping TMS-EDL")

    # ── 2. Ablations ─────────────────────────────────────────────────
    ablation_dir = output_dir / "ablations"
    ablation_results = {}

    # Map run dir name -> (model builder fn, config modifier)
    def cfg_fusion(method):
        c = copy.deepcopy(cfg)
        c["model"]["fusion"]["method"] = method
        return c

    def cfg_stream(stream):
        c = copy.deepcopy(cfg)
        c["_stream"] = stream
        return c

    def cfg_kl_disabled():
        c = copy.deepcopy(cfg)
        c["training"]["loss"]["kl_weight"] = 0.0
        return c

    def cfg_kl_no_annealing():
        c = copy.deepcopy(cfg)
        c["training"]["loss"]["annealing_epochs"] = 0
        return c

    def cfg_attn_crop_off():
        c = copy.deepcopy(cfg)
        c["model"]["local_expert"]["use_attention_crop"] = False
        return c

    def cfg_loss_mse():
        c = copy.deepcopy(cfg)
        c["training"]["loss"]["type"] = "mse"
        return c

    ablation_specs = {
        "ablation_fusion_dempster_shafer": cfg_fusion("dempster_shafer"),
        "ablation_fusion_average":         cfg_fusion("average"),
        "ablation_fusion_gating":          cfg_fusion("gating"),
        "ablation_stream_global_only":     cfg_stream("global_only"),
        "ablation_stream_local_only":      cfg_stream("local_only"),
        "ablation_kl_disabled":            cfg_kl_disabled(),
        "ablation_kl_no_annealing":        cfg_kl_no_annealing(),
        "ablation_attn_crop_off":          cfg_attn_crop_off(),
        "ablation_loss_edl_mse":           cfg_loss_mse(),
    }

    for vname, vcfg in ablation_specs.items():
        ckpt_path = ablation_dir / vname / "best_model.pth"
        if not ckpt_path.exists():
            logger.warning(f"  No checkpoint: {ckpt_path}, skipping")
            continue

        logger.info(f"\n[Ablation: {vname}]")
        stream = vcfg.pop("_stream", None)
        if stream == "global_only":
            model = EDLSingleScale(
                backbone=cfg["model"]["global_expert"]["backbone"],
                pretrained=False,
                num_classes=num_classes,
            ).to(device)
        elif stream == "local_only":
            model = EDLSingleScale(
                backbone=cfg["model"]["local_expert"]["backbone"],
                pretrained=False,
                num_classes=num_classes,
            ).to(device)
        else:
            model = build_model(vcfg).to(device)

        metrics = evaluate_checkpoint(ckpt_path, model, test_loader, device, multi_label)
        ablation_results[vname] = metrics
        logger.info(f"  AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  ECE={metrics['ece']:.4f}")
        # Update stored test_metrics.json
        with open(ablation_dir / vname / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        del model
        torch.cuda.empty_cache()

    # ── 3. Generate report ────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("  TMS-EDL Experiment Report  [CORRECTED — evaluated from best checkpoints]")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Dataset: {cfg['data']['dataset']}")
    lines.append(f"  Note: All metrics use best_model.pth (early-stopped checkpoint)")
    lines.append("=" * 80)

    lines.append("")
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│                          MAIN RESULTS TABLE                                 │")
    lines.append("├──────────────────────────┬────────┬────────┬────────┬────────┬──────────────┤")
    lines.append("│ Method                   │  AUC   │  ACC   │  ECE   │BestEp. │  Val AUC     │")
    lines.append("├──────────────────────────┼────────┼────────┼────────┼────────┼──────────────┤")
    for name, m in all_results.items():
        marker = " ***" if "Ours" in name else ""
        lines.append(
            f"│ {name:<24s} │ {m['auc']:.4f} │ {m['acc']:.4f} │ {m['ece']:.4f} │"
            f" {str(m.get('best_epoch','?')):>6s} │ {m.get('best_val_auc',0):.4f}{marker:<7s} │"
        )
    lines.append("└──────────────────────────┴────────┴────────┴────────┴────────┴──────────────┘")
    lines.append("  *** = Our method (TMS-EDL)")

    if ablation_results:
        lines.append("")
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│                         ABLATION STUDY RESULTS                              │")
        lines.append("├──────────────────────────────────┬────────┬────────┬────────┬───────────────┤")
        lines.append("│ Variant                          │  AUC   │  ACC   │  ECE   │ BestEpoch     │")
        lines.append("├──────────────────────────────────┼────────┼────────┼────────┼───────────────┤")
        for vname, m in ablation_results.items():
            short = vname.replace("ablation_", "")
            lines.append(
                f"│ {short:<32s} │ {m['auc']:.4f} │ {m['acc']:.4f} │ {m['ece']:.4f} │"
                f" ep={str(m.get('best_epoch','?')):<10s} │"
            )
        lines.append("└──────────────────────────────────┴────────┴────────┴────────┴───────────────┘")

    lines.append("")
    lines.append("=" * 80)
    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"\nReport saved to: {report_path}")

    json_path = str(report_path).replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump({"main_results": all_results, "ablation_results": ablation_results}, f, indent=2)
    logger.info(f"JSON saved to:   {json_path}")


if __name__ == "__main__":
    main()
