"""Compare all trained models (ours + baselines) on the test set.

Produces the main results table from the proposal:
  Method | Global | Local | Multi-scale | UQ | Fusion | AUC | ACC | ECE | OOD AUROC
"""

import os
import sys
import argparse
import logging
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.utils import set_seed, load_config
from src.data.builder import build_dataloaders
from src.evaluation.metrics import compute_all_metrics, compute_ood_auroc
from src.evaluation.calibration import reliability_diagram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Model metadata for the results table ──────────────────────────────── #
MODEL_META = {
    "resnet50_ce":          {"Global": "-",   "Local": "-",       "MS": "-", "UQ": "None",       "Fusion": "-"},
    "vit_b16_ce":           {"Global": "ViT", "Local": "-",       "MS": "-", "UQ": "None",       "Fusion": "-"},
    "densenet121_ce":       {"Global": "-",   "Local": "-",       "MS": "-", "UQ": "None",       "Fusion": "-"},
    "resnet50_temp_scaling":{"Global": "-",   "Local": "-",       "MS": "-", "UQ": "TempScale",  "Fusion": "-"},
    "resnet50_mc_dropout":  {"Global": "-",   "Local": "-",       "MS": "-", "UQ": "MC Dropout", "Fusion": "-"},
    "deep_ensemble":        {"Global": "-",   "Local": "-",       "MS": "-", "UQ": "Ensemble",   "Fusion": "-"},
    "edl_single_resnet50":  {"Global": "-",   "Local": "-",       "MS": "-", "UQ": "EDL",        "Fusion": "-"},
    "multiscale_avg_prob":  {"Global": "ViT", "Local": "ResNet",  "MS": "✓", "UQ": "None",       "Fusion": "Avg prob"},
    "tms_edl":              {"Global": "ViT", "Local": "ResNet",  "MS": "✓", "UQ": "EDL",        "Fusion": "D-S"},
}


def collect_results(results_dir: str) -> dict:
    """Scan results directory for metrics.json files."""
    results = {}
    # Look for metrics.json in subdirectories
    for metrics_file in glob.glob(os.path.join(results_dir, "**", "metrics.json"), recursive=True):
        parent = os.path.basename(os.path.dirname(metrics_file))
        with open(metrics_file) as f:
            results[parent] = json.load(f)
    # Also look for test_metrics.json (from ablation runs)
    for metrics_file in glob.glob(os.path.join(results_dir, "**", "test_metrics.json"), recursive=True):
        parent = os.path.basename(os.path.dirname(metrics_file))
        if parent not in results:
            with open(metrics_file) as f:
                results[parent] = json.load(f)
    return results


def print_main_table(results: dict, output_path: str = None):
    """Print the main comparison table."""
    header = f"{'Method':<30s} {'Global':>6s} {'Local':>8s} {'MS':>3s} {'UQ':>12s} {'Fusion':>8s} {'AUC':>7s} {'ACC':>7s} {'ECE':>7s} {'OOD':>7s}"
    sep = "-" * len(header)

    lines = [sep, header, sep]

    for name in MODEL_META:
        meta = MODEL_META[name]
        m = results.get(name, {})
        auc = m.get("auc", "-")
        acc = m.get("acc", "-")
        ece = m.get("ece", "-")
        ood = m.get("ood_auroc", "-")

        auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        ece_s = f"{ece:.4f}" if isinstance(ece, float) else str(ece)
        ood_s = f"{ood:.4f}" if isinstance(ood, float) else str(ood)

        prefix = "**" if name == "tms_edl" else "  "
        line = (f"{prefix}{name:<28s} {meta['Global']:>6s} {meta['Local']:>8s} "
                f"{meta['MS']:>3s} {meta['UQ']:>12s} {meta['Fusion']:>8s} "
                f"{auc_s:>7s} {acc_s:>7s} {ece_s:>7s} {ood_s:>7s}")
        lines.append(line)

    lines.append(sep)
    table = "\n".join(lines)
    logger.info(f"\n{table}")

    if output_path:
        with open(output_path, "w") as f:
            f.write(table)


def plot_comparison_bar(results: dict, save_path: str):
    """Bar chart comparing AUC, ACC, ECE across methods."""
    methods = [n for n in MODEL_META if n in results]
    if not methods:
        return

    auc_vals = [results[n].get("auc", 0) for n in methods]
    acc_vals = [results[n].get("acc", 0) for n in methods]
    ece_vals = [results[n].get("ece", 0) for n in methods]

    x = np.arange(len(methods))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # AUC & ACC
    ax1.bar(x - width/2, auc_vals, width, label="AUC", alpha=0.8)
    ax1.bar(x + width/2, acc_vals, width, label="ACC", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Score")
    ax1.set_title("AUC & ACC Comparison")
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # ECE (lower is better)
    colors = ["#e74c3c" if n == "tms_edl" else "#3498db" for n in methods]
    ax2.bar(x, ece_vals, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("ECE (lower is better)")
    ax2.set_title("Calibration Error Comparison")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison plot saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare all models")
    parser.add_argument("--results_dir", type=str, default="outputs",
                        help="Root directory containing all model outputs")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = collect_results(args.results_dir)
    logger.info(f"Found results for {len(results)} models: {list(results.keys())}")

    # Print main table
    print_main_table(results, os.path.join(args.output_dir, "main_table.txt"))

    # Save as JSON
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison
    if results:
        plot_comparison_bar(results, os.path.join(args.output_dir, "comparison.png"))

    logger.info(f"Comparison results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
