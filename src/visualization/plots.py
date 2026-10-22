"""Visualization utilities for TMS-EDL experiments."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
):
    """Plot loss and metric curves over epochs.

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'val_auc', 'val_ece'.
        save_path: output image path.
        title: figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC
    if "val_auc" in history:
        axes[1].plot(epochs, history["val_auc"], color="green", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Validation AUC")
        axes[1].grid(True, alpha=0.3)

    # ECE
    if "val_ece" in history:
        axes[2].plot(epochs, history["val_ece"], color="red", linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("ECE")
        axes[2].set_title("Validation ECE (lower is better)")
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_histogram(
    id_uncertainty: np.ndarray,
    ood_uncertainty: Optional[np.ndarray] = None,
    save_path: str = "uncertainty_hist.png",
    title: str = "Uncertainty Distribution",
):
    """Histogram of uncertainty scores for ID and OOD samples.

    Args:
        id_uncertainty: (N,) uncertainty for in-distribution samples.
        ood_uncertainty: (N,) uncertainty for OOD samples (optional).
        save_path: output image path.
        title: figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(id_uncertainty, bins=50, alpha=0.6, label="In-Distribution",
            color="#3498db", density=True, edgecolor="white")
    if ood_uncertainty is not None:
        ax.hist(ood_uncertainty, bins=50, alpha=0.6, label="OOD",
                color="#e74c3c", density=True, edgecolor="white")

    ax.set_xlabel("Uncertainty Score (u = K / S)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_table(
    results: Dict[str, Dict[str, float]],
    save_path: str = "ablation_table.png",
    title: str = "Ablation Study Results",
    metrics: List[str] = None,
):
    """Render ablation results as a formatted table image.

    Args:
        results: {variant_name: {metric_name: value}}.
        save_path: output image path.
        title: figure title.
        metrics: which metrics to show (default: auc, acc, ece).
    """
    if metrics is None:
        metrics = ["auc", "acc", "ece"]

    variants = list(results.keys())
    cell_text = []
    for v in variants:
        row = []
        for m in metrics:
            val = results[v].get(m, "-")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 2.5), max(3, len(variants) * 0.5 + 1.5)))
    ax.axis("off")

    col_labels = [m.upper() for m in metrics]
    table = ax.table(
        cellText=cell_text,
        rowLabels=variants,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight best values
    for col_idx, m in enumerate(metrics):
        vals = []
        for v in variants:
            val = results[v].get(m, None)
            if isinstance(val, (int, float)):
                vals.append(val)
            else:
                vals.append(None)

        if all(v is None for v in vals):
            continue

        # ECE: lower is better; others: higher is better
        if m == "ece":
            best_idx = min(range(len(vals)), key=lambda i: vals[i] if vals[i] is not None else float("inf"))
        else:
            best_idx = max(range(len(vals)), key=lambda i: vals[i] if vals[i] is not None else -float("inf"))

        cell = table[best_idx + 1, col_idx]
        cell.set_facecolor("#d5f5e3")
        cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_map(
    image: np.ndarray,
    attn_weights: np.ndarray,
    patch_coords: Optional[List] = None,
    save_path: str = "attention_map.png",
    title: str = "Attention-based ROI Selection",
):
    """Visualize attention-based patch selection on an image.

    Args:
        image: (H, W, 3) original image (uint8 or float [0,1]).
        attn_weights: (K,) attention weights for top-K patches.
        patch_coords: list of (y1, x1, y2, x2) for each patch.
        save_path: output image path.
        title: figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    ax.imshow(image)

    if patch_coords is not None:
        colors = plt.cm.hot(np.linspace(0.3, 0.9, len(attn_weights)))
        for i, (y1, x1, y2, x2) in enumerate(patch_coords):
            w = attn_weights[i]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2 + w * 4,
                edgecolor=colors[i],
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"w={w:.2f}", color=colors[i],
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_belief_bar(
    belief: np.ndarray,
    uncertainty: float,
    class_names: List[str],
    save_path: str = "belief_bar.png",
    title: str = "Belief Mass Distribution",
):
    """Bar chart of belief masses with uncertainty indicator.

    Args:
        belief: (K,) belief mass vector.
        uncertainty: scalar uncertainty score.
        class_names: list of class names.
        save_path: output image path.
        title: figure title.
    """
    K = len(belief)
    fig, ax = plt.subplots(figsize=(max(8, K * 0.6), 5))

    # Sort by belief value
    order = np.argsort(belief)[::-1]
    top_n = min(15, K)  # show top 15 classes
    order = order[:top_n]

    names = [class_names[i] for i in order]
    vals = belief[order]

    bars = ax.barh(range(len(names)), vals, color="#3498db", alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Belief Mass", fontsize=11)
    ax.set_title(f"{title}\nUncertainty: {uncertainty:.4f}", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals.max() * 1.2, 0.1))
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ece_comparison(
    model_names: List[str],
    ece_values: List[float],
    save_path: str = "ece_comparison.png",
):
    """Bar chart comparing ECE across models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#e74c3c" if "tms_edl" in n.lower() or "ours" in n.lower()
              else "#3498db" for n in model_names]

    bars = ax.bar(range(len(model_names)), ece_values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("ECE (lower is better)", fontsize=11)
    ax.set_title("Calibration Error Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, ece_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
