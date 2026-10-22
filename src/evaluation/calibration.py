"""Calibration visualization utilities."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def reliability_diagram(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """Plot a reliability (calibration) diagram.

    Args:
        probs: (N, K) predicted probabilities.
        targets: (N,) or (N, K).
        n_bins: number of bins.
        save_path: if given, save figure to this path.
        title: plot title.
    Returns:
        matplotlib Figure.
    """
    if targets.ndim == 2:
        targets = targets.argmax(axis=1)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accs.append(0)
            bin_counts.append(0)
        else:
            bin_confs.append(confidences[mask].mean())
            bin_accs.append(accuracies[mask].mean())
            bin_counts.append(mask.sum())

    bin_confs = np.array(bin_confs)
    bin_accs = np.array(bin_accs)
    bin_counts = np.array(bin_counts)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.6,
            edgecolor="black", label="Model")
    ax1.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of confidences
    ax2.bar(bin_confs, bin_counts, width=1.0 / n_bins, alpha=0.6, edgecolor="black")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
