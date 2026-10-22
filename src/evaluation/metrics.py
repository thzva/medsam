"""Evaluation metrics: AUC, ACC, ECE, OOD AUROC."""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, accuracy_score


def compute_auc(
    probs: np.ndarray,
    targets: np.ndarray,
    multi_label: bool = False,
) -> float:
    """Compute AUC (macro-averaged for multi-label / multi-class).

    Args:
        probs: (N, K) predicted probabilities.
        targets: (N, K) one-hot / multi-hot  or  (N,) class indices.
    """
    if targets.ndim == 1:
        # Multi-class single-label
        if probs.shape[1] == 2:
            return float(roc_auc_score(targets, probs[:, 1]))
        return float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    else:
        # Multi-label: per-class AUC then macro average
        aucs = []
        for k in range(targets.shape[1]):
            col = targets[:, k]
            if col.sum() == 0 or col.sum() == len(col):
                continue  # skip degenerate columns
            aucs.append(roc_auc_score(col, probs[:, k]))
        return float(np.mean(aucs)) if aucs else 0.0


def compute_per_class_auc(probs: np.ndarray, targets: np.ndarray) -> Dict[int, Optional[float]]:
    """Per-class AUC for multi-label targets.

    Returns {class_idx: auc_value_or_None}. None for classes with all 0s / all 1s.
    """
    per_class = {}
    for k in range(targets.shape[1]):
        col = targets[:, k]
        if col.sum() == 0 or col.sum() == len(col):
            per_class[k] = None
        else:
            per_class[k] = float(roc_auc_score(col, probs[:, k]))
    return per_class


def compute_accuracy(
    probs: np.ndarray,
    targets: np.ndarray,
    multi_label: bool = False,
    threshold: float = 0.5,
) -> float:
    """Compute accuracy.

    For multi-label: per-sample exact match is too strict, so we use
    per-label accuracy averaged over classes.
    """
    if multi_label or (targets.ndim == 2 and targets.shape[1] > 1):
        preds = (probs >= threshold).astype(int)
        if targets.ndim == 1:
            targets_oh = np.zeros_like(probs)
            targets_oh[np.arange(len(targets)), targets] = 1
            targets = targets_oh
        # Per-label accuracy, macro averaged
        accs = []
        for k in range(targets.shape[1]):
            accs.append(accuracy_score(targets[:, k], preds[:, k]))
        return float(np.mean(accs))
    else:
        preds = probs.argmax(axis=1)
        if targets.ndim == 2:
            targets = targets.argmax(axis=1)
        return float(accuracy_score(targets, preds))


def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
    multi_label: bool = False,
) -> float:
    """Expected Calibration Error.

    For multi-class single-label: standard ECE on the max-confidence class.
    For multi-label: average ECE across labels.

    Args:
        probs: (N, K) predicted probabilities.
        targets: (N, K) or (N,).
        n_bins: number of confidence bins.
    Returns:
        ECE value (lower is better).
    """
    if multi_label or (targets.ndim == 2 and targets.shape[1] > 1):
        eces = []
        if targets.ndim == 1:
            targets_oh = np.zeros_like(probs)
            targets_oh[np.arange(len(targets)), targets] = 1
            targets = targets_oh
        for k in range(targets.shape[1]):
            eces.append(_binary_ece(probs[:, k], targets[:, k], n_bins))
        return float(np.mean(eces))
    else:
        # Standard multi-class ECE
        if targets.ndim == 2:
            targets = targets.argmax(axis=1)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == targets).astype(float)
        return _ece_from_conf_acc(confidences, accuracies, n_bins)


def _binary_ece(probs_k: np.ndarray, targets_k: np.ndarray, n_bins: int) -> float:
    """ECE for a single binary label."""
    confidences = np.where(targets_k == 1, probs_k, 1 - probs_k)
    # Actually: confidence = max(p, 1-p) and accuracy = (pred == target)
    preds = (probs_k >= 0.5).astype(int)
    conf = np.maximum(probs_k, 1 - probs_k)
    acc = (preds == targets_k).astype(float)
    return _ece_from_conf_acc(conf, acc, n_bins)


def _ece_from_conf_acc(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ood_auroc(
    id_uncertainty: np.ndarray,
    ood_uncertainty: np.ndarray,
) -> float:
    """OOD detection AUROC using uncertainty scores.

    In-distribution samples should have LOW uncertainty,
    OOD samples should have HIGH uncertainty.

    Args:
        id_uncertainty: (N_id,) uncertainty scores for in-distribution.
        ood_uncertainty: (N_ood,) uncertainty scores for OOD.
    Returns:
        AUROC for distinguishing ID vs OOD.
    """
    labels = np.concatenate([
        np.zeros(len(id_uncertainty)),   # ID = 0
        np.ones(len(ood_uncertainty)),   # OOD = 1
    ])
    scores = np.concatenate([id_uncertainty, ood_uncertainty])
    return float(roc_auc_score(labels, scores))


def compute_all_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    ood_uncertainty: Optional[np.ndarray] = None,
    multi_label: bool = False,
    n_bins: int = 15,
    return_per_class: bool = False,
) -> Dict[str, float]:
    """Compute all metrics at once."""
    results = {
        "auc": compute_auc(probs, targets, multi_label),
        "acc": compute_accuracy(probs, targets, multi_label),
        "ece": compute_ece(probs, targets, n_bins, multi_label),
    }
    if uncertainty is not None and ood_uncertainty is not None:
        results["ood_auroc"] = compute_ood_auroc(uncertainty, ood_uncertainty)
    if return_per_class and targets.ndim == 2 and targets.shape[1] > 1:
        results["per_class_auc"] = compute_per_class_auc(probs, targets)
    return results
