"""Calibration, discrimination and clinical-utility metrics for the revision.

Every metric here is defined for the MULTI-LABEL setting and states its
aggregation rule explicitly, because reviewers M2/M8 objected that the
original ECE was reported without one.

Conventions
-----------
probs   : (N, C) independent per-label probabilities of "finding present"
targets : (N, C) binary ground truth
All "macro" metrics compute the metric per label, then average over the C
labels, skipping degenerate labels (all-positive or all-negative).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)

_EPS = 1e-12


# ───────────────────────────── calibration ──────────────────────────────── #

def binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15,
               scheme: str = "equal_width") -> float:
    """Top-label ECE for one binary problem.

    confidence = max(p, 1-p), correctness = 1[(p>=.5) == y].
    `scheme`: 'equal_width' (original) or 'equal_mass' (quantile bins, which
    removes the empty-bin bias that R1 comment 5 objected to).
    """
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(int) == y.astype(int)).astype(float)
    if scheme == "equal_mass":
        qs = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(qs)
        if len(edges) < 2:
            return 0.0
    else:
        edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def binary_class_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15,
                     scheme: str = "equal_width") -> float:
    """Calibration of the POSITIVE-class probability itself (|E[y|p] - p|).

    For heavily imbalanced labels this is the clinically meaningful notion;
    top-label ECE can look excellent simply because almost everything is
    predicted negative.
    """
    if scheme == "equal_mass":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            return 0.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 15
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centre, empirical_freq, bin_weight) for a reliability plot."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    cen, freq, wt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            cen.append(np.nan); freq.append(np.nan); wt.append(0.0); continue
        cen.append(p[m].mean()); freq.append(y[m].mean()); wt.append(m.mean())
    return np.array(cen), np.array(freq), np.array(wt)


def multilabel_nll(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean per-label Bernoulli negative log-likelihood."""
    p = np.clip(probs, _EPS, 1 - _EPS)
    return float(-(targets * np.log(p) + (1 - targets) * np.log(1 - p)).mean())


def multilabel_brier(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean per-label Brier score."""
    return float(((probs - targets) ** 2).mean())


# ───────────────────────── discrimination / utility ─────────────────────── #

def _valid_cols(targets: np.ndarray) -> List[int]:
    return [k for k in range(targets.shape[1])
            if 0 < targets[:, k].sum() < len(targets)]


def macro_auc(probs, targets, cols: Optional[List[int]] = None) -> float:
    cols = _valid_cols(targets) if cols is None else cols
    return float(np.mean([roc_auc_score(targets[:, k], probs[:, k]) for k in cols]))


def macro_auprc(probs, targets, cols: Optional[List[int]] = None) -> float:
    cols = _valid_cols(targets) if cols is None else cols
    return float(np.mean([average_precision_score(targets[:, k], probs[:, k]) for k in cols]))


def micro_auprc(probs, targets, cols: Optional[List[int]] = None) -> float:
    cols = _valid_cols(targets) if cols is None else cols
    return float(average_precision_score(targets[:, cols].ravel(), probs[:, cols].ravel()))


def tune_thresholds(probs_val: np.ndarray, targets_val: np.ndarray,
                    n_grid: int = 199) -> np.ndarray:
    """Per-label decision threshold maximising F1 on the VALIDATION split.

    Reviewer M6: the 91.04% accuracy came from a fixed 0.5 threshold that was
    never justified.  Thresholds must be fixed on validation, not test.
    """
    C = probs_val.shape[1]
    thr = np.full(C, 0.5)
    grid = np.linspace(0.005, 0.995, n_grid)
    for k in range(C):
        y = targets_val[:, k]
        if y.sum() == 0:
            continue
        best, best_t = -1.0, 0.5
        for t in grid:
            pred = (probs_val[:, k] >= t)
            tp = float((pred & (y == 1)).sum())
            if tp == 0:
                continue
            prec = tp / pred.sum()
            rec = tp / y.sum()
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best:
                best, best_t = f1, t
        thr[k] = best_t
    return thr


def operating_point_metrics(probs, targets, thr: np.ndarray) -> Dict[str, object]:
    """Per-label sensitivity / specificity / F1 / PPV at fixed thresholds."""
    C = probs.shape[1]
    out = {"sensitivity": [], "specificity": [], "f1": [], "ppv": [], "threshold": thr.tolist()}
    for k in range(C):
        y = targets[:, k].astype(int)
        pred = (probs[:, k] >= thr[k]).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            out["sensitivity"].append(float("nan")); out["specificity"].append(float("nan"))
            out["f1"].append(float("nan")); out["ppv"].append(float("nan")); continue
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        f1 = 2 * ppv * sens / max(ppv + sens, _EPS)
        out["sensitivity"].append(float(sens)); out["specificity"].append(float(spec))
        out["f1"].append(float(f1)); out["ppv"].append(float(ppv))
    for key in ("sensitivity", "specificity", "f1", "ppv"):
        out[f"macro_{key}"] = float(np.nanmean(out[key]))
    return out


# ───────────────────────────── uncertainty ──────────────────────────────── #

def ood_auroc(u_id: np.ndarray, u_ood: np.ndarray) -> Dict[str, float]:
    """AUROC of the uncertainty score separating ID from OOD.

    NaNs are reported, never silently dropped (defect D1): a non-finite
    uncertainty is scored as maximally uncertain so that a numerically broken
    branch cannot be rewarded by having its worst samples deleted.
    """
    n_bad_id = int((~np.isfinite(u_id)).sum())
    n_bad_ood = int((~np.isfinite(u_ood)).sum())
    finite = np.concatenate([u_id[np.isfinite(u_id)], u_ood[np.isfinite(u_ood)]])
    fill = finite.max() if finite.size else 1.0
    a = np.where(np.isfinite(u_id), u_id, fill)
    b = np.where(np.isfinite(u_ood), u_ood, fill)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return {
        "auroc": float(roc_auc_score(y, np.r_[a, b])),
        "n_nonfinite_id": n_bad_id,
        "n_nonfinite_ood": n_bad_ood,
        "mean_u_id": float(np.nanmean(u_id)),
        "mean_u_ood": float(np.nanmean(u_ood)),
    }


# ───────────────────────────── uncertainty via CI ───────────────────────── #

def bootstrap_ci(fn, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for any metric fn(*arrays) over the sample axis."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = float(fn(*arrays))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(float(fn(*[a[idx] for a in arrays])))
        except Exception:
            continue
    if not vals:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": len(vals)}


# ─────────────────────────── one-call summary ───────────────────────────── #

def full_metrics(probs, targets, label_names: List[str], n_bins: int = 15,
                 thresholds: Optional[np.ndarray] = None,
                 disease_only_idx: Optional[List[int]] = None,
                 n_boot: int = 1000) -> Dict[str, object]:
    cols = _valid_cols(targets)
    res: Dict[str, object] = {}

    res["macro_auc_all"] = macro_auc(probs, targets, cols)
    if disease_only_idx is not None:
        dcols = [k for k in cols if k in disease_only_idx]
        res["macro_auc_14disease"] = macro_auc(probs, targets, dcols)
        res["macro_auprc_14disease"] = macro_auprc(probs, targets, dcols)
    res["macro_auprc"] = macro_auprc(probs, targets, cols)
    res["micro_auprc"] = micro_auprc(probs, targets, cols)

    res["per_class_auc"] = {label_names[k]: float(roc_auc_score(targets[:, k], probs[:, k]))
                            for k in cols}
    res["per_class_auprc"] = {label_names[k]: float(average_precision_score(targets[:, k], probs[:, k]))
                              for k in cols}
    res["prevalence"] = {label_names[k]: float(targets[:, k].mean()) for k in range(targets.shape[1])}

    # calibration, both binning schemes, both notions
    for scheme in ("equal_width", "equal_mass"):
        res[f"macro_ece_toplabel_{scheme}"] = float(np.mean(
            [binary_ece(probs[:, k], targets[:, k], n_bins, scheme) for k in cols]))
        res[f"macro_ece_positive_{scheme}"] = float(np.mean(
            [binary_class_ece(probs[:, k], targets[:, k], n_bins, scheme) for k in cols]))
    res["per_class_ece_toplabel"] = {label_names[k]: binary_ece(probs[:, k], targets[:, k], n_bins)
                                     for k in cols}
    res["brier"] = multilabel_brier(probs, targets)
    res["nll"] = multilabel_nll(probs, targets)

    # pooled (all label decisions in one pool) -- the alternative aggregation
    res["pooled_ece_toplabel"] = binary_ece(probs[:, cols].ravel(), targets[:, cols].ravel(), n_bins)

    if thresholds is not None:
        res["operating_point"] = operating_point_metrics(probs, targets, thresholds)
        preds = (probs >= thresholds[None, :]).astype(int)
        res["per_label_accuracy"] = float((preds == targets).mean())
        res["exact_match_accuracy"] = float((preds == targets).all(1).mean())

    if n_boot:
        res["ci"] = {
            "macro_auc_all": bootstrap_ci(lambda p, t: macro_auc(p, t), probs, targets, n_boot=n_boot),
            "macro_auprc": bootstrap_ci(lambda p, t: macro_auprc(p, t), probs, targets, n_boot=n_boot),
            "macro_ece_toplabel_equal_width": bootstrap_ci(
                lambda p, t: float(np.mean([binary_ece(p[:, k], t[:, k], n_bins) for k in _valid_cols(t)])),
                probs, targets, n_boot=max(n_boot // 5, 100)),
            "brier": bootstrap_ci(multilabel_brier, probs, targets, n_boot=n_boot),
        }
    return res
