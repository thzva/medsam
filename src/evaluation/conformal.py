"""Split-conformal prediction and temperature scaling baselines.

R1 comment 6: the single-forward-pass claim is not itself novel, so the paper
must compare against (a) a deterministic single-pass network with post-hoc
calibration, and (b) conformal prediction, which gives distribution-free
finite-sample coverage guarantees without any evidential machinery.

Multi-label conformal is done label-wise (Mondrian / class-conditional): for
each label c we calibrate a threshold on a held-out split so that the
resulting prediction set covers the true outcome with probability >= 1-alpha.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.optimize import minimize_scalar


# ─────────────────────────── temperature scaling ────────────────────────── #

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(probs_cal: np.ndarray, targets_cal: np.ndarray,
                    per_label: bool = False) -> np.ndarray:
    """Fit temperature(s) minimising NLL on the calibration split."""
    z = _logit(probs_cal)

    def nll(T, zz, yy):
        p = 1.0 / (1.0 + np.exp(-zz / T))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())

    if not per_label:
        r = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                            args=(z, targets_cal))
        return np.full(probs_cal.shape[1], r.x)
    Ts = np.ones(probs_cal.shape[1])
    for k in range(probs_cal.shape[1]):
        r = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                            args=(z[:, k], targets_cal[:, k]))
        Ts[k] = r.x
    return Ts


def apply_temperature(probs: np.ndarray, T: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(probs) / T[None, :]))


# ───────────────────────────── conformal ────────────────────────────────── #

def calibrate_conformal(probs_cal: np.ndarray, targets_cal: np.ndarray,
                        alpha: float = 0.1) -> np.ndarray:
    """Label-wise conformal thresholds at miscoverage alpha.

    Non-conformity score for label c on a calibration point is
        s = 1 - p_c   if y_c = 1
            p_c       if y_c = 0
    i.e. one minus the probability assigned to the realised outcome.  The
    threshold is the ceil((n+1)(1-alpha))/n empirical quantile of s.
    """
    C = probs_cal.shape[1]
    q = np.zeros(C)
    for k in range(C):
        p, y = probs_cal[:, k], targets_cal[:, k]
        s = np.where(y == 1, 1.0 - p, p)
        n = len(s)
        lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q[k] = np.quantile(s, lvl, method="higher")
    return q


def conformal_sets(probs: np.ndarray, q: np.ndarray) -> Dict[str, np.ndarray]:
    """Return the per-label prediction set as two boolean masks.

    include_pos[i,c]: outcome "present" is in the set for sample i, label c.
    include_neg[i,c]: outcome "absent"  is in the set.
    A set of size 2 ({present, absent}) is an explicit abstention -- the
    conformal analogue of a high-uncertainty flag.
    """
    include_pos = (1.0 - probs) <= q[None, :]
    include_neg = probs <= q[None, :]
    return {"pos": include_pos, "neg": include_neg,
            "size": include_pos.astype(int) + include_neg.astype(int)}


def conformal_metrics(probs: np.ndarray, targets: np.ndarray,
                      q: np.ndarray, alpha: float = 0.1) -> Dict[str, object]:
    """Empirical coverage and set size, macro-averaged over labels."""
    s = conformal_sets(probs, q)
    covered = np.where(targets == 1, s["pos"], s["neg"])
    per_label_cov = covered.mean(0)
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "macro_coverage": float(per_label_cov.mean()),
        "min_label_coverage": float(per_label_cov.min()),
        "per_label_coverage": per_label_cov.tolist(),
        "mean_set_size": float(s["size"].mean()),
        "abstention_rate": float((s["size"] == 2).mean()),
        "empty_set_rate": float((s["size"] == 0).mean()),
    }
