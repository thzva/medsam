"""Figures for the revision: reliability diagrams, uncertainty distributions,
and the near-OOD severity sweep.  All read the machine-readable results only.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/2025user/zhou/medsam")
FIG = ROOT / "revision" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight"})

DISEASE = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
           "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
           "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def load(mode="tms_edl"):
    best = None
    for f in glob.glob(str(ROOT / "outputs_revision" / "**" / "eval_revision.json"),
                       recursive=True):
        d = json.load(open(f))
        if d.get("mode") == mode:
            best = d
            if "seed42" in f:
                break
    return best


def fig_reliability(d):
    rel = d.get("reliability", {})
    if not rel:
        return
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.2), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), DISEASE):
        r = rel.get(name)
        ax.plot([0, 1], [0, 1], "--", c="0.6", lw=0.8)
        if r:
            c = np.array(r["conf"], dtype=float)
            f = np.array(r["freq"], dtype=float)
            w = np.array(r["weight"], dtype=float)
            m = np.isfinite(c) & np.isfinite(f)
            ax.scatter(c[m], f[m], s=np.clip(w[m] * 900, 3, 90),
                       c="#2166ac", alpha=0.8, edgecolors="none")
            ax.plot(c[m], f[m], c="#2166ac", lw=0.8, alpha=0.7)
        ax.set_title(name.replace("_", " "), fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("empirical frequency")
    for ax in axes[-1, :]:
        ax.set_xlabel("predicted probability")
    fig.suptitle("Reliability diagrams per finding (marker area $\\propto$ bin mass)",
                 fontsize=9)
    fig.savefig(FIG / "reliability.pdf"); plt.close(fig)
    print("  reliability.pdf")


def fig_uncertainty(d):
    bs = d.get("branch_stats", {})
    far = d.get("far_ood", {})
    ext = d.get("external", {})
    groups = [("ID test", bs.get("test_all", {})),
              ("ID errors", bs.get("test_error", {}))]
    if "chexpert_val" in ext:
        groups.append(("CheXpert\n(near-OOD)", ext["chexpert_val"].get("branch_stats", {})))
    for tag, lab in [("ct_liver", "Liver CT\n(far)"), ("imagenet", "Imagenette\n(far)")]:
        if tag in far:
            groups.append((lab, far[tag].get("branch_stats", {})))
    keys = [("u_global", "$u_g$"), ("u_local", "$u_\\ell$"),
            ("conflict", "$K$"), ("u_fused", "$u_{\\mathrm{fused}}$")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 2.6))
    x = np.arange(len(groups))
    for ax, (k, lab) in zip(axes, keys):
        mu = [g.get(k, {}).get("mean", np.nan) for _, g in groups]
        lo = [g.get(k, {}).get("p05", np.nan) for _, g in groups]
        hi = [g.get(k, {}).get("p95", np.nan) for _, g in groups]
        mu, lo, hi = map(np.array, (mu, lo, hi))
        err = np.vstack([np.clip(mu - lo, 0, None), np.clip(hi - mu, 0, None)])
        ax.errorbar(x, mu, yerr=err, fmt="o", capsize=3, ms=4, c="#b2182b")
        ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
        ax.set_title(lab, fontsize=9); ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle("Branch-wise uncertainty and conflict by split "
                 "(mean, 5th--95th percentile)", fontsize=9)
    fig.savefig(FIG / "uncertainty_by_split.pdf"); plt.close(fig)
    print("  uncertainty_by_split.pdf")


def fig_near_ood(d):
    near = d.get("near_ood", {})
    if not near:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    sevs = [1, 3, 5]
    for kind, rec in near.items():
        au = [rec.get(f"severity_{s}", {}).get("auroc", np.nan) for s in sevs]
        ac = [rec.get(f"severity_{s}", {}).get("macro_auc", np.nan) for s in sevs]
        a1.plot(sevs, au, "o-", ms=4, label=kind.replace("_", " "))
        a2.plot(sevs, ac, "o-", ms=4, label=kind.replace("_", " "))
    a1.set_xlabel("corruption severity"); a1.set_ylabel("OOD AUROC of $u$")
    a1.axhline(0.5, ls="--", c="0.6", lw=0.8); a1.grid(alpha=0.25, lw=0.4)
    a2.set_xlabel("corruption severity"); a2.set_ylabel("macro AUC (diagnosis)")
    a2.grid(alpha=0.25, lw=0.4)
    a1.set_xticks(sevs); a2.set_xticks(sevs)
    a2.legend(fontsize=6.5, frameon=False)
    fig.suptitle("Near-OOD: does uncertainty rise as accuracy falls?", fontsize=9)
    fig.savefig(FIG / "near_ood_sweep.pdf"); plt.close(fig)
    print("  near_ood_sweep.pdf")


def main():
    d = load("tms_edl")
    if d is None:
        print("no tms_edl results yet"); return
    fig_reliability(d); fig_uncertainty(d); fig_near_ood(d)


if __name__ == "__main__":
    main()
