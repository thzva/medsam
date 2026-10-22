"""Post-hoc analyses for the TVC manuscript, from saved per-sample arrays only.

  1. decoupled read-out: predict with p_f, score OOD with u_g / max(u_g,u_l) / mean
  2. deep ensemble of ViT-B/16 BCE members (probabilities averaged; OOD score =
     mean binary entropy of the ensemble, and mutual information)
  3. risk-coverage: macro AUC on the retained fraction after rejecting the most
     uncertain images; AURC of per-image Brier error
  4. OOD density figure in the (log u_g, log u_l) plane with iso-u_f contours

Writes revision/tvc/generated/posthoc.json and revision/tvc/figures/*.pdf.
Run from the repo root."""
import glob, json, os
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "revision/tvc"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
EPS = 1e-7


def auroc(u_id, u_ood):
    return float(roc_auc_score(np.r_[np.zeros(len(u_id)), np.ones(len(u_ood))], np.r_[u_id, u_ood]))


def macro_auc(p, y):
    return float(np.mean([roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]))


def bin_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return (-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean(1)


def load_run(d):
    if not os.path.exists(f"{d}/arrays.npz"):
        return None, None
    a = np.load(f"{d}/arrays.npz")
    o = np.load(f"{d}/ood_arrays.npz") if os.path.exists(f"{d}/ood_arrays.npz") else None
    if o is None:  # new runs keep OOD arrays inside arrays.npz
        o = a if "ood_imagenet_u_fused" in a.files else None
    return a, o


# ---------------------------------------------------------------- 1. read-outs
DUAL = {
    "crop disabled": "outputs_fixed/ablations/ablation_attn_crop_off",
    "uniform MIL": "outputs_tvc/ablations/ablation_mil_uniform",
    "discounting": "outputs_tvc/ablations/ablation_fusion_discounted",
    "gated MIL": "outputs_mil2/ablations/ablation_mil_no_nms",
    "gated MIL + NMS": "outputs_mil2/tms_edl",
}
for m in ["fusion_yager", "fusion_cautious", "fusion_average", "fusion_gating"]:
    DUAL[m] = f"outputs_tvc/ablations/ablation_{m}"
DUAL["Dempster (re-run)"] = "outputs_tvc/tms_edl"

res = {"readout": {}, "ensemble": {}, "risk_coverage": {}}
for name, run in DUAL.items():
    rows = []
    for d in sorted(glob.glob(f"{run}/seed*")):
        a, o = load_run(d)
        if a is None or o is None or "test_u_global" not in a.files:
            continue
        r = {"auc_fused": macro_auc(a["test_probs"], a["test_targets"])}
        for ds in ["imagenet", "ct_liver"]:
            ug, ul, uf = (o[f"ood_{ds}_u_global"], o[f"ood_{ds}_u_local"], o[f"ood_{ds}_u_fused"])
            g, l, f = a["test_u_global"], a["test_u_local"], a["test_u_fused"]
            r[f"{ds}_u_f"] = auroc(f, uf)
            r[f"{ds}_u_g"] = auroc(g, ug)
            r[f"{ds}_u_l"] = auroc(l, ul)
            r[f"{ds}_max"] = auroc(np.maximum(g, l), np.maximum(ug, ul))
            r[f"{ds}_mean"] = auroc((g + l) / 2, (ug + ul) / 2)
        rows.append(r)
    if rows:
        res["readout"][name] = {k: [r[k] for r in rows] for k in rows[0]}

# ---------------------------------------------------------------- 2. ensemble
members = sorted(d for d in glob.glob("outputs_tvc/baselines/vit_b16_ce/seed*") if os.path.exists(f"{d}/eval_revision.json"))
if len(members) >= 2:
    A = [load_run(d) for d in members]
    y = A[0][0]["test_targets"]
    P = np.stack([a["test_probs"] for a, _ in A])
    p = P.mean(0)
    ens = {"n_members": len(members), "auc": macro_auc(p, y),
           "member_auc": [macro_auc(q, y) for q in P]}
    if all(o is not None for _, o in A):
        for ds in ["imagenet", "ct_liver"]:
            Q = np.stack([o[f"ood_{ds}_probs"] for _, o in A])
            q = Q.mean(0)
            ens[f"{ds}_entropy"] = auroc(bin_entropy(p), bin_entropy(q))
            mi_id = bin_entropy(p) - np.mean([bin_entropy(x) for x in P], 0)
            mi_ood = bin_entropy(q) - np.mean([bin_entropy(x) for x in Q], 0)
            ens[f"{ds}_mutual_info"] = auroc(mi_id, mi_ood)
    # calibration of the ensemble with the same ECE definition as the paper
    from src.evaluation.metrics_v2 import full_metrics
    fm = full_metrics(p, y, [str(i) for i in range(y.shape[1])], thresholds=None, n_boot=0)
    ens["ece"] = fm["macro_ece_toplabel_equal_width"]; ens["brier"] = fm["brier"]; ens["nll"] = fm["nll"]
    res["ensemble"] = ens

# ---------------------------------------------------------------- 3. risk-coverage
COV = np.linspace(0.1, 1.0, 19)
RC = {"global only": ("outputs_fixed/ablations/ablation_stream_global_only", "test_u_fused"),
      "dual, u_f": ("outputs_mil2/tms_edl", "test_u_fused"),
      "dual, u_g": ("outputs_mil2/tms_edl", "test_u_global"),
      "ViT BCE, entropy": ("outputs_tvc/baselines/vit_b16_ce", None)}
for name, (run, key) in RC.items():
    curves, aurcs = [], []
    for d in sorted(glob.glob(f"{run}/seed*")):
        if not os.path.exists(f"{d}/arrays.npz"):
            continue
        a = np.load(f"{d}/arrays.npz"); p, y = a["test_probs"], a["test_targets"]
        u = bin_entropy(p) if key is None else a[key]
        order = np.argsort(u)
        err = ((p - y) ** 2).mean(1)[order]
        risk = np.cumsum(err) / np.arange(1, len(err) + 1)          # Brier risk of the retained set
        aurcs.append(float(np.mean(risk)))
        curves.append([float(risk[int(c * len(order)) - 1]) for c in COV])
    if curves:
        res["risk_coverage"][name] = {"coverage": COV.tolist(), "risk_mean": np.mean(curves, 0).tolist(),
                                      "risk_std": np.std(curves, 0).tolist(), "aurc_brier": aurcs}

json.dump(res, open(f"{OUT}/generated/posthoc.json", "w"), indent=1)

# ---------------------------------------------------------------- figures
if res["risk_coverage"]:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (name, r), col in zip(res["risk_coverage"].items(), [BLUE, AQUA, ORANGE, GREY]):
        ax.plot(np.array(r["coverage"]) * 100, r["risk_mean"], lw=2, color=col, label=name, marker="o", ms=3)
    ax.set_xlabel("coverage (% images retained)"); ax.set_ylabel("Brier risk on retained set")
    ax.invert_xaxis(); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/risk_coverage.pdf")

d = "outputs_mil2/tms_edl/seed42"
if os.path.exists(f"{d}/ood_arrays.npz"):
    a, o = load_run(d)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    sets = [("NIH test (ID)", np.log10(a["test_u_global"]), np.log10(a["test_u_local"]), BLUE),
            ("ImageNet (OOD)", np.log10(o["ood_imagenet_u_global"]), np.log10(o["ood_imagenet_u_local"]), ORANGE),
            ("liver CT (OOD)", np.log10(o["ood_ct_liver_u_global"]), np.log10(o["ood_ct_liver_u_local"]), AQUA)]
    rng = np.random.default_rng(0)
    for lab, x, yv, col in sets:
        i = rng.choice(len(x), min(len(x), 3000), replace=False)
        ax.scatter(x[i], yv[i], s=3, alpha=0.4, color=col, linewidths=0, label=lab, rasterized=True)
    gx, gy = np.meshgrid(np.linspace(-1.8, -0.8, 200), np.linspace(-4.6, 0.1, 200))
    ug, ul = 10 ** gx, 10 ** gy
    bound = ug * ul / (ug + ul - ug * ul)            # Proposition 1 upper bound on u_f
    cs = ax.contour(gx, gy, np.log10(bound), levels=[-4.5, -3.5, -2.5, -1.5], colors=GREY, linewidths=0.7)
    ax.clabel(cs, fmt=lambda v: f"$u_f\\leq10^{{{v:g}}}$", fontsize=5.5)
    ax.set_xlabel("$\\log_{10} u_g$ (global)"); ax.set_ylabel("$\\log_{10} u_\\ell$ (local)")
    ax.set_xlim(-1.8, -0.8); ax.set_ylim(-4.6, 0.1)
    ax.legend(frameon=False, fontsize=6.5, markerscale=3, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/ood_density.pdf", dpi=300)

print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in res.items()}, indent=1)[:2000])
