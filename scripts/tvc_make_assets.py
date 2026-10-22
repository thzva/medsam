"""Generate every number and figure in revision/tvc/main.tex from saved results.
Run from the repo root: python scripts/tvc_make_assets.py"""
import json, glob, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "revision/tvc"
os.makedirs(f"{OUT}/generated", exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#444",
                     "ytick.color": "#444", "pdf.fonttype": 42})

def load(run):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{run}/seed*/eval_revision.json"))]

def ms(vals, nd=4):
    vals = [v for v in vals if v is not None]
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}$^\\ast$"
    sd = f"{st.stdev(vals):.{nd}f}".lstrip("0")
    return f"{st.mean(vals):.{nd}f}{{\\scriptsize$\\,\\pm${sd}}}"

def mean(vals): return st.mean(vals)

T = lambda d: d["test"]
FAR = lambda d, k, f="auroc": d["far_ood"][k][f]
EXT = lambda d: d["external"]["chexpert_val"]["macro_auc"]

# ---------------- main table (corrected protocol) ----------------
main_rows = [
    ("Global stream only (ViT-B/16, Beta)", "outputs_fixed/ablations/ablation_stream_global_only"),
    ("Dual stream, crop disabled",           "outputs_fixed/ablations/ablation_attn_crop_off"),
    ("Dual stream, uniform MIL",             "outputs_tvc/ablations/ablation_mil_uniform"),
    ("Dual stream, gated MIL",               "outputs_mil2/ablations/ablation_mil_no_nms"),
    ("Dual stream, gated MIL + NMS",         "outputs_mil2/tms_edl"),
    ("Dual stream + Shafer discounting",     "outputs_tvc/ablations/ablation_fusion_discounted"),
]
lines = []
for name, run in main_rows:
    R = load(run)
    lines.append(f"{name} & {ms([T(d)['macro_auc_all'] for d in R])} & {ms([T(d)['macro_auprc'] for d in R])} & "
                 f"{ms([T(d)['macro_ece_toplabel_equal_width'] for d in R])} & {ms([T(d)['brier'] for d in R])} & "
                 f"{ms([T(d)['nll'] for d in R])} \\\\")
open(f"{OUT}/generated/main_rows.tex", "w").write("\n".join(lines) + "\n")

# CE baselines (earlier protocol, single seed) -- reported separately and flagged
base_rows = [("ViT-B/16, BCE", "outputs_final/baselines/vit_b16_ce"),
             ("DenseNet-121, BCE", "outputs_final/baselines/densenet121_ce"),
             ("ResNet-50, BCE", "outputs_final/baselines/resnet50_ce"),
             ("ResNet-50, MC dropout", "outputs_final/baselines/resnet50_mc_dropout"),
             ("ViT-B/16, Beta EDL (ours, same protocol)", "outputs_final/ablations/ablation_stream_global_only"),
             ("TMS-EDL (same protocol)", "outputs_final/tms_edl")]
lines = []
for name, run in base_rows:
    R = load(run)
    lines.append(f"{name} & {ms([T(d)['macro_auc_all'] for d in R])} & {ms([T(d)['macro_ece_toplabel_equal_width'] for d in R])} & "
                 f"{ms([d['test_temperature_scaled']['macro_ece_toplabel_equal_width'] for d in R])} & "
                 f"{ms([FAR(d,'imagenet') for d in R],3)} & {ms([FAR(d,'ct_liver') for d in R],3)} \\\\")
open(f"{OUT}/generated/baseline_rows.tex", "w").write("\n".join(lines) + "\n")

# ---------------- Beta vs categorical Dirichlet (matched seed 42) ----------------
def s42(run): return json.load(open(f"{run}/seed42/eval_revision.json"))
pairs = [("Dual stream (ViT + ResNet-50)", "outputs_final/ablations/ablation_evid_dirichlet", "outputs_final/tms_edl"),
         ("Single stream (ResNet-50)", "outputs_final/baselines/edl_single_resnet50", "outputs_final/ablations/ablation_stream_local_only")]
lines = []
for name, a, b in pairs:
    A, B = s42(a), s42(b)
    lines.append(f"{name} & {T(A)['macro_auc_all']:.4f} & {T(B)['macro_auc_all']:.4f} & {T(B)['macro_auc_all']-T(A)['macro_auc_all']:+.4f} & "
                 f"{EXT(A):.4f} & {EXT(B):.4f} & {EXT(B)-EXT(A):+.4f} \\\\")
open(f"{OUT}/generated/beta_rows.tex", "w").write("\n".join(lines) + "\n")

# ---------------- combination rules (earlier protocol) ----------------
rules = [("Dempster", "outputs_final/tms_edl"), ("Yager", "outputs_final/ablations/ablation_fusion_yager"),
         ("Mean-BBA (Murphy)", "outputs_final/ablations/ablation_fusion_cautious"),
         ("Opinion averaging", "outputs_final/ablations/ablation_fusion_average"),
         ("Learned gating", "outputs_final/ablations/ablation_fusion_gating"),
         ("Dempster + Shafer discounting", "outputs_final/ablations/ablation_fusion_discounted")]
lines = []
for name, run in rules:
    R = load(run)
    ug = [FAR(d, 'imagenet', 'auroc_u_global') for d in R]; uf = [FAR(d, 'imagenet') for d in R]
    lines.append(f"{name} & {len(R)} & {ms([T(d)['macro_auc_all'] for d in R])} & {ms([T(d)['macro_ece_toplabel_equal_width'] for d in R])} & "
                 f"{mean(ug):.3f} & {mean(uf):.3f} & {mean(uf)-mean(ug):+.3f} \\\\")
open(f"{OUT}/generated/rule_rows.tex", "w").write("\n".join(lines) + "\n")

# ---------------- branch-wise OOD + external ----------------
ood_rows = main_rows
lines = []
branch = {}
for name, run in ood_rows:
    R = load(run)
    g = [FAR(d,'imagenet','auroc_u_global') for d in R]; l = [FAR(d,'imagenet','auroc_u_local') for d in R]; f = [FAR(d,'imagenet') for d in R]
    ct = [FAR(d,'ct_liver') for d in R]; ex = [EXT(d) for d in R]
    branch[name] = (g, l, f)
    single = "global_only" in run
    lines.append(f"{name} & {mean(g):.3f} & {'--' if single else f'{mean(l):.3f}'} & {mean(f):.3f} & {mean(f)-mean(g):+.3f} & {mean(ct):.3f} & {ms(ex)} \\\\")
open(f"{OUT}/generated/ood_rows.tex", "w").write("\n".join(lines) + "\n")

# ---------------- Figure: vacuity bound on the ID test set ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
z = np.load("outputs_mil2/tms_edl/seed42/arrays.npz")
ug, ul, uf = z["test_u_global"], z["test_u_local"], z["test_u_fused"]
rng = np.random.default_rng(0); idx = rng.choice(len(uf), 4000, replace=False)
ax = axes[0]
m = np.minimum(ug, ul)
ax.scatter(m[idx], uf[idx], s=3, alpha=0.35, color=BLUE, linewidths=0, rasterized=True)
lo, hi = 1e-3, 3e-1
ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--")
ax.text(1.3e-3, 2.1e-3, "$y = x$", color="#444", fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("$\\min(u_g, u_\\ell)$  (mean vacuity)"); ax.set_ylabel("fused vacuity $u_f$")
ax.set_title("(a) NIH test, 25,596 images: $u_f \\leq \\min(u_g,u_\\ell)$", fontsize=8, loc="left")

ax = axes[1]
names = [n for n, _ in ood_rows if "global_only" not in _]
short = ["crop\noff", "uniform\nMIL", "gated\nMIL", "gated MIL\n+NMS", "Shafer\ndiscount"]
x = np.arange(len(names)); w = 0.26
for k, (lab, col) in enumerate([("$u_g$ (global)", BLUE), ("$u_\\ell$ (local)", ORANGE), ("$u_f$ (fused)", AQUA)]):
    vals = [branch[n][k] for n in names]
    ax.bar(x + (k - 1) * w, [mean(v) for v in vals], w * 0.9, color=col, label=lab)
    for i, v in enumerate(vals):
        ax.scatter([x[i] + (k - 1) * w] * len(v), v, s=6, color="#222", zorder=3, linewidths=0)
g0 = mean(branch["Global stream only (ViT-B/16, Beta)"][0])
ax.axhline(0.5, color=GREY, lw=0.8, ls=":"); ax.axhline(g0, color=BLUE, lw=0.8, ls="--")
ax.text(len(names) - 0.4, g0, "global-\nonly", color="#444", fontsize=6.5, ha="left", va="center", clip_on=False)
ax.text(len(names) - 0.4, 0.5, "chance", color="#444", fontsize=6.5, ha="left", va="center", clip_on=False)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7); ax.set_ylim(0, 1.0)
ax.set_ylabel("OOD AUROC (ImageNet vs. NIH)")
ax.set_title("(b) fused score tracks the local branch", fontsize=8, loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.42), handlelength=1.0, columnspacing=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/vacuity_collapse.pdf", dpi=300)
print("ok")

# ---------------- TVC re-runs (final protocol, outputs_tvc); written only when complete ----------------
def have3(run): return len(glob.glob(f"{run}/seed*/eval_revision.json")) >= 3

TV = "outputs_tvc"
new_base = [("ViT-B/16, BCE", f"{TV}/baselines/vit_b16_ce"),
            ("DenseNet-121, BCE", f"{TV}/baselines/densenet121_ce"),
            ("ResNet-50, BCE", f"{TV}/baselines/resnet50_ce"),
            ("ResNet-50, MC dropout", f"{TV}/baselines/resnet50_mc_dropout"),
            ("ViT-B/16, Beta EDL (global stream)", "outputs_fixed/ablations/ablation_stream_global_only"),
            ("Dual stream, gated MIL + NMS (Dempster)", "outputs_mil2/tms_edl")]
lines = []
for name, run in new_base:
    if not glob.glob(f"{run}/seed*/eval_revision.json"):
        continue
    R = load(run)[:3]  # ensemble members 45/46 are excluded from the per-model row
    lines.append(f"{name} & {ms([T(d)['macro_auc_all'] for d in R])} & {ms([T(d)['macro_ece_toplabel_equal_width'] for d in R])} & "
                 f"{ms([d['test_temperature_scaled']['macro_ece_toplabel_equal_width'] for d in R])} & "
                 f"{ms([FAR(d,'imagenet') for d in R],3)} & {ms([FAR(d,'ct_liver') for d in R],3)} \\\\")
open(f"{OUT}/generated/baseline_rows_tvc.tex", "w").write("\n".join(lines) + "\n")

new_rules = [("Dempster", "outputs_mil2/tms_edl"),
             ("Dempster (re-run, protocol check)", f"{TV}/tms_edl"),
             ("Yager", f"{TV}/ablations/ablation_fusion_yager"),
             ("Mean-BBA (Murphy)", f"{TV}/ablations/ablation_fusion_cautious"),
             ("Opinion averaging", f"{TV}/ablations/ablation_fusion_average"),
             ("Learned gating", f"{TV}/ablations/ablation_fusion_gating")]
lines = []
for name, run in new_rules:
    R = load(run)
    if not R: continue
    ug = [FAR(d, 'imagenet', 'auroc_u_global') for d in R]; uf = [FAR(d, 'imagenet') for d in R]
    lines.append(f"{name} & {len(R)} & {ms([T(d)['macro_auc_all'] for d in R])} & {ms([T(d)['macro_ece_toplabel_equal_width'] for d in R])} & "
                 f"{mean(ug):.3f} & {mean(uf):.3f} & {mean(uf)-mean(ug):+.3f} \\\\")
open(f"{OUT}/generated/rule_rows_tvc.tex", "w").write("\n".join(lines) + "\n")

new_beta = [("Dual stream (ViT + ResNet-50)", f"{TV}/ablations/ablation_evid_dirichlet", "outputs_mil2/tms_edl"),
            ("Single stream (ResNet-50)", f"{TV}/baselines/edl_single_resnet50", f"{TV}/ablations/ablation_stream_local_only")]
lines = []
for name, a, b in new_beta:
    A, B = load(a), load(b)
    if not A or not B: continue
    ia = [T(d)['macro_auc_all'] for d in A]; ib = [T(d)['macro_auc_all'] for d in B]
    ea = [EXT(d) for d in A]; eb = [EXT(d) for d in B]
    lines.append(f"{name} & {ms(ia)} & {ms(ib)} & {mean(ib)-mean(ia):+.4f} & {ms(ea)} & {ms(eb)} & {mean(eb)-mean(ea):+.4f} \\\\")
open(f"{OUT}/generated/beta_rows_tvc.tex", "w").write("\n".join(lines) + "\n")
print("tvc tables:", {k: have3(v) for k, v in [("vit", f"{TV}/baselines/vit_b16_ce"), ("yager", f"{TV}/ablations/ablation_fusion_yager"), ("dirichlet", f"{TV}/ablations/ablation_evid_dirichlet")]})

# ---------------- post-hoc: decoupled read-out and deep ensemble (needs scripts/tvc_posthoc.py first) ----------------
ph_path = f"{OUT}/generated/posthoc.json"
if os.path.exists(ph_path):
    ph = json.load(open(ph_path))
    order = [("Crop disabled (Dempster)", "crop disabled"), ("Uniform MIL (Dempster)", "uniform MIL"),
             ("Gated MIL (Dempster)", "gated MIL"),
             ("Gated MIL + NMS (Dempster)", "gated MIL + NMS"), ("Yager", "fusion_yager"),
             ("Mean-BBA (Murphy)", "fusion_cautious"), ("Opinion averaging", "fusion_average"),
             ("Learned gating", "fusion_gating"), ("Dempster + Shafer discounting", "discounting")]
    lines = []
    for name, key in order:
        v = ph["readout"].get(key)
        if not v: continue
        f = lambda k: f"{st.mean(v[k]):.3f}"
        lines.append(f"{name} & {st.mean(v['auc_fused']):.4f} & {f('imagenet_u_f')} & \\textbf{{{f('imagenet_u_g')}}} & {f('imagenet_max')} & "
                     f"{f('ct_liver_u_f')} & {f('ct_liver_u_g')} \\\\")
    open(f"{OUT}/generated/readout_rows.tex", "w").write("\n".join(lines) + "\n")
    e = ph.get("ensemble", {})
    if e:
        open(f"{OUT}/generated/ensemble_row.tex", "w").write(
            f"Deep ensemble ({e['n_members']}$\\times$ ViT-B/16 BCE), MI & {e['auc']:.4f} & {e['ece']:.4f} & -- & "
            f"{e['imagenet_mutual_info']:.3f} & {e['ct_liver_mutual_info']:.3f} \\\\\n")

# ---------------- appendix: calibration under every binning / notion ----------------
calib_rows = [("Global stream only", "outputs_fixed/ablations/ablation_stream_global_only"),
              ("Dual, crop disabled", "outputs_fixed/ablations/ablation_attn_crop_off"),
              ("Dual, uniform MIL", "outputs_tvc/ablations/ablation_mil_uniform"),
              ("Dual, gated MIL + NMS", "outputs_mil2/tms_edl"),
              ("Dual + Shafer discounting", "outputs_tvc/ablations/ablation_fusion_discounted"),
              ("ViT-B/16, BCE", "outputs_tvc/baselines/vit_b16_ce")]
lines = []
for name, run in calib_rows:
    R = load(run)[:3]
    c = lambda k: ms([T(d)[k] for d in R])
    lines.append(f"{name} & {c('macro_ece_toplabel_equal_width')} & {c('macro_ece_toplabel_equal_mass')} & "
                 f"{c('macro_ece_positive_equal_width')} & {c('macro_ece_positive_equal_mass')} \\\\")
open(f"{OUT}/generated/calib_rows.tex", "w").write("\n".join(lines) + "\n")

# ---------------- figure: graded corruption sweep ----------------
SW = [("global stream only", "outputs_fixed/ablations/ablation_stream_global_only", BLUE),
      ("dual stream, fused $u_f$", "outputs_mil2/tms_edl", AQUA)]
FAM = [("gaussian_noise", "Gaussian noise"), ("blur", "blur"), ("contrast", "contrast"), ("brightness", "brightness")]
SEV = [0, 1, 3, 5]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex=True)
for j, (fam, title) in enumerate(FAM):
    for name, run, col in SW:
        R = load(run)
        for i, key in enumerate(["auroc", "macro_auc"]):
            vals = np.array([[d["near_ood"][fam][f"severity_{s}"][key] for s in SEV] for d in R])
            ax = axes[i, j]
            ax.plot(SEV, vals.mean(0), color=col, lw=2, marker="o", ms=3, label=name)
            ax.fill_between(SEV, vals.min(0), vals.max(0), color=col, alpha=0.15, linewidth=0)
    axes[0, j].set_title(title, fontsize=8)
    axes[0, j].axhline(0.5, color=GREY, lw=0.8, ls=":")
    axes[0, j].set_ylim(0, 1); axes[1, j].set_ylim(0.5, 0.82)
    axes[1, j].set_xticks(SEV); axes[1, j].set_xlabel("severity")
axes[0, 0].set_ylabel("OOD AUROC of vacuity"); axes[1, 0].set_ylabel("macro AUC")
axes[0, 0].text(5, 0.52, "chance", color="#444", fontsize=6, ha="right")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{OUT}/figures/corruption_sweep.pdf")
