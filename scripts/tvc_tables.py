"""Styled LaTeX tables for the TVC manuscript, generated from saved results.

Every table body is written to revision/tvc/generated/tab_*.tex and contains only
rows; the column specification, header and colours live in main.tex. In each
numeric column the best value is highlighted with \\best{}; rows can be grouped
with a shaded group header. Run from the repo root after scripts/tvc_posthoc.py."""
import glob, json, os, statistics as st

OUT = "revision/tvc/generated"
os.makedirs(OUT, exist_ok=True)
TV = "outputs_tvc"
R = {  # run directories, all under the final protocol
    "global": "outputs_fixed/ablations/ablation_stream_global_only",
    "local": f"{TV}/ablations/ablation_stream_local_only",
    "crop_off": "outputs_fixed/ablations/ablation_attn_crop_off",
    "uniform": f"{TV}/ablations/ablation_mil_uniform",
    "gated": "outputs_mil2/ablations/ablation_mil_no_nms",
    "gated_nms": "outputs_mil2/tms_edl",
    "dempster_rerun": f"{TV}/tms_edl",
    "discount": f"{TV}/ablations/ablation_fusion_discounted",
    "yager": f"{TV}/ablations/ablation_fusion_yager",
    "meanbba": f"{TV}/ablations/ablation_fusion_cautious",
    "average": f"{TV}/ablations/ablation_fusion_average",
    "gating": f"{TV}/ablations/ablation_fusion_gating",
    "dir_dual": f"{TV}/ablations/ablation_evid_dirichlet",
    "dir_res": f"{TV}/baselines/edl_single_resnet50",
    "kl_off": f"{TV}/ablations/ablation_kl_disabled",
    "kl_noanneal": f"{TV}/ablations/ablation_kl_no_annealing",
    "mse": f"{TV}/ablations/ablation_loss_edl_mse",
    "vit_bce": f"{TV}/baselines/vit_b16_ce",
    "dense_bce": f"{TV}/baselines/densenet121_ce",
    "res_bce": f"{TV}/baselines/resnet50_ce",
    "mc": f"{TV}/baselines/resnet50_mc_dropout",
}


def load(key, n=3):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{R[key]}/seed*/eval_revision.json"))[:n]]


M = {  # metric extractors
    "auc": lambda d: d["test"]["macro_auc_all"],
    "auprc": lambda d: d["test"]["macro_auprc"],
    "ece": lambda d: d["test"]["macro_ece_toplabel_equal_width"],
    "ece_ts": lambda d: d["test_temperature_scaled"]["macro_ece_toplabel_equal_width"],
    "brier": lambda d: d["test"]["brier"],
    "nll": lambda d: d["test"]["nll"],
    "ood_f": lambda d: d["far_ood"]["imagenet"]["auroc"],
    "ood_g": lambda d: d["far_ood"]["imagenet"].get("auroc_u_global", d["far_ood"]["imagenet"]["auroc"]),
    "ood_l": lambda d: d["far_ood"]["imagenet"].get("auroc_u_local"),
    "ct_f": lambda d: d["far_ood"]["ct_liver"]["auroc"],
    "chex": lambda d: d["external"]["chexpert_val"]["macro_auc"],
    "e_eqm": lambda d: d["test"]["macro_ece_toplabel_equal_mass"],
    "e_pos": lambda d: d["test"]["macro_ece_positive_equal_width"],
    "e_posm": lambda d: d["test"]["macro_ece_positive_equal_mass"],
}


class Cell:
    """A numeric cell: mean, optional std, display precision."""
    def __init__(self, vals, nd=4, std=False):
        vals = [v for v in vals if v is not None]
        self.m = st.mean(vals) if vals else None
        self.s = st.stdev(vals) if (std and len(vals) > 1) else None
        self.nd = nd

    def fmt(self, best=False):
        if self.m is None:
            return "n/a"
        t = f"{self.m:.{self.nd}f}"
        if self.s is not None:
            sd = f"{self.s:.{self.nd}f}".lstrip("0")
            t += f"{{\\scriptsize$\\,\\pm${sd}}}"
        return f"\\best{{{t}}}" if best else t


def cell(key, metric, nd=4, std=False):
    D = load(key)
    if len(D) < 3:
        return None
    return Cell([M[metric](d) for d in D], nd, std)


def write(name, groups, higher):
    """groups: list of (group title or None, [(label, [Cell or str or None, ...]), ...]).
    higher: list of True / False / None per column (None = no highlighting)."""
    ncol = len(higher)
    best = [None] * ncol
    for j, h in enumerate(higher):
        if h is None:
            continue
        vals = [r[1][j].m for g in groups for r in g[1] if isinstance(r[1][j], Cell) and r[1][j].m is not None]
        if vals:
            best[j] = max(vals) if h else min(vals)
    lines, stripe = [], False
    for title, rows in groups:
        if title:
            lines.append(f"\\grouprow{{{ncol + 1}}}{{{title}}}")
        for label, cells in rows:
            out = []
            for j, c in enumerate(cells):
                if c is None:
                    out.append("\\running")
                elif isinstance(c, Cell):
                    out.append(c.fmt(best[j] is not None and c.m is not None and abs(c.m - best[j]) < 1e-12))
                else:
                    out.append(str(c))
            lines.append(label + " & " + " & ".join(out) + " \\\\")
            stripe = not stripe
    open(f"{OUT}/tab_{name}.tex", "w").write("\n".join(lines) + "\n")


def row(key, metrics, nd=None):
    nd = nd or [4] * len(metrics)
    return [cell(key, m, n) for m, n in zip(metrics, nd)]


# ---------------- Table: main comparison
mm = ["auc", "auprc", "ece", "brier", "nll"]
write("main", [
    ("Single stream", [("Global stream only, ViT-B/16", row("global", mm))]),
    ("Dual stream, Dempster fusion", [
        ("Crop disabled", row("crop_off", mm)),
        ("Uniform MIL", row("uniform", mm)),
        ("Gated MIL", row("gated", mm)),
        ("Gated MIL + NMS", row("gated_nms", mm)),
        ("Gated MIL + NMS, Shafer discounting", row("discount", mm))]),
], [True, True, False, False, False])

# ---------------- Table: baselines
bm = ["auc", "ece", "ece_ts", "ood_f", "ct_f"]
bnd = [4, 4, 4, 3, 3]
ph = json.load(open("revision/tvc/generated/posthoc.json")) if os.path.exists("revision/tvc/generated/posthoc.json") else {}
ens = ph.get("ensemble", {})
ens_row = [Cell([ens["auc"]], 4, False), Cell([ens["ece"]], 4, False), "n/a",
           Cell([ens["imagenet_mutual_info"]], 3, False), Cell([ens["ct_liver_mutual_info"]], 3, False)] if ens else [None] * 5
write("baselines", [
    ("Deterministic and sampling baselines", [
        ("ViT-B/16, BCE", row("vit_bce", bm, bnd)),
        ("DenseNet-121, BCE", row("dense_bce", bm, bnd)),
        ("ResNet-50, BCE", row("res_bce", bm, bnd)),
        ("ResNet-50, MC dropout", row("mc", bm, bnd)),
        ("Deep ensemble, 5$\\times$ ViT-B/16 BCE", ens_row)]),
    ("Evidential, ours", [
        ("ViT-B/16 Beta EDL, global stream", row("global", bm, bnd)),
        ("Dual stream, gated MIL + NMS", row("gated_nms", bm, bnd))]),
], [True, False, False, True, True])

# ---------------- Table: Beta vs Dirichlet
def delta(a, b, metric):
    A, B = cell(a, metric), cell(b, metric)
    if A is None or B is None:
        return None
    return f"{B.m - A.m:.3f}"
write("beta", [
    (None, [
        ("Dual stream", [cell("dir_dual", "auc", 4, False), cell("gated_nms", "auc", 4, False), delta("dir_dual", "gated_nms", "auc"),
                         cell("dir_dual", "chex", 4, False), cell("gated_nms", "chex", 4, False), delta("dir_dual", "gated_nms", "chex"),
                         cell("dir_dual", "ood_f", 3, False), cell("gated_nms", "ood_f", 3, False)]),
        ("Single stream, ResNet-50", [cell("dir_res", "auc", 4, False), cell("local", "auc", 4, False), delta("dir_res", "local", "auc"),
                                      cell("dir_res", "chex", 4, False), cell("local", "chex", 4, False), delta("dir_res", "local", "chex"),
                                      cell("dir_res", "ood_f", 3, False), cell("local", "ood_f", 3, False)])]),
], [None] * 8)

# ---------------- Table: branch-wise OOD
om = ["ood_g", "ood_l", "ood_f", "ct_f", "chex"]
ond = [3, 3, 3, 3, 4]
def orow(key):
    c = row(key, om, ond)
    if key == "global":
        c[1] = "n/a"
    return c
write("ood", [
    (None, [("Global stream only", orow("global"))]),
    ("Dual stream, Dempster fusion", [
        ("Crop disabled", orow("crop_off")), ("Uniform MIL", orow("uniform")),
        ("Gated MIL", orow("gated")), ("Gated MIL + NMS", orow("gated_nms")),
        ("Gated MIL + NMS, Shafer discounting", orow("discount"))]),
], [True, None, True, True, True])

# ---------------- Table: combination rules
rm = ["auc", "ece", "ood_g", "ood_f"]
rnd = [4, 4, 3, 3]
def rrow(key):
    c = row(key, rm, rnd)
    c.append(f"{c[3].m - c[2].m:.3f}" if c[2] and c[3] else None)
    return c
write("rules", [
    ("Conjunctive", [("Dempster", rrow("gated_nms")), ("Dempster, independent re-run", rrow("dempster_rerun")),
                     ("Dempster + Shafer discounting", rrow("discount")), ("Yager", rrow("yager"))]),
    ("Averaging and learned", [("Mean-BBA of Murphy", rrow("meanbba")), ("Opinion averaging", rrow("average")),
                               ("Learned gating", rrow("gating"))]),
], [True, False, None, True, None])

# ---------------- Table: unified ablation
am = ["auc", "ece", "ood_f", "chex"]
and_ = [4, 4, 3, 4]
write("ablation", [
    ("Reference", [("Full model", row("gated_nms", am, and_))]),
    ("Streams", [("Global stream only", row("global", am, and_)), ("Local stream only", row("local", am, and_))]),
    ("Evidential parameterisation", [("Categorical Dirichlet instead of per-label Beta", row("dir_dual", am, and_))]),
    ("Local stream", [("Crop disabled", row("crop_off", am, and_)), ("Uniform MIL", row("uniform", am, and_)),
                      ("Gated MIL without NMS", row("gated", am, and_))]),
    ("Combination rule", [("Yager", row("yager", am, and_)), ("Mean-BBA of Murphy", row("meanbba", am, and_)),
                          ("Opinion averaging", row("average", am, and_)), ("Learned gating", row("gating", am, and_)),
                          ("Shafer discounting", row("discount", am, and_))]),
], [True, False, True, True])

# ---------------- Table: decoupled read-out (from posthoc.json)
order = [("Crop disabled", "crop disabled"), ("Uniform MIL", "uniform MIL"), ("Gated MIL", "gated MIL"),
         ("Gated MIL + NMS", "gated MIL + NMS"), ("Yager", "fusion_yager"), ("Mean-BBA of Murphy", "fusion_cautious"),
         ("Opinion averaging", "fusion_average"), ("Learned gating", "fusion_gating"),
         ("Shafer discounting", "discounting")]
rows = []
for name, k in order:
    v = ph.get("readout", {}).get(k)
    if not v:
        continue
    c = lambda key, nd=3: Cell(v[key], nd, False)
    rows.append((name, [c("auc_fused", 4), c("imagenet_u_f"), c("imagenet_u_g"), c("imagenet_max"),
                        c("ct_liver_u_f"), c("ct_liver_u_g")]))
write("readout", [(None, rows)], [None, None, True, None, None, True])

# ---------------- Table: calibration variants
cm = ["ece", "e_eqm", "e_pos", "e_posm"]
write("calib", [(None, [(n, row(k, cm)) for n, k in [
    ("Global stream only", "global"), ("Dual, crop disabled", "crop_off"), ("Dual, uniform MIL", "uniform"),
    ("Dual, gated MIL + NMS", "gated_nms"), ("Dual, Shafer discounting", "discount"), ("ViT-B/16, BCE", "vit_bce")]])],
    [False, False, False, False])

# ---------------- Table: efficiency
ef = json.load(open("revision/tvc/generated/efficiency.json"))
ren = {"ResNet-50, MC dropout (10 passes)": "ResNet-50, MC dropout, 10 passes",
       "Deep ensemble (5$\\times$ ViT-B/16)": "Deep ensemble, 5$\\times$ ViT-B/16",
       "ViT-B/16, Beta EDL (global stream)": "ViT-B/16 Beta EDL, global stream"}
erows = []
for name, r in ef.items():
    erows.append((ren.get(name, name), [Cell([r["params_M"]], 1, False), Cell([r["latency_ms"]], 1, False),
                                        Cell([r["throughput"]], 0, False), Cell([r["train_step_s"]], 2, False),
                                        Cell([r["peak_mem_GB"]], 1, False)]))
write("efficiency", [(None, erows)], [None, False, True, False, False])
print("tables written:", sorted(os.path.basename(p) for p in glob.glob(f"{OUT}/tab_*.tex")))
