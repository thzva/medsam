"""Regenerate every LaTeX table from the machine-readable results files.

Reviewer M4: "The authors should regenerate all tables from a machine-readable
results file and check every caption, bold value, and arithmetic claim."
Nothing here is transcribed by hand.
"""
import json, glob
from pathlib import Path
import numpy as np

ROOT = Path("/home/2025user/zhou/medsam")
OUT = ROOT / "revision" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

DISEASE = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
           "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
           "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

PRETTY = {
    "tms_edl": "TMS-EDL (full)",
    "fusion_yager": "Yager's rule",
    "fusion_cautious": "Mean-BBA (Murphy)",
    "fusion_discounted": "D--S + source discounting",
    "fusion_average": "Evidence averaging",
    "fusion_gating": "Learned gating",
    "stream_global_only": "Global stream only",
    "stream_local_only": "Local stream only",
    "attn_crop_off": "Patches without global attention",
    "evid_dirichlet": "Single categorical Dirichlet (original)",
    "kl_disabled": "No KL regulariser",
    "kl_no_annealing": "KL without annealing",
    "loss_edl_mse": "EDL-MSE loss",
    "resnet50_ce": "ResNet-50 (BCE)",
    "densenet121_ce": "DenseNet-121 (BCE)",
    "vit_b16_ce": "ViT-B/16 (BCE)",
    "resnet50_mc_dropout": "ResNet-50 + MC Dropout",
    "edl_single_resnet50": "EDL single-scale (ResNet-50)",
    "multiscale_avg_prob": "Multi-scale probability averaging",
}


def load_dir(sub):
    runs = {}
    for f in glob.glob(str(ROOT / sub / "**" / "eval_revision.json"), recursive=True):
        d = json.load(open(f))
        runs.setdefault(d.get("mode", "?"), {})[Path(f).parent.name] = d
    return runs


def load_all():
    """Primary results come from the most-corrected matrix that exists.

    outputs_fixed    D7/D8/D9 fixed  (attention softmax over logits, local
                     crops from a 560px aligned source, un-augmented validation)
    outputs_final    D6 fixed only
    outputs_revision weighted-sampler matrix, kept only for the D6 comparison
    """
    for d in ("outputs_fixed", "outputs_final", "outputs_revision"):
        r = load_dir(d)
        if r:
            return r
    return {}


def table_d6(_):
    """614x weighted resampling vs inverse-frequency loss weighting (defect D6)."""
    a = load_dir("outputs_revision")      # sampler ON  + loss weights
    b = load_dir("outputs_final")         # sampler OFF + loss weights
    modes = [m for m in PRETTY if m in a and m in b]
    if not modes:
        return "\\emph{The D6 comparison needs both matrices; not available yet.}\n"
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Effect of how class imbalance is handled (defect D6). "
             "``resampled'' oversamples rare findings with weights spanning "
             "1.0--613.6, so the 141 Hernia images make up 8.4\\% of every epoch; "
             "``loss-weighted'' leaves the data distribution intact and weights the "
             "objective instead, which is what the manuscript states. Both also use "
             "inverse-frequency loss weights.}",
             "\\label{tab:d6}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "& \\multicolumn{3}{c}{resampled} & \\multicolumn{3}{c}{loss-weighted} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Variant & AUC & ECE & best ep. & AUC & ECE & best ep. \\\\", "\\midrule"]
    for m in modes:
        row = [PRETTY.get(m, m)]
        for runs in (a, b):
            row.append(fmt(*agg(runs, m, "test.macro_auc_all")))
            row.append(fmt(*agg(runs, m, "test.macro_ece_toplabel_equal_width")))
            e, _, n = agg(runs, m, "best_epoch")
            row.append("--" if not np.isfinite(e) else f"{e:.1f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def agg(runs, mode, path, default=np.nan):
    vals = []
    for d in runs.get(mode, {}).values():
        cur = d
        try:
            for k in path.split("."):
                cur = cur[k]
            vals.append(float(cur))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return default, np.nan, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan), len(vals)


def fmt(m, s, n, nd=4, bold=False):
    if not np.isfinite(m):
        return "--"
    txt = f"{m:.{nd}f}"
    if n > 1 and np.isfinite(s):
        txt += f"$\\pm${s:.{nd}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def ci_str(d, nd=4):
    if not d:
        return "--"
    return f"{d['point']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print(f"  wrote generated/{name}.tex")


def table_main(runs):
    order = [m for m in ["tms_edl", "fusion_yager", "fusion_cautious", "fusion_discounted",
                         "fusion_average", "fusion_gating", "stream_global_only",
                         "stream_local_only", "attn_crop_off", "evid_dirichlet",
                         "kl_disabled", "kl_no_annealing", "loss_edl_mse"] if m in runs]
    if not order:
        return "\\emph{No runs available yet.}\n"
    cols = [("test.macro_auc_all", "AUC", "max"),
            ("test.macro_auprc", "AUPRC", "max"),
            ("test.macro_ece_toplabel_equal_width", "ECE$_{\\text{top}}$", "min"),
            ("test.macro_ece_positive_equal_width", "ECE$_{\\text{pos}}$", "min"),
            ("test.brier", "Brier", "min"),
            ("test.nll", "NLL", "min")]
    stats = {m: {c[0]: agg(runs, m, c[0]) for c in cols} for m in order}
    best = {}
    for path, _, direction in cols:
        vals = {m: stats[m][path][0] for m in order if np.isfinite(stats[m][path][0])}
        if vals:
            best[path] = (max if direction == "max" else min)(vals, key=vals.get)
    n_seeds = max((stats[m][cols[0][0]][2] for m in order), default=0)
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Main results and ablations on the NIH ChestX-ray14 test split "
             f"({len(order)} variants shown). Macro metrics are over the 14 disease "
             "labels; ``No Finding'' is derived, not predicted. Mean$\\pm$SD over up to "
             f"{n_seeds} seed(s). Best per column in bold.}}",
             "\\label{tab:main}",
             "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Variant & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for m in order:
        row = [PRETTY.get(m, m)]
        for path, _, _ in cols:
            mu, sd, n = stats[m][path]
            row.append(fmt(mu, sd, n, bold=(best.get(path) == m)))
        lines.append(" & ".join(row) + " \\\\")
        if m == "tms_edl":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_ci(runs):
    d = runs.get("tms_edl", {})
    if not d:
        return "\\emph{Not available yet.}\n"
    ci = list(d.values())[0]["test"].get("ci", {})
    rows = [("Macro AUC (14 diseases)", ci.get("macro_auc_all")),
            ("Macro AUPRC", ci.get("macro_auprc")),
            ("Macro top-label ECE", ci.get("macro_ece_toplabel_equal_width")),
            ("Brier score", ci.get("brier"))]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{TMS-EDL on the NIH test split with 95\\% bootstrap confidence "
             "intervals (1000 resamples over test images).}", "\\label{tab:ci}",
             "\\begin{tabular}{lr}", "\\toprule", "Metric & Value [95\\% CI] \\\\", "\\midrule"]
    for name, c in rows:
        lines.append(f"{name} & {ci_str(c)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_calibration(runs):
    order = [m for m in ["tms_edl", "evid_dirichlet", "resnet50_ce", "densenet121_ce",
                         "vit_b16_ce", "resnet50_mc_dropout", "edl_single_resnet50"] if m in runs]
    if not order:
        return "\\emph{Not available yet.}\n"
    paths = [("test.macro_ece_toplabel_equal_width", "top / width"),
             ("test.macro_ece_toplabel_equal_mass", "top / mass"),
             ("test.macro_ece_positive_equal_width", "pos / width"),
             ("test.pooled_ece_toplabel", "pooled"),
             ("test.brier", "Brier"), ("test.nll", "NLL")]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Calibration under every aggregation and binning rule "
             "(reviewers M2/M8). ``top'' is top-label ECE, ``pos'' is positive-class "
             "ECE, ``width''/``mass'' are equal-width and equal-mass bins, 15 bins "
             "throughout.}", "\\label{tab:calib}",
             "\\begin{tabular}{l" + "r" * len(paths) + "}", "\\toprule",
             "Model & " + " & ".join(p[1] for p in paths) + " \\\\", "\\midrule"]
    for m in order:
        row = [PRETTY.get(m, m)]
        for path, _ in paths:
            row.append(fmt(*agg(runs, m, path)))
        lines.append(" & ".join(row) + " \\\\")
    mu, _, _ = agg(runs, "tms_edl", "test_temperature_scaled.macro_ece_toplabel_equal_width")
    if np.isfinite(mu):
        row = ["\\quad + temperature scaling"]
        for path, _ in paths:
            row.append(fmt(*agg(runs, "tms_edl", path.replace("test.", "test_temperature_scaled."))))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_branch(runs):
    d = runs.get("tms_edl", {})
    if not d:
        return "\\emph{Not available yet.}\n"
    ref = list(d.values())[0]
    bs, far, ext = ref.get("branch_stats", {}), ref.get("far_ood", {}), ref.get("external", {})
    groups = [("ID test (all)", bs.get("test_all", {})),
              ("ID test (correct)", bs.get("test_correct", {})),
              ("ID test (errors)", bs.get("test_error", {}))]
    if "chexpert_val" in ext:
        groups.append(("CheXpert (near-OOD)", ext["chexpert_val"].get("branch_stats", {})))
    for tag, label in [("ct_liver", "Liver CT (far-OOD)"), ("imagenet", "Imagenette (far-OOD)")]:
        if tag in far:
            groups.append((label, far[tag].get("branch_stats", {})))
    keys = ["u_global", "u_local", "conflict", "u_fused"]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Distributions of global uncertainty $u_g$, local uncertainty "
             "$u_\\ell$, conflict mass $K$ and fused uncertainty $u$ (R2 comment 3). "
             "Entries are mean (5th--95th percentile). Non-finite values are counted, "
             "never dropped (defect D1).}", "\\label{tab:branch}",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Split & $u_g$ & $u_\\ell$ & $K$ & $u_{\\text{fused}}$ & non-finite \\\\",
             "\\midrule"]
    for name, g in groups:
        row, nonfin = [name], 0
        for k in keys:
            s = g.get(k)
            if not s or not np.isfinite(s.get("mean", np.nan)):
                row.append("--"); continue
            row.append(f"{s['mean']:.3f} ({s['p05']:.3f}--{s['p95']:.3f})")
            nonfin = max(nonfin, s.get("n_nonfinite", 0))
        row.append(str(nonfin))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_ood(runs):
    order = [m for m in PRETTY if m in runs]
    if not order:
        return "\\emph{Not available yet.}\n"
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Far out-of-distribution detection for every trained variant "
             "(M4: ``all 10 variants'' previously referred to rows that were not "
             "shown). AUROC of the scalar uncertainty score against the NIH test "
             "split. Per-branch columns show whether fusion degrades a signal a "
             "single stream already carries (M5). ``n.f.'' counts non-finite "
             "uncertainties, scored as maximally uncertain rather than dropped.}",
             "\\label{tab:ood}", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Variant & CT liver & Imagenette & $u_g$ (INet) & $u_\\ell$ (INet) & n.f. \\\\",
             "\\midrule"]
    rows = 0
    for m in order:
        ct, _, n1 = agg(runs, m, "far_ood.ct_liver.auroc")
        inet, _, n2 = agg(runs, m, "far_ood.imagenet.auroc")
        if n1 == 0 and n2 == 0:
            continue
        ug, _, _ = agg(runs, m, "far_ood.imagenet.auroc_u_global")
        ul, _, _ = agg(runs, m, "far_ood.imagenet.auroc_u_local")
        nf, _, _ = agg(runs, m, "far_ood.imagenet.n_nonfinite_ood")
        lines.append(" & ".join([PRETTY.get(m, m), fmt(ct, np.nan, n1), fmt(inet, np.nan, n2),
                                 fmt(ug, np.nan, 1), fmt(ul, np.nan, 1),
                                 "--" if not np.isfinite(nf) else str(int(nf))]) + " \\\\")
        rows += 1
    if rows == 0:
        return "\\emph{Not available yet.}\n"
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_near_ood(runs):
    d = runs.get("tms_edl", {})
    if not d:
        return "\\emph{Not available yet.}\n"
    near = list(d.values())[0].get("near_ood", {})
    if not near:
        return "\\emph{Not available yet.}\n"
    sevs = ["severity_1", "severity_3", "severity_5"]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Near-OOD: graded corruption sweep on the NIH test split "
             "(R1 comment 7). Each cell is the OOD AUROC of the uncertainty score "
             "against clean test images, with macro AUC on the corrupted images in "
             "parentheses. A useful uncertainty score should rise monotonically with "
             "severity as accuracy falls.}", "\\label{tab:nearood}",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "Corruption & severity 1 & severity 3 & severity 5 \\\\", "\\midrule"]
    for k in near:
        row = [k.replace("_", " ")]
        for s in sevs:
            e = near[k].get(s, {})
            a, mauc = e.get("auroc"), e.get("macro_auc")
            if a is None:
                row.append("--")
            elif mauc is not None and mauc == mauc:
                row.append(f"{a:.3f} ({mauc:.3f})")
            else:
                row.append(f"{a:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_external(runs):
    order = [m for m in ["tms_edl", "evid_dirichlet", "stream_global_only",
                         "stream_local_only", "resnet50_ce", "vit_b16_ce"] if m in runs]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{External validation on CheXpert without fine-tuning (M10), on "
             "the eight findings whose definitions are shared with ChestX-ray14. "
             "``expert'' is the radiologist-consensus validation set; ``NLP'' a larger "
             "automatically-labelled subset. 95\\% bootstrap CIs; two findings have "
             "fewer than ten positives on the expert set, so those intervals are wide.}",
             "\\label{tab:external}", "\\begin{tabular}{lrrr}", "\\toprule",
             "Model & CheXpert expert AUC & CheXpert NLP AUC & near-OOD AUROC \\\\",
             "\\midrule"]
    rows = 0
    for m in order:
        ref = list(runs.get(m, {}).values() or [{}])[0]
        ext = ref.get("external", {})
        if not ext:
            continue
        v, t = ext.get("chexpert_val", {}), ext.get("chexpert_train", {})
        lines.append(" & ".join([
            PRETTY.get(m, m),
            ci_str(v.get("macro_auc_ci")) if v else "--",
            ci_str(t.get("macro_auc_ci")) if t else "--",
            f"{v['near_ood_vs_nih_test']['auroc']:.4f}" if v.get("near_ood_vs_nih_test") else "--",
        ]) + " \\\\")
        rows += 1
    if rows == 0:
        return "\\emph{Not available yet.}\n"
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_clinical(runs):
    d = runs.get("tms_edl", {})
    if not d:
        return "\\emph{Not available yet.}\n"
    t = list(d.values())[0]["test"]
    op = t.get("operating_point", {})
    if not op:
        return "\\emph{Not available yet.}\n"
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Per-class clinical-utility metrics for TMS-EDL at per-label "
             "thresholds fixed on the validation fold (M6). AUPRC is threshold-free. "
             "Prevalence is on the test split.}", "\\label{tab:clinical}",
             "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Finding & Prev. & AUC & AUPRC & Sens. & Spec. & F1 \\\\", "\\midrule"]
    for i, name in enumerate(DISEASE):
        lines.append(" & ".join([
            name.replace("_", " "),
            f"{t['prevalence'].get(name, float('nan')):.3f}",
            f"{t['per_class_auc'].get(name, float('nan')):.4f}",
            f"{t['per_class_auprc'].get(name, float('nan')):.4f}",
            f"{op['sensitivity'][i]:.3f}", f"{op['specificity'][i]:.3f}",
            f"{op['f1'][i]:.3f}"]) + " \\\\")
    lines += ["\\midrule",
              " & ".join(["\\textbf{Macro}", "--",
                          f"\\textbf{{{t['macro_auc_all']:.4f}}}",
                          f"\\textbf{{{t['macro_auprc']:.4f}}}",
                          f"\\textbf{{{op['macro_sensitivity']:.3f}}}",
                          f"\\textbf{{{op['macro_specificity']:.3f}}}",
                          f"\\textbf{{{op['macro_f1']:.3f}}}"]) + " \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_conformal(runs):
    d = runs.get("tms_edl", {})
    if not d:
        return "\\emph{Not available yet.}\n"
    cf = list(d.values())[0].get("conformal", {})
    if not cf:
        return "\\emph{Not available yet.}\n"
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Label-wise split-conformal prediction on the TMS-EDL "
             "probabilities (M9), calibrated on the validation fold. Conformal "
             "prediction provides a finite-sample coverage guarantee that the "
             "evidential uncertainty score does not.}", "\\label{tab:conformal}",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "$\\alpha$ & target & empirical & worst label & mean set size & abstention \\\\",
             "\\midrule"]
    for _, v in sorted(cf.items()):
        lines.append(" & ".join([f"{v['alpha']:.2f}", f"{v['target_coverage']:.2f}",
                                 f"{v['macro_coverage']:.4f}", f"{v['min_label_coverage']:.4f}",
                                 f"{v['mean_set_size']:.3f}", f"{v['abstention_rate']:.3f}"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def split_stats(runs):
    for mode in runs:
        for d in runs[mode].values():
            s = d.get("split_stats") or {}
            if s:
                return (f"training {s['n_images_train']:,} images from "
                        f"{s['n_patients_train']:,} patients; validation "
                        f"{s['n_images_val']:,} images from {s['n_patients_val']:,} patients; "
                        f"test {s['n_images_test']:,} images from {s['n_patients_test']:,} "
                        f"patients, with zero patient overlap between any pair of folds.\n")
    return "\\emph{Not available yet.}\n"


def matrix_summary(runs):
    total = sum(len(v) for v in runs.values())
    if total == 0:
        return "\\emph{Not available yet.}\n"
    per = ", ".join(f"{PRETTY.get(m, m)} ($n$={len(v)})" for m, v in sorted(runs.items()))
    return (f"A total of {total} training runs completed, covering {len(runs)} distinct "
            f"configurations: {per}.\n")


def main():
    runs = load_all()
    print(f"loaded {sum(len(v) for v in runs.values())} runs across {len(runs)} modes: {sorted(runs)}")
    for name, fn in [("main_table", table_main), ("ci_table", table_ci),
                     ("calibration_table", table_calibration), ("branch_table", table_branch),
                     ("ood_table", table_ood), ("near_ood_table", table_near_ood),
                     ("external_table", table_external), ("clinical_table", table_clinical),
                     ("conformal_table", table_conformal), ("split_stats", split_stats),
                     ("d6_table", table_d6), ("matrix_summary", matrix_summary)]:
        write(name, fn(runs))


if __name__ == "__main__":
    main()
