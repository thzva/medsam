"""Comprehensive post-training evaluation for the revision.

One inference pass per split, everything else derived from cached per-sample
arrays.  Writes:
    <run_dir>/eval_revision.json   all metrics
    <run_dir>/arrays.npz           per-sample probs / targets / u_g / u_l / K / u_f

Splits
------
val            NIH validation fold  (thresholds, temperature, conformal calib)
test           NIH official test split
chexpert_val   CheXpert expert-labelled validation set   (EXTERNAL, M10)
chexpert_train CheXpert NLP-labelled shards              (EXTERNAL, larger)
corrupt_*      NIH test + graded corruption              (NEAR-OOD, R1-7)
ct_liver       liver CT slices                           (FAR OOD)
imagenet       natural images                            (FAR OOD)
"""
import os
# Bound the intra-op thread pools BEFORE torch is imported.  Unset, each
# process opens one OpenMP thread per core (80 here); with several training
# and evaluation jobs running concurrently that produced ~1.5M context
# switches per second and a load average of 400+ while the CPU sat 62% idle.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import os, sys, json, argparse, logging, glob
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
torch.set_num_threads(4)
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.training.utils import load_config
from src.data.builder import build_dataloaders
from src.data.ood_dataset import OODDataset
from src.data.external_cxr import CheXpertParquet, SHARED_NIH_IDX, SHARED_NAMES
from src.data.corruption_ood import CorruptedNIH, CORRUPTIONS
from src.models.opinion_utils import opinion_prob, opinion_uncertainty
from src.evaluation.metrics_v2 import (full_metrics, tune_thresholds, ood_auroc,
                                       macro_auc, bootstrap_ci, reliability_curve)
from src.evaluation.conformal import (fit_temperature, apply_temperature,
                                      calibrate_conformal, conformal_metrics)
from train_ddp import build_model_for_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DISEASE_NAMES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                 "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
                 "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


@torch.no_grad()
def collect(model, loader, device, want_branches=True, max_items=None):
    """Run inference, returning per-sample arrays.  NaNs are NOT dropped."""
    model.eval()
    P, T, UG, UL, UF, K = [], [], [], [], [], []
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        hr = batch.get("image_hr")
        hr = hr.to(device, non_blocking=True) if hr is not None else None
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = model(x, hr=hr) if hr is not None else model(x)
        fused = out["fused"] if "fused" in out else out
        P.append(opinion_prob(fused).float().cpu().numpy())
        UF.append(opinion_uncertainty(fused).float().cpu().numpy())
        if batch.get("label") is not None:
            T.append(batch["label"].numpy())
        if want_branches and "global" in out and "local" in out:
            UG.append(opinion_uncertainty(out["global"]).float().cpu().numpy())
            UL.append(opinion_uncertainty(out["local"]).float().cpu().numpy())
            c = fused.get("conflict")
            K.append((c.float().mean(-1) if c is not None and c.dim() > 1
                      else c.float()).cpu().numpy() if c is not None
                     else np.full(len(x), np.nan))
        n += len(x)
        if max_items and n >= max_items:
            break
    cat = lambda L: np.concatenate(L) if L else None
    return dict(probs=cat(P), targets=cat(T), u_fused=cat(UF),
                u_global=cat(UG), u_local=cat(UL), conflict=cat(K))


def branch_stats(a, mask=None, prefix=""):
    """Distribution summary of the branch-wise uncertainties (R2-3 / M5)."""
    out = {}
    for key in ("u_global", "u_local", "conflict", "u_fused"):
        v = a.get(key)
        if v is None:
            continue
        v = v[mask] if mask is not None else v
        v = np.asarray(v, dtype=float)
        fin = v[np.isfinite(v)]
        out[prefix + key] = {
            "mean": float(fin.mean()) if fin.size else float("nan"),
            "std": float(fin.std()) if fin.size else float("nan"),
            "p05": float(np.percentile(fin, 5)) if fin.size else float("nan"),
            "p50": float(np.percentile(fin, 50)) if fin.size else float("nan"),
            "p95": float(np.percentile(fin, 95)) if fin.size else float("nan"),
            "n_nonfinite": int((~np.isfinite(v)).sum()),
            "n": int(v.size),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mode", default="tms_edl")
    ap.add_argument("--config", default="config/revision.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--id_cap", type=int, default=None)
    ap.add_argument("--corrupt_n", type=int, default=2000)
    # TVC runs: control (0) and strongest (5) only; the full 0,1,3,5 sweep was
    # run for the main configurations and costs ~4x more CPU under load.
    ap.add_argument("--near_sevs", default="0,5")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    cfg = load_config(args.config)

    model, run_cfg = build_model_for_mode(args.mode, cfg, device)
    ck = torch.load(run_dir / "best_model.pth", map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in
          ck.get("model_state_dict", ck).items()}
    model.load_state_dict(sd)
    log.info(f"loaded {run_dir}  epoch={ck.get('epoch')}  val={ck.get('best_val_metric')}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg, rank=-1, world_size=1)
    HR = cfg["data"].get("hr_size", 0)
    bs = cfg["training"]["batch_size"]
    # Evaluation loaders are re-created per OOD/corruption set; with many jobs in
    # parallel 16 workers each exhausted host RAM, so cap them.
    nw = min(cfg["data"].get("num_workers", 16), int(os.environ.get("EVAL_WORKERS", 4)))
    res = {"run_dir": str(run_dir), "mode": args.mode,
           "best_epoch": int(ck.get("epoch", -1)),
           "val_metric_at_best": float(ck.get("best_val_metric", float("nan"))),
           "split_stats": cfg.get("_split_stats", {})}

    # ── in-distribution ────────────────────────────────────────────────
    A_val = collect(model, val_loader, device)
    A_test = collect(model, test_loader, device, max_items=args.id_cap)
    log.info("ID inference done")

    thr = tune_thresholds(A_val["probs"], A_val["targets"])
    res["thresholds_from_val"] = thr.tolist()

    res["test"] = full_metrics(A_test["probs"], A_test["targets"], DISEASE_NAMES,
                               thresholds=thr, n_boot=args.n_boot)
    res["val"] = full_metrics(A_val["probs"], A_val["targets"], DISEASE_NAMES,
                              thresholds=thr, n_boot=0)

    # Derived "No Finding" (M1): P(no finding) = prod_c (1 - p_c)
    nf_p = np.prod(1.0 - A_test["probs"], axis=1)
    nf_y = (A_test["targets"].sum(1) == 0).astype(float)
    from sklearn.metrics import roc_auc_score
    res["derived_no_finding_auc"] = float(roc_auc_score(nf_y, nf_p))
    res["derived_no_finding_prevalence"] = float(nf_y.mean())

    # ── post-hoc calibration / conformal baselines (M9) ────────────────
    T_lab = fit_temperature(A_val["probs"], A_val["targets"], per_label=True)
    res["temperature_per_label"] = T_lab.tolist()
    p_ts = apply_temperature(A_test["probs"], T_lab)
    res["test_temperature_scaled"] = full_metrics(p_ts, A_test["targets"],
                                                  DISEASE_NAMES, thresholds=thr, n_boot=0)
    res["conformal"] = {}
    for alpha in (0.10, 0.05):
        q = calibrate_conformal(A_val["probs"], A_val["targets"], alpha)
        res["conformal"][f"alpha_{alpha}"] = conformal_metrics(
            A_test["probs"], A_test["targets"], q, alpha)
    # Diagnostic: split-conformal only guarantees coverage when the calibration
    # and evaluation sets are exchangeable.  Calibrating on the validation fold
    # (drawn from the official train list) and evaluating on the official test
    # list violates that.  Re-running the same procedure with both halves drawn
    # from the test split isolates how much of any under-coverage is due to the
    # official split shift rather than to a defect in the method.
    res["conformal_within_test"] = {}
    rs = np.random.default_rng(0)
    perm = rs.permutation(len(A_test["probs"]))
    half = len(perm) // 2
    ca, ev = perm[:half], perm[half:]
    for alpha in (0.10, 0.05):
        q2 = calibrate_conformal(A_test["probs"][ca], A_test["targets"][ca], alpha)
        res["conformal_within_test"][f"alpha_{alpha}"] = conformal_metrics(
            A_test["probs"][ev], A_test["targets"][ev], q2, alpha)

    # ── branch diagnostics: ID, correct vs error (R2-3, M5) ────────────
    pred = (A_test["probs"] >= thr[None, :]).astype(int)
    err_any = (pred != A_test["targets"]).any(1)
    res["branch_stats"] = {"test_all": branch_stats(A_test),
                           "test_correct": branch_stats(A_test, ~err_any),
                           "test_error": branch_stats(A_test, err_any)}

    # ── reliability curves for plotting ────────────────────────────────
    rel = {}
    for k, name in enumerate(DISEASE_NAMES):
        c, f, w = reliability_curve(A_test["probs"][:, k], A_test["targets"][:, k])
        rel[name] = {"conf": c.tolist(), "freq": f.tolist(), "weight": w.tolist()}
    res["reliability"] = rel

    # ── EXTERNAL validation: CheXpert (M10) ────────────────────────────
    ext = {}
    cx_dir = Path("data/external_cxr")
    for tag, files, cap in [
        ("chexpert_val", [cx_dir / "chexpert_val.parquet"], None),
        ("chexpert_train", sorted(glob.glob(str(cx_dir / "chexpert_train-*.parquet"))), 5000),
    ]:
        files = [f for f in files if Path(f).exists()]
        if not files:
            continue
        ds = CheXpertParquet(files, image_size=cfg["data"]["image_size"], max_items=cap, hr_size=HR)
        dl = DataLoader(ds, batch_size=bs, num_workers=nw)
        A = collect(model, dl, device)
        # only the 8 shared findings; map NIH-15 index -> 14-disease index
        cols = [i - 1 for i in SHARED_NIH_IDX if i >= 1]
        names = [n for i, n in zip(SHARED_NIH_IDX, SHARED_NAMES) if i >= 1]
        y = A["targets"][:, [i for i in SHARED_NIH_IDX if i >= 1]]
        p = A["probs"][:, cols]
        keep = [j for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
        ext[tag] = {
            "n": int(len(ds)),
            "shared_labels": [names[j] for j in keep],
            "macro_auc": macro_auc(p[:, keep], y[:, keep]),
            "macro_auc_ci": bootstrap_ci(lambda a, b: macro_auc(a, b),
                                         p[:, keep], y[:, keep], n_boot=args.n_boot),
            "per_label_auc": {names[j]: float(roc_auc_score(y[:, j], p[:, j])) for j in keep},
            "prevalence": {names[j]: float(y[:, j].mean()) for j in keep},
            "mean_u": float(np.nanmean(A["u_fused"])),
            "branch_stats": branch_stats(A),
        }
        # CheXpert also serves as NEAR-OOD relative to NIH
        ext[tag]["near_ood_vs_nih_test"] = ood_auroc(A_test["u_fused"], A["u_fused"])
        log.info(f"external {tag}: macro AUC {ext[tag]['macro_auc']:.4f}")
    res["external"] = ext

    # ── NEAR-OOD: graded corruption sweep (R1-7) ───────────────────────
    near = {}
    base_test = test_loader.dataset
    for kind in CORRUPTIONS:
        near[kind] = {}
        for sev in [int(v) for v in args.near_sevs.split(',')]:
            ds = CorruptedNIH(base_test, kind, sev, cfg["data"]["image_size"], hr_size=HR)
            sub = torch.utils.data.Subset(ds, list(range(min(args.corrupt_n, len(ds)))))
            dl = DataLoader(sub, batch_size=bs, num_workers=nw)
            A = collect(model, dl, device)
            m = ood_auroc(A_test["u_fused"][:len(sub)], A["u_fused"])
            m["macro_auc"] = macro_auc(A["probs"], A["targets"]) \
                if A["targets"] is not None else float("nan")
            m["branch_stats"] = branch_stats(A)
            near[kind][f"severity_{sev}"] = m
            log.info(f"near-OOD {kind} s{sev}: AUROC {m['auroc']:.4f} AUC {m['macro_auc']:.4f}")
    res["near_ood"] = near

    # ── FAR OOD ────────────────────────────────────────────────────────
    far = {}
    far_arrays = {}
    for name, path in [("ct_liver", "data/ood_ct_liver"), ("imagenet", "data/ood_imagenet")]:
        if not Path(path).exists():
            continue
        dl = DataLoader(OODDataset(path, image_size=cfg["data"]["image_size"], hr_size=HR),
                        batch_size=bs, num_workers=nw)
        A = collect(model, dl, device)
        far_arrays[name] = A
        far[name] = ood_auroc(A_test["u_fused"], A["u_fused"])
        far[name]["branch_stats"] = branch_stats(A)
        # Per-branch OOD AUROC: does fusion DESTROY a signal a branch had? (M5)
        for br in ("u_global", "u_local"):
            if A.get(br) is not None and A_test.get(br) is not None:
                far[name][f"auroc_{br}"] = ood_auroc(A_test[br], A[br])["auroc"]
        log.info(f"far-OOD {name}: fused AUROC {far[name]['auroc']:.4f} "
                 f"(nonfinite ood={far[name]['n_nonfinite_ood']})")
    res["far_ood"] = far

    with open(run_dir / "eval_revision.json", "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(run_dir / "arrays.npz",
                        **{f"test_{k}": v for k, v in A_test.items() if v is not None},
                        **{f"val_{k}": v for k, v in A_val.items() if v is not None},
                        **{f"ood_{n}_{k}": v for n, A in far_arrays.items()
                           for k, v in A.items() if v is not None},
                        thresholds=thr)
    log.info(f"wrote {run_dir/'eval_revision.json'}")


if __name__ == "__main__":
    main()
