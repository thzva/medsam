"""Qualitative figure for the TVC manuscript.

For four NIH test radiographs (chosen automatically from saved predictions: two
true positives, one false negative, one false positive) and one natural / one CT
OOD image, show the local stream's attention map, the six selected crops with
their MIL pooling weights, and the per-image vacuities (u_g, u_l, u_f), for the
gated MIL + NMS model and the uniform-MIL model (seed 42 of each).
Writes revision/tvc/figures/qualitative.pdf. Run from the repo root."""
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.training.utils import load_config
from src.data.builder import build_dataloaders
from src.data.ood_dataset import OODDataset
from src.models.opinion_utils import opinion_prob, opinion_uncertainty
from train_ddp import build_model_for_mode

NAMES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
         "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural thickening", "Hernia"]
DEV = torch.device("cuda:0")
CFG = load_config("config/revision_tvc.yaml")
HR = CFG["data"]["hr_size"]
MODELS = [("gated MIL + NMS", "tms_edl", "outputs_mil2/tms_edl/seed42"),
          ("uniform MIL", "mil_uniform", "outputs_tvc/ablations/ablation_mil_uniform/seed42")]
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

# ---- pick NIH cases from the gated model's saved test predictions
a = np.load(f"{MODELS[0][2]}/arrays.npz")
P, Y, thr = a["test_probs"], a["test_targets"], a["thresholds"]
single = Y.sum(1) == 1
def pick(cls, positive, predicted, rank):
    m = single & (Y[:, cls] == positive) & ((P[:, cls] >= thr[cls]) == predicted)
    idx = np.where(m)[0]
    order = np.argsort(-P[idx, cls]) if predicted else np.argsort(P[idx, cls])
    return int(idx[order[rank]])
cases = [("TP", 1, pick(1, 1, True, 0)),     # Cardiomegaly, confident true positive
         ("TP", 2, pick(2, 1, True, 0)),     # Effusion, confident true positive
         ("FN", 7, pick(7, 1, False, 0)),    # Pneumothorax, missed
         ("FP", 4, pick(4, 0, True, 0))]     # Mass predicted on a radiograph without mass

_, _, test_loader = build_dataloaders(CFG, rank=-1, world_size=1)
ds = test_loader.dataset
items = []
for tag, cls, i in cases:
    s = ds[i]
    items.append(dict(x=s["image"], hr=s["image_hr"], title=f"{tag}: {NAMES[cls]}", cls=cls))
for name, path in [("OOD: natural image", "data/ood_imagenet"), ("OOD: liver CT", "data/ood_ct_liver")]:
    od = OODDataset(path, image_size=224, hr_size=HR)
    s = od[7]
    items.append(dict(x=s["image"], hr=s["image_hr"], title=name, cls=None))


@torch.no_grad()
def analyse(model, x, hr):
    x, hr = x[None].to(DEV), hr[None].to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, hr=hr)
        le = model.local_expert
        feat = le.shallow(x)
        logits = le.attn_crop.attn_conv(feat)[0, 0].float().cpu().numpy()
        patches, idx, sel = le.attn_crop(feat, x, hr_image=hr)
        if le.mil_pool == "legacy_uniform":
            w = torch.softmax(torch.sigmoid(sel.float()), 1)
        else:
            f = le.proj(le.encoder(patches.view(-1, *patches.shape[2:]))).view(1, patches.shape[1], -1)
            _, w = le.mil(f, sel)
    u = {k: float(opinion_uncertainty(out[k]).float()) for k in ("global", "local", "fused")}
    p = opinion_prob(out["fused"]).float().cpu().numpy()[0]
    hgrid = logits.shape[0]
    return dict(attn=logits, idx=idx[0].cpu().numpy(), w=w[0].float().cpu().numpy(), u=u, p=p, h=hgrid,
                frac=le.attn_crop.crop_frac)


res = []
for label, mode, run in MODELS:
    m, _ = build_model_for_mode(mode, CFG, DEV)
    ck = torch.load(f"{run}/best_model.pth", map_location=DEV)
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.get("model_state_dict", ck).items()})
    m.eval()
    res.append([analyse(m, it["x"], it["hr"]) for it in items])
    del m; torch.cuda.empty_cache()

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
fig, axes = plt.subplots(len(MODELS), len(items), figsize=(7.2, 3.15))
for r, (label, _, _) in enumerate(MODELS):
    for c, it in enumerate(items):
        ax = axes[r, c]; d = res[r][c]
        img = (it["x"].numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1)
        ax.imshow(img.mean(2), cmap="gray", extent=(0, 1, 1, 0))
        att = d["attn"]; att = (att - att.min()) / (np.ptp(att) + 1e-8)
        ax.imshow(att, cmap="magma", alpha=0.35, extent=(0, 1, 1, 0), interpolation="bilinear")
        h = d["h"]; half = d["frac"] / 2
        wmax = d["w"].max()
        for k, (ix, wk) in enumerate(zip(d["idx"], d["w"])):
            cy, cx = (ix // h + 0.5) / h, (ix % h + 0.5) / h
            lw = 0.6 + 2.4 * wk / wmax
            ax.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False,
                                   edgecolor="#2a78d6" if wk == wmax else "#eb6834", lw=lw))
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_xticks([]); ax.set_yticks([])
        u = d["u"]
        txt = f"$u_g$={u['global']:.3f}\n$u_\\ell$={u['local']:.4f}\n$u_f$={u['fused']:.4f}"
        if it["cls"] is not None:
            txt += f"\np={d['p'][it['cls']]:.2f}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=5.6, color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=1.2, linewidth=0))
        ax.text(0.98, 0.02, f"max w={wmax:.2f}", transform=ax.transAxes, va="bottom", ha="right",
                fontsize=5.6, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=1.0, linewidth=0))
        if r == 0:
            ax.set_title(it["title"], fontsize=7)
        if c == 0:
            ax.set_ylabel(label, fontsize=7)
fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.3)
fig.savefig("revision/tvc/figures/qualitative.pdf", dpi=300)
print("cases:", [(it["title"]) for it in items])
for r, (label, _, _) in enumerate(MODELS):
    print(label, [(round(d["u"]["global"], 4), round(d["u"]["local"], 5), round(d["u"]["fused"], 5), round(float(d["w"].max()), 3)) for d in res[r]])
