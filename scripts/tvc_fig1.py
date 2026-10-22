"""Architecture figure (Fig. 1) for the TVC manuscript, drawn from a real test
radiograph and the crops actually selected by the gated-MIL + NMS model.
Writes revision/tvc/figures/architecture.pdf. Run from the repo root."""
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from src.training.utils import load_config
from src.data.builder import build_dataloaders
from train_ddp import build_model_for_mode

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#222222"
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
DEV = torch.device("cuda:0")
CFG = load_config("config/revision_tvc.yaml")

_, _, tl = build_dataloaders(CFG, rank=-1, world_size=1)
s = tl.dataset[123]
m, _ = build_model_for_mode("tms_edl", CFG, DEV)
ck = torch.load("outputs_mil2/tms_edl/seed42/best_model.pth", map_location=DEV)
m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.get("model_state_dict", ck).items()}); m.eval()
le = m.local_expert
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    x, hr = s["image"][None].to(DEV), s["image_hr"][None].to(DEV)
    f = le.shallow(x)
    att = le.attn_crop.attn_conv(f)[0, 0].float().cpu().numpy()
    patches, idx, sel = le.attn_crop(f, x, hr_image=hr)
    feats = le.proj(le.encoder(patches.view(-1, *patches.shape[2:]))).view(1, patches.shape[1], -1)
    _, w = le.mil(feats, sel)
img = (s["image"].numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1).mean(2)
crops = [((p.float().cpu().numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1)).mean(2) for p in patches[0]]
w = w[0].float().cpu().numpy(); idx = idx[0].cpu().numpy(); h = att.shape[0]

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 3.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 72); ax.set_ylim(0, 33); ax.axis("off")


def box(x, y, wd, ht, text, fc, ec=INK, fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0.25,rounding_size=0.6",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + wd / 2, y + ht / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(x0, y0, x1, y1, col=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=col, linestyle=ls, shrinkA=0, shrinkB=0))


def image(arr, x, y, size, cmap="gray", alpha=1.0):
    ax.imshow(arr, cmap=cmap, extent=(x, x + size, y, y + size), alpha=alpha, zorder=2)


# input radiograph
image(img, 0.8, 11.5, 9)
ax.text(5.3, 21.3, "input radiograph", ha="center", fontsize=7)
ax.text(5.3, 10.6, "$224^2$ global\n$560^2$ local", ha="center", va="top", fontsize=5.8, color="#444")

# ---------------- global stream (top)
arrow(10.1, 18.5, 12.5, 26.5)
box(12.8, 24.5, 9.5, 4, "ViT-B/16\n(ImageNet-21k)", "#e7f0fb")
arrow(22.6, 26.5, 25, 26.5)
box(25.3, 24.5, 9.5, 4, "Beta head\nper finding $c$", "#e7f0fb")
arrow(35.1, 26.5, 37.5, 26.5)
box(37.8, 24.5, 9.5, 4, "$(b^+_{g,c}, b^-_{g,c}, u_{g,c})$", "#ffffff", fs=6.5)

# ---------------- local stream (bottom)
arrow(10.1, 13.0, 12.5, 9.0)
image(img, 12.8, 3.0, 8)
attn = (att - att.min()) / (np.ptp(att) + 1e-8)
image(attn, 12.8, 3.0, 8, cmap="magma", alpha=0.45)
half = le.attn_crop.crop_frac / 2
clip = Rectangle((12.8, 3.0), 8, 8, transform=ax.transData, fc="none", ec="none")
ax.add_patch(clip)
for k, (ix, wk) in enumerate(zip(idx, w)):
    cy, cx = (ix // h + 0.5) / h, (ix % h + 0.5) / h
    r = Rectangle((12.8 + 8 * (cx - half), 3.0 + 8 * (1 - cy - half)), 8 * 2 * half, 8 * 2 * half,
                  fill=False, ec=BLUE if wk == w.max() else ORANGE, lw=0.9, zorder=3)
    ax.add_patch(r); r.set_clip_path(clip)
ax.text(16.8, 11.6, "attention map + top-$K$ crops (NMS), $K{=}6$", ha="center", va="bottom", fontsize=5.8)
arrow(21.1, 7, 23.4, 7)
for k, c in enumerate(crops):
    xo, yo = 23.7 + (k % 3) * 2.9, 8.1 - (k // 3) * 2.9
    image(c, xo, yo, 2.6)
    ax.add_patch(Rectangle((xo, yo), 2.6, 2.6, fill=False, ec=BLUE if w[k] == w.max() else ORANGE,
                           lw=0.4 + 1.6 * w[k] / w.max(), zorder=3))
ax.text(28.0, 4.1, "$224^2$ crops from $560^2$", ha="center", va="top", fontsize=5.8)
arrow(32.6, 7, 34.4, 7)
box(34.7, 5.0, 7.3, 4, "ResNet-50\n+ gated MIL", "#fdeee6")
arrow(42.3, 7, 44.0, 7)
box(44.3, 5.0, 6.5, 4, "Beta head", "#fdeee6")
arrow(51.1, 7, 52.6, 7)
box(52.9, 5.0, 9.0, 4, "$(b^+_{\\ell,c}, b^-_{\\ell,c}, u_{\\ell,c})$", "#ffffff", fs=6.5)

# ---------------- fusion
box(52.3, 14.5, 10.5, 5.2, "per-label\nDempster's rule\n$u_f \\leq \\min(u_g,u_\\ell)$", "#e6f6ef", bold=False)
arrow(47.6, 26.0, 55.5, 20.0)
arrow(57.4, 9.3, 57.4, 14.2)
arrow(63.1, 17.1, 65.0, 17.1)
box(65.3, 15.0, 6.2, 4.2, "$p_{f,c}$\n(prediction)", "#ffffff", fs=6.5)
ax.text(68.4, 13.8, "fused $u_f$:\nvacuity collapse", ha="center", va="top", fontsize=5.8, color=ORANGE)
# decoupled read-out
arrow(47.6, 27.5, 65.3, 27.5, col=BLUE)
box(65.3, 25.6, 6.2, 3.8, "OOD score\n$\\bar u_g$", "#e7f0fb", fs=6.5)
ax.text(56.5, 28.3, "decoupled read-out (no retraining)", ha="center", fontsize=5.8, color=BLUE)
ax.text(1, 31.6, "global stream", fontsize=7, color=BLUE, fontweight="bold")
ax.text(1, 1.2, "local stream", fontsize=7, color=ORANGE, fontweight="bold")
fig.savefig("revision/tvc/figures/architecture.pdf", dpi=300)
print("ok")
