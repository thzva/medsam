"""Dump per-sample far-OOD arrays (probs, u_global, u_local, u_fused, conflict)
for an already-evaluated run, without redoing the full evaluation.
Writes <run_dir>/ood_arrays.npz with keys ood_<set>_<field>."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training.utils import load_config
from src.data.ood_dataset import OODDataset
from eval_revision import collect
from train_ddp import build_model_for_mode

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--config", required=True)
args = ap.parse_args()
run_dir = Path(args.run_dir)
if (run_dir / "ood_arrays.npz").exists():
    print("[skip]", run_dir); sys.exit(0)
device = torch.device("cuda:0")
cfg = load_config(args.config)
model, _ = build_model_for_mode(args.mode, cfg, device)
ck = torch.load(run_dir / "best_model.pth", map_location=device)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.get("model_state_dict", ck).items()})
HR = cfg["data"].get("hr_size", 0)
out = {}
for name, path in [("ct_liver", "data/ood_ct_liver"), ("imagenet", "data/ood_imagenet")]:
    dl = DataLoader(OODDataset(path, image_size=cfg["data"]["image_size"], hr_size=HR),
                    batch_size=40, num_workers=8)
    A = collect(model, dl, device)
    out.update({f"ood_{name}_{k}": v for k, v in A.items() if v is not None})
np.savez_compressed(run_dir / "ood_arrays.npz", **out)
print("wrote", run_dir / "ood_arrays.npz", sorted(out))
