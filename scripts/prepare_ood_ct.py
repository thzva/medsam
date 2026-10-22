"""Extract axial CT slices from MSD Liver volumes as PNG OOD samples.

For each volume, sample N evenly-spaced middle slices (avoiding empty top/bottom).
Apply soft-tissue HU windowing then normalize to 0-255.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm


def hu_window(arr: np.ndarray, level: int = 50, width: int = 400) -> np.ndarray:
    """Soft-tissue windowing: clip then map to [0, 255]."""
    lo, hi = level - width / 2, level + width / 2
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/Camelyon16/Task03_Liver/imagesTr",
                    help="MSD Liver imagesTr directory")
    ap.add_argument("--dst", default="data/ood_ct_liver", help="Output PNG folder")
    ap.add_argument("--per_volume", type=int, default=4,
                    help="Slices to extract per volume")
    ap.add_argument("--target_count", type=int, default=500,
                    help="Total OOD samples to generate")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    vols = sorted(src.glob("*.nii.gz"))
    if not vols:
        print(f"No NIfTI files in {src}", file=sys.stderr)
        sys.exit(1)

    saved = 0
    pbar = tqdm(vols, desc="Volumes")
    for vol_path in pbar:
        if saved >= args.target_count:
            break
        try:
            img = nib.load(str(vol_path))
            data = img.get_fdata()  # (H, W, D) usually
        except Exception as e:
            pbar.write(f"Skip {vol_path.name}: {e}")
            continue
        if data.ndim != 3:
            continue
        D = data.shape[-1]
        # Sample slices from middle 60%
        lo, hi = int(D * 0.2), int(D * 0.8)
        idxs = np.linspace(lo, hi, args.per_volume, dtype=int)
        for i, k in enumerate(idxs):
            if saved >= args.target_count:
                break
            sl = data[..., k]
            sl = hu_window(sl)
            # Most NIfTI scans store axial as (cols, rows); rotate so head-up
            sl = np.rot90(sl, k=1)
            Image.fromarray(sl).convert("L").save(
                dst / f"{vol_path.stem.replace('.nii','')}_z{k:03d}.png"
            )
            saved += 1
        pbar.set_postfix(saved=saved)

    print(f"Saved {saved} CT slices to {dst}")


if __name__ == "__main__":
    main()
