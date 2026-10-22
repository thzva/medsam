"""Data preparation script – extract NIH Chest X-ray zip files."""

import os
import sys
import argparse
import zipfile
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_nih_images(data_root: str):
    """Extract all images_*.zip files into a flat image directory."""
    zip_dir = Path(data_root) / "NIH_Chest_Xray_Dataset" / "data" / "images"
    out_dir = zip_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(zip_dir.glob("images_*.zip"))
    if not zips:
        logger.warning(f"No zip files found in {zip_dir}")
        return

    logger.info(f"Found {len(zips)} zip archives in {zip_dir}")

    for zp in zips:
        logger.info(f"Extracting {zp.name} ...")
        with zipfile.ZipFile(zp, "r") as zf:
            # Extract only .png files, flatten into out_dir
            for member in zf.namelist():
                if member.lower().endswith(".png"):
                    fname = os.path.basename(member)
                    target = out_dir / fname
                    if target.exists():
                        continue
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        logger.info(f"  Done: {zp.name}")

    n_images = len(list(out_dir.glob("*.png")))
    logger.info(f"Total extracted images: {n_images}")


def verify_nih(data_root: str):
    """Quick sanity check on NIH dataset."""
    base = Path(data_root) / "NIH_Chest_Xray_Dataset" / "data"
    csv = base / "Data_Entry_2017_v2020.csv"
    train_list = base / "train_val_list.txt"
    test_list = base / "test_list.txt"
    img_dir = base / "images" / "images"

    checks = {
        "CSV": csv.exists(),
        "train_list": train_list.exists(),
        "test_list": test_list.exists(),
        "image_dir": img_dir.exists(),
    }
    if img_dir.exists():
        checks["num_images"] = len(list(img_dir.glob("*.png")))

    logger.info("NIH Chest X-ray verification:")
    for k, v in checks.items():
        logger.info(f"  {k}: {v}")

    return all(v for k, v in checks.items() if k != "num_images")


def verify_camelyon16(data_root: str):
    """Quick sanity check on Camelyon16 dataset."""
    cam_dir = Path(data_root) / "Camelyon16"
    logger.info("Camelyon16 verification:")
    logger.info(f"  Directory exists: {cam_dir.exists()}")
    if cam_dir.exists():
        for split in ["train", "test"]:
            split_dir = cam_dir / split
            if split_dir.exists():
                for cls in ["normal", "tumor"]:
                    cls_dir = split_dir / cls
                    if cls_dir.exists():
                        n = len(list(cls_dir.glob("*.png"))) + len(list(cls_dir.glob("*.jpg")))
                        logger.info(f"  {split}/{cls}: {n} images")
                    else:
                        logger.info(f"  {split}/{cls}: not found")
            else:
                logger.info(f"  {split}/: not found")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--extract", action="store_true",
                        help="Extract NIH zip files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset integrity")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.extract:
        extract_nih_images(args.data_root)

    if args.verify or not args.extract:
        verify_nih(args.data_root)
        verify_camelyon16(args.data_root)


if __name__ == "__main__":
    main()
