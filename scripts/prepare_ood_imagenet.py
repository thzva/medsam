"""Download a small natural-image OOD set: 500 images from Imagenette validation.

Imagenette is a 10-class subset of ImageNet (fastai), well-suited as natural-image OOD.
"""
import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data/ood_imagenet", help="Output PNG/JPG folder")
    ap.add_argument("--target_count", type=int, default=500)
    ap.add_argument("--cache", default="data/.cache_imagenette.tgz")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        print(f"Downloading {URL} ...")
        try:
            urllib.request.urlretrieve(URL, cache)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Extract validation images only
    print(f"Extracting from {cache} ...")
    saved = 0
    with tarfile.open(cache, "r:gz") as tar:
        members = [m for m in tar.getmembers() if "/val/" in m.name and m.name.lower().endswith(".jpeg")]
        members = members[:args.target_count]
        for m in members:
            f = tar.extractfile(m)
            if f is None:
                continue
            out = dst / Path(m.name).name
            with open(out, "wb") as fp:
                shutil.copyfileobj(f, fp)
            saved += 1

    print(f"Saved {saved} natural images to {dst}")


if __name__ == "__main__":
    main()
