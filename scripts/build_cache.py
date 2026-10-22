"""Pre-decode the NIH radiographs to a cached resolution.

The D8 fix requires a high-resolution tensor, so every __getitem__ was
decoding a 1024x1024 PNG and then augmenting at 560x560 -- both CPU-bound,
which starved the GPUs and capped useful concurrency at three runs.

Caching removes the decode cost only.  It must not change what the model
sees, so the cache resolution is chosen to be lossless with respect to the
augmentation: RandomResizedCrop(560, scale=(0.7, 1.0)) takes a crop covering
at least 70% of the area, i.e. >= 0.837 of the linear extent, so producing a
560px output needs a source of at least 560 / 0.837 = 669px.  768 is the next
comfortable size above that, so no augmented sample is ever upsampled.
"""
import argparse, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

SRC = Path("data/NIH_Chest_Xray_Dataset/data/images/images")
DST = Path("data/nih_cache_768")
SIZE = 768
QUALITY = 95


def convert(name):
    out = DST / (name.rsplit(".", 1)[0] + ".jpg")
    if out.exists():
        return 0
    try:
        im = Image.open(SRC / name).convert("RGB")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BILINEAR)
        im.save(out, "JPEG", quality=QUALITY, optimize=False)
        return 1
    except Exception as e:
        print(f"FAIL {name}: {e}", file=sys.stderr)
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in SRC.glob("*.png"))
    print(f"{len(names)} images -> {DST} at {SIZE}px")
    done = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert, n) for n in names]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            done += r == 1; skip += r == 0; fail += r == -1
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(names)}  converted={done} skipped={skip} failed={fail}", flush=True)
    print(f"DONE converted={done} skipped={skip} failed={fail}")


if __name__ == "__main__":
    main()
