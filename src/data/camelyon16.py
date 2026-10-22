"""Camelyon16 patch-level binary classification dataset."""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

NUM_CLASSES = 2


class Camelyon16Dataset(Dataset):
    """Camelyon16 dataset – works on pre-extracted patches.

    Expected directory layout
    -------------------------
    data_dir/
      train/
        normal/   *.png
        tumor/    *.png
      test/
        normal/   *.png
        tumor/    *.png
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        self.transform = transform
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        self.samples: List[Tuple[str, int]] = []
        self._scan_directory()

    def _scan_directory(self):
        """Scan split directory for normal / tumor sub-folders."""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            # Fallback: treat entire data_dir as flat with label in filename
            self._scan_flat()
            return

        for label, cls_name in enumerate(["normal", "tumor"]):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.png")):
                self.samples.append((str(p), label))
            for p in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((str(p), label))

    def _scan_flat(self):
        """Fallback scanner: infer label from filename containing 'tumor'."""
        for ext in ("*.png", "*.jpg"):
            for p in sorted(self.data_dir.glob(f"**/{ext}")):
                label = 1 if "tumor" in p.stem.lower() else 0
                self.samples.append((str(p), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"image": img, "label": label_tensor, "filename": os.path.basename(path)}
