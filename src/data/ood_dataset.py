"""Out-of-Distribution dataset for OOD detection evaluation."""

import os
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class OODDataset(Dataset):
    """Generic image-folder dataset used as OOD test set.

    Any folder of images (e.g. a small ImageNet subset) can be pointed at.
    All samples are treated as OOD (label = -1).
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        max_samples: int = 5000,
        hr_size: int = 0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.hr_tf = None
        if hr_size:
            from .hr_utils import hr_transform
            self.hr_tf = hr_transform(hr_size)

        self.transform = transform
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        self.paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPEG"):
            self.paths.extend(sorted(self.data_dir.rglob(ext)))
        self.paths = self.paths[:max_samples]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        raw = Image.open(path).convert("RGB")
        out = {"image": self.transform(raw),
               "label": torch.tensor(-1, dtype=torch.long),
               "filename": path.name}
        if self.hr_tf is not None:
            out["image_hr"] = self.hr_tf(raw)
        return out
