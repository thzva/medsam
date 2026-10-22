"""NIH Chest X-ray 14 dataset loader."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, List

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

LABEL_NAMES = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
LABEL2IDX = {name: i for i, name in enumerate(LABEL_NAMES)}
NUM_CLASSES = len(LABEL_NAMES)


class NIHChestXrayDataset(Dataset):
    """NIH Chest X-ray 14 multi-label classification dataset."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        file_list: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        drop_no_finding: bool = False,
        hr_size: int = 0,
        cache_dir: str = "",
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        # Optional pre-decoded cache (see scripts/build_cache.py).  Sized so
        # that no augmented sample is ever upsampled, so the cache changes
        # decode cost only, not what the model sees.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.hr_size = hr_size
        self.dual_tf = None      # set by the builder when hr_size > 0
        self.transform = transform
        self.image_size = image_size

        # Load metadata CSV
        df = pd.read_csv(csv_path)

        # Filter by split list if provided
        if file_list is not None:
            with open(file_list, "r") as f:
                keep = set(line.strip() for line in f if line.strip())
            df = df[df["Image Index"].isin(keep)].reset_index(drop=True)

        self.image_names: List[str] = df["Image Index"].tolist()
        self.labels_raw: List[str] = df["Finding Labels"].tolist()

        # Patient identity -- required for a leakage-free patient-level split.
        if "Patient ID" in df.columns:
            self.patient_ids = df["Patient ID"].astype(str).tolist()
        else:
            self.patient_ids = [n.split("_")[0] for n in self.image_names]

        # Pre-encode multi-hot labels
        self.labels = self._encode_labels(self.labels_raw)

        # "No Finding" is logically the conjunction of the 14 findings being
        # absent (reviewer M1).  Rather than predict it as a competing 15th
        # label, we drop it from the target and derive it at evaluation time
        # as P(no finding) = prod_c (1 - p_c).
        self.drop_no_finding = drop_no_finding
        if drop_no_finding:
            self.labels = self.labels[:, 1:].contiguous()
            self.label_names = LABEL_NAMES[1:]
        else:
            self.label_names = list(LABEL_NAMES)

        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    # ------------------------------------------------------------------ #
    def _encode_labels(self, raw: List[str]) -> torch.Tensor:
        """Convert '|'-separated label strings to multi-hot vectors."""
        encoded = torch.zeros(len(raw), NUM_CLASSES, dtype=torch.float32)
        for i, label_str in enumerate(raw):
            for name in label_str.split("|"):
                name = name.strip()
                if name in LABEL2IDX:
                    encoded[i, LABEL2IDX[name]] = 1.0
        return encoded

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        fname = self.image_names[idx]
        img_path = self.image_dir / fname
        if self.cache_dir is not None:
            cached = self.cache_dir / (fname.rsplit(".", 1)[0] + ".jpg")
            if cached.exists():
                img_path = cached

        raw = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.dual_tf is not None:
            lo, hi = self.dual_tf(raw)
            return {"image": lo, "image_hr": hi, "label": label, "filename": fname}
        return {"image": self.transform(raw), "label": label, "filename": fname}
