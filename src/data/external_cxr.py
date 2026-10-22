"""External chest-radiograph datasets for the revision.

M10 / R1-8 asked for external validation; R1-7 asked for a NEAR-OOD regime
(the submitted paper only tested liver CT and natural images, both "far").

CheXpert (Irvin et al., 2019) is acquired at a different institution with
different scanners and a different labelling pipeline, so relative to a model
trained on NIH ChestX-ray14 it serves as
  (a) an external validation set on the 8 shared findings, and
  (b) a NEAR-OOD set (same modality, same anatomy, different distribution).

Label encoding in the parquet mirror is offset by 2:
    2 -> negative/absent, 3 -> positive/present.
"""

import io
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from .nih_chestxray import LABEL_NAMES, LABEL2IDX

# CheXpert column -> NIH label name.  Only findings with the same clinical
# definition are mapped; the rest are ignored for the shared-label evaluation.
CHEXPERT_TO_NIH = {
    "No Finding": "No Finding",
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",
}
SHARED_NIH_IDX = [LABEL2IDX[v] for v in CHEXPERT_TO_NIH.values()]
SHARED_NAMES = list(CHEXPERT_TO_NIH.values())


class CheXpertParquet(Dataset):
    """CheXpert frontal radiographs from the HuggingFace parquet mirror."""

    def __init__(self, parquet_paths, image_size: int = 224,
                 frontal_only: bool = True, transform=None,
                 uncertain_as: int = 0, max_items: int = None, hr_size: int = 0):
        if isinstance(parquet_paths, (str, Path)):
            parquet_paths = [parquet_paths]
        # Read shards lazily and stop as soon as max_items is satisfied.  The
        # previous version concatenated every shard first and sliced afterwards,
        # so asking for 8000 rows pulled ~1 GB of embedded image bytes into a
        # DataFrame.  Three concurrent evaluations doing that alongside five
        # training jobs thrashed the page cache badly enough to stall an
        # evaluation for three hours.
        parts, kept = [], 0
        for path in parquet_paths:
            part = pd.read_parquet(path)
            if frontal_only and "Frontal/Lateral" in part.columns:
                part = part[part["Frontal/Lateral"] == 0]
            if max_items:
                part = part.iloc[: max_items - kept]
            parts.append(part)
            kept += len(part)
            if max_items and kept >= max_items:
                break
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0].reset_index(drop=True)
        self.df = df
        self.image_size = image_size
        self.uncertain_as = uncertain_as

        # Build an (N, 15) NIH-shaped multi-hot matrix; unmapped columns stay 0.
        y = torch.zeros(len(df), len(LABEL_NAMES), dtype=torch.float32)
        for cx_col, nih_name in CHEXPERT_TO_NIH.items():
            if cx_col not in df.columns:
                continue
            raw = df[cx_col].to_numpy()
            pos = (raw == 3).astype(np.float32)          # 3 -> present
            unc = (raw == 1).astype(np.float32)          # 1 -> uncertain (-1)
            y[:, LABEL2IDX[nih_name]] = torch.from_numpy(pos + unc * float(uncertain_as))
        self.labels = y

        self.hr_tf = None
        if hr_size:
            from .hr_utils import hr_transform
            self.hr_tf = hr_transform(hr_size)
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        cell = self.df["image"].iloc[i]
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        out = {"image": self.transform(img), "label": self.labels[i],
               "filename": str(self.df["Path"].iloc[i]) if "Path" in self.df else str(i)}
        if self.hr_tf is not None:
            out["image_hr"] = self.hr_tf(img)
        return out
