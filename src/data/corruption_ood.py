"""Severity-graded near-OOD: corrupted NIH chest radiographs.

R1 comment 7: both OOD tests in the submitted paper live in the "far" regime
(liver CT, natural images), where almost any uncertainty score succeeds.  A
graded corruption sweep gives a continuum from in-distribution (severity 0)
to progressively-further OOD, so the uncertainty score can be shown to
degrade monotonically rather than only to solve an easy problem.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as T

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Severity grids follow the ImageNet-C convention (5 increasing levels).
_SEVERITY = {
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],
    "blur":           [0.7,  1.2,  1.8,  2.6,  3.6],
    "contrast":       [0.75, 0.60, 0.45, 0.32, 0.22],
    "brightness":     [0.12, 0.22, 0.32, 0.44, 0.58],
}
CORRUPTIONS = list(_SEVERITY.keys())


def _apply(img: Image.Image, kind: str, sev: int, rng) -> Image.Image:
    v = _SEVERITY[kind][sev - 1]
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=v))
    a = np.asarray(img).astype(np.float32) / 255.0
    if kind == "gaussian_noise":
        a = a + rng.normal(0, v, a.shape)
    elif kind == "contrast":
        a = (a - a.mean()) * v + a.mean()
    elif kind == "brightness":
        a = a + v
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


class CorruptedNIH(Dataset):
    """Wraps an NIH dataset and applies one corruption at one severity."""

    def __init__(self, base_dataset, kind: str, severity: int,
                 image_size: int = 224, seed: int = 0, hr_size: int = 0):
        # severity 0 = identity.  It is a control, not a corruption: the OOD
        # AUROC against the clean test split must come out at ~0.5, which
        # proves the corrupted and clean pipelines are otherwise identical.
        assert kind in _SEVERITY and 0 <= severity <= 5
        self.base = base_dataset
        self.kind, self.severity = kind, severity
        self.image_size = image_size
        self.seed = seed
        # MUST match _eval_transforms exactly.  Using PIL's Image.resize here
        # instead applies a different resampling kernel (bicubic) than the
        # clean path (bilinear); the resulting high-frequency difference is
        # read by the model as a distribution shift and swamps the corruption
        # signal, which is what produced OOD AUROCs of exactly 0.000 and
        # severity-1 values scattered between 0.04 and 0.95 across models.
        self.resize = T.Resize((image_size, image_size))
        self.hr_size = hr_size
        self.hr_resize = T.Resize((hr_size, hr_size)) if hr_size else None

    def __len__(self):
        return len(self.base)

    def _raw_image(self, i):
        ds, idx = self.base, i
        from torch.utils.data import Subset
        while isinstance(ds, Subset):
            idx = ds.indices[idx]; ds = ds.dataset
        return Image.open(ds.image_dir / ds.image_names[idx]).convert("RGB"), ds.labels[idx]

    def __getitem__(self, i):
        img, label = self._raw_image(i)
        lo = self.resize(img)                       # identical to the clean path
        if self.severity > 0:
            lo = _apply(lo, self.kind, self.severity, np.random.default_rng(self.seed + i))
        out = {"image": _NORM(T.functional.to_tensor(lo)), "label": label,
               "filename": f"{self.kind}_s{self.severity}_{i}"}
        if self.hr_resize is not None:
            hi = self.hr_resize(img)
            if self.severity > 0:
                hi = _apply(hi, self.kind, self.severity, np.random.default_rng(self.seed + i))
            out["image_hr"] = _NORM(T.functional.to_tensor(hi))
        return out
