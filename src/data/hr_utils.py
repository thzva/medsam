"""Geometrically aligned dual-resolution transform.

The local expert crops regions chosen by attention computed on the 224x224
global view.  Those normalised coordinates are only meaningful if the
high-resolution tensor the crops are sampled from is the SAME view.

Applying an independent `Resize` to build the HR tensor breaks that: the
training pipeline's RandomResizedCrop selects a random sub-region, so the two
tensors framed different parts of the radiograph and the attention coordinates
pointed somewhere else entirely in the HR image.

Both tensors are therefore derived from a single augmented image: augment once
at high resolution, then downsample to get the global view.  Geometry and
photometry are identical by construction.
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_NORM = T.Normalize(mean=_MEAN, std=_STD)


class DualResTransform:
    """Returns (global_224, hr) derived from one augmented high-res image."""

    def __init__(self, image_size: int = 224, hr_size: int = 672, train: bool = False,
                 radiograph_safe: bool = False):
        self.image_size = image_size
        self.hr_size = hr_size
        self.train = train
        if train:
            jit = (T.ColorJitter(brightness=0.2, contrast=0.2) if radiograph_safe
                   else T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05))
            self.aug = T.Compose([
                T.RandomResizedCrop(hr_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                jit,
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
        else:
            self.aug = T.Resize((hr_size, hr_size))

    def __call__(self, img):
        hi = self.aug(img)                                  # PIL, hr_size
        hi_t = _NORM(TF.to_tensor(hi))
        lo_t = TF.resize(hi_t, [self.image_size, self.image_size], antialias=True)
        if self.train:
            lo_t = T.RandomErasing(p=0.15, scale=(0.02, 0.1))(lo_t)
        return lo_t, hi_t


def hr_transform(hr_size: int = 672):
    """Eval-only HR transform, for datasets that carry no augmentation."""
    return T.Compose([
        T.Resize((hr_size, hr_size)),
        T.ToTensor(),
        _NORM,
    ])
