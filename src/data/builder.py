"""Build dataloaders from config."""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from typing import Dict, Tuple, Optional

from .nih_chestxray import NIHChestXrayDataset
from .camelyon16 import Camelyon16Dataset


def _train_transforms(image_size: int, radiograph_safe: bool = False) -> T.Compose:
    """Training augmentation.

    `radiograph_safe` drops the operations that are meaningless or actively
    harmful on grayscale radiographs: RandAugment applies natural-image ops
    (solarize, posterize, equalize) that destroy the intensity semantics a
    radiologist reads, and hue/saturation jitter is a no-op on grayscale that
    still costs signal once replicated to 3 channels.
    """
    ops = [
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
    ]
    if radiograph_safe:
        ops += [T.ColorJitter(brightness=0.2, contrast=0.2)]
    else:
        ops += [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)]
    ops += [T.RandomAffine(degrees=0, translate=(0.05, 0.05))]
    if not radiograph_safe:
        ops += [T.RandAugment(num_ops=2, magnitude=9)]
    ops += [T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.15, scale=(0.02, 0.1))]
    return T.Compose(ops)


def _eval_transforms(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(cfg: Dict, rank: int = -1, world_size: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    When rank >= 0, uses DistributedSampler for DDP training.
    Otherwise uses WeightedRandomSampler for class-imbalance correction.
    """
    is_ddp = rank >= 0 and world_size > 1
    data_cfg = cfg["data"]
    ds_name = data_cfg["dataset"]
    img_size = data_cfg.get("image_size", 224)
    bs = cfg["training"]["batch_size"]
    nw = data_cfg.get("num_workers", 4)
    pin = data_cfg.get("pin_memory", True)

    train_tf = _train_transforms(img_size, data_cfg.get("radiograph_safe_aug", False))
    eval_tf = _eval_transforms(img_size)

    if ds_name == "nih_chestxray":
        nih = data_cfg["nih"]
        train_ds = NIHChestXrayDataset(
            csv_path=nih["csv_path"],
            image_dir=nih["image_dir"],
            file_list=nih["train_list"],
            transform=train_tf,
            image_size=img_size,
            drop_no_finding=nih.get("drop_no_finding", False),
            hr_size=data_cfg.get("hr_size", 0),
            cache_dir=data_cfg.get("cache_dir", ""),
        )
        test_ds = NIHChestXrayDataset(
            csv_path=nih["csv_path"],
            image_dir=nih["image_dir"],
            file_list=nih["test_list"],
            transform=eval_tf,
            image_size=img_size,
            drop_no_finding=nih.get("drop_no_finding", False),
            hr_size=data_cfg.get("hr_size", 0),
            cache_dir=data_cfg.get("cache_dir", ""),
        )
        # Attach the aligned dual-resolution transform when the local expert
        # needs a high-resolution crop source (D8).
        hr_size = data_cfg.get("hr_size", 0)
        if hr_size:
            from .hr_utils import DualResTransform
            train_ds.dual_tf = DualResTransform(img_size, hr_size, train=True,
                radiograph_safe=data_cfg.get("radiograph_safe_aug", False))
            test_ds.dual_tf = DualResTransform(img_size, hr_size, train=False)

        # Split train into train / val AT THE PATIENT LEVEL.
        # The previous image-level random_split allowed different radiographs
        # of the same patient to land in both folds (defect D4); the official
        # NIH train/test lists are patient-disjoint, but the internal
        # validation fold was not.
        val_ratio = cfg["training"].get("val_ratio", 0.1)
        from torch.utils.data import Subset
        pids = np.asarray(train_ds.patient_ids)
        uniq = np.unique(pids)
        rng = np.random.default_rng(cfg.get("seed", 42))
        rng.shuffle(uniq)
        n_val_pat = int(round(len(uniq) * val_ratio))
        val_patients = set(uniq[:n_val_pat].tolist())
        is_val = np.array([pid in val_patients for pid in pids])
        val_idx = np.flatnonzero(is_val).tolist()
        train_idx = np.flatnonzero(~is_val).tolist()
        full_train_ds = train_ds
        # D9: the validation fold was a Subset of the *training* dataset object,
        # so it inherited RandomResizedCrop / RandAugment / RandomErasing.  Every
        # validation pass therefore scored randomly augmented images (the same
        # index fetched twice differed by up to 4.46 in normalised units), which
        # corrupted early stopping, best-checkpoint selection, per-label
        # threshold tuning, temperature scaling and conformal calibration -- all
        # of which are then applied to clean test images.  Validation now uses a
        # separate dataset instance carrying the evaluation transform.
        val_source = NIHChestXrayDataset(
            csv_path=nih["csv_path"],
            image_dir=nih["image_dir"],
            file_list=nih["train_list"],
            transform=eval_tf,
            image_size=img_size,
            drop_no_finding=nih.get("drop_no_finding", False),
            hr_size=data_cfg.get("hr_size", 0),
            cache_dir=data_cfg.get("cache_dir", ""),
        )
        if data_cfg.get("hr_size", 0):
            from .hr_utils import DualResTransform
            val_source.dual_tf = DualResTransform(img_size, data_cfg["hr_size"], train=False)
        train_ds = Subset(full_train_ds, train_idx)
        val_ds = Subset(val_source, val_idx)
        # Record the split for reporting (reviewer M11 asks for exact counts).
        cfg.setdefault("_split_stats", {}).update({
            "n_patients_total": int(len(uniq)),
            "n_patients_train": int(len(uniq) - n_val_pat),
            "n_patients_val": int(n_val_pat),
            "n_images_train": len(train_idx),
            "n_images_val": len(val_idx),
            "n_images_test": len(test_ds),
            "n_patients_test": int(len(np.unique(np.asarray(test_ds.patient_ids)))),
        })

        # Class imbalance is handled in the LOSS (inverse-frequency weights),
        # not by resampling.  The previous WeightedRandomSampler used
        #     w_i = max_c (N / n_c)  over the positive labels of image i,
        # which spans 1.0 to 613.6 on this split: the 141 Hernia images made up
        # 8.4% of every sampled epoch (a 51x oversample) and each was drawn
        # roughly 46 times per epoch.  Combined with the loss weights that is a
        # ~4690x effective emphasis on the rarest class relative to a negative
        # sample, which memorises those 141 images -- train loss collapses while
        # validation AUC peaks at epoch 2 and then degrades (defect D6).
        # Weighting the loss reweights gradients without distorting the data
        # distribution, and it is what the manuscript says the objective is.
        use_weighted_sampler = cfg["training"].get("weighted_sampler", False)
        if use_weighted_sampler:
            labels_tensor = full_train_ds.labels[torch.as_tensor(train_ds.indices)]
            class_pos = labels_tensor.sum(dim=0).clamp(min=1)
            pos_weight = len(labels_tensor) / class_pos
            sample_weights = torch.ones(len(labels_tensor))
            nz = labels_tensor.nonzero()
            for i_, c_ in nz.tolist():
                sample_weights[i_] = max(sample_weights[i_], pos_weight[c_].item())
        if is_ddp:
            # DDP: DistributedSampler handles shard splitting;
            # class weighting via loss or oversample not compatible with DDP split cleanly
            _train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )
        elif use_weighted_sampler:
            _train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
        else:
            _train_sampler = None      # plain shuffling; imbalance handled in the loss

    elif ds_name == "camelyon16":
        cam = data_cfg["camelyon16"]
        train_full = Camelyon16Dataset(
            data_dir=cam["data_dir"], split="train",
            transform=train_tf, image_size=img_size,
        )
        test_ds = Camelyon16Dataset(
            data_dir=cam["data_dir"], split="test",
            transform=eval_tf, image_size=img_size,
        )
        val_ratio = cfg["training"].get("val_ratio", 0.1)
        val_len = int(len(train_full) * val_ratio)
        train_len = len(train_full) - val_len
        g = torch.Generator().manual_seed(cfg.get("seed", 42))
        train_ds, val_ds = random_split(train_full, [train_len, val_len], generator=g)
        _train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    persistent = nw > 0
    prefetch = 4 if nw > 0 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    train_loader = DataLoader(train_ds, batch_size=bs,
                              sampler=_train_sampler,
                              shuffle=(_train_sampler is None),
                              num_workers=nw, pin_memory=pin, drop_last=True,
                              persistent_workers=persistent, prefetch_factor=prefetch)
    val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_sampler, shuffle=False,
                            num_workers=nw, pin_memory=pin,
                            persistent_workers=persistent, prefetch_factor=prefetch)
    test_loader = DataLoader(test_ds, batch_size=bs, sampler=test_sampler, shuffle=False,
                             num_workers=nw, pin_memory=pin,
                             persistent_workers=persistent, prefetch_factor=prefetch)
    return train_loader, val_loader, test_loader
