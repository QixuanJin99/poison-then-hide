# datasets_cifar10.py
import os
from typing import Tuple, Optional, Dict, Callable

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms as T

from utils import set_seed, _worker_init_fn

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Standard CIFAR-10 stats (float in [0,1])
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD:  Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

# CLIP (OpenAI) normalization
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

__all__ = [
    "CIFAR10_MEAN", "CIFAR10_STD",
    "CLIP_MEAN", "CLIP_STD",
    "_tf_cifar10",
    "build_cifar10_train_loader", "build_cifar10_val_loader",
    "build_cifar10_loaders",
    "denormalize_cifar10", "denormalize_clip",
    "class_mappings",
]

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def _tf_cifar10(
    train: bool,
    *,
    clip_normalization: bool = False,
    image_size: int = 32,
    resize_shorter_to: Optional[int] = None,           # if not None, resize shorter side first (useful if image_size > 32)
    rand_resize_scale: Tuple[float, float] = (0.8, 1.0),
) -> T.Compose:
    """
    CIFAR-10 transforms with optional CLIP normalization and resizing.

    Notes:
    - CIFAR-10 images are 32x32; for CLIP backbones, set image_size=224 (and optionally resize_shorter_to).
    - For val: (optional) Resize(shorter_side=...) -> CenterCrop(image_size)
    - For train: (optional) Resize -> RandomResizedCrop(image_size, scale=rand_resize_scale) -> Flip
    """
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (CIFAR10_MEAN, CIFAR10_STD)

    aug = []
    if not train:
        if resize_shorter_to is not None:
            aug += [T.Resize(resize_shorter_to)]
        if image_size != 32 or resize_shorter_to is not None:
            aug += [T.CenterCrop(image_size)]
    else:
        if resize_shorter_to is not None:
            aug += [T.Resize(resize_shorter_to)]
        if image_size != 32 or resize_shorter_to is not None:
            aug += [T.RandomResizedCrop(image_size, scale=rand_resize_scale)]
        else:
            # keep native 32x32 but still do light aug
            aug += [T.RandomCrop(32, padding=4)]
        aug += [T.RandomHorizontalFlip()]

    return T.Compose(aug + [T.ToTensor(), T.Normalize(mean, std)])


def _resolve_transform(
    *,
    train: bool,
    transform_override: Optional[Callable],
    clip_normalization: bool,
    image_size: int,
    resize_shorter_to: Optional[int],
    rand_resize_scale: Tuple[float, float],
) -> T.Compose:
    """Use custom transform if provided; otherwise fall back to _tf_cifar10(...)."""
    if transform_override is not None:
        return transform_override
    return _tf_cifar10(
        train=train,
        clip_normalization=clip_normalization,
        image_size=image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
    )

# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_cifar10_val_loader(
    root_cifar10: str,
    bs: int = 256,
    nw: int = 4,
    *,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    download: bool = False,
    # used only if transform_val is None
    clip_normalization: bool = False,
    clip_image_size: int = 32,
    resize_shorter_to: Optional[int] = None,
    # NEW:
    transform_val: Optional[Callable] = None,
) -> DataLoader:
    """Validation loader for CIFAR-10 (test split)."""
    tf_val = _resolve_transform(
        train=False,
        transform_override=transform_val,
        clip_normalization=clip_normalization,
        image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=(0.8, 1.0),  # unused for val
    )

    val_ds = datasets.CIFAR10(root=root_cifar10, train=False, transform=tf_val, download=download)
    sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    if persistent_workers is None:
        persistent_workers = (nw > 0)

    return DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=_worker_init_fn if nw > 0 else None,
    )


def build_cifar10_train_loader(
    root_cifar10: str,
    bs: int = 256,
    nw: int = 4,
    *,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: Optional[int] = None,
    download: bool = False,
    # used only if transform_train is None
    clip_normalization: bool = False,
    clip_image_size: int = 32,
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.8, 1.0),
    # NEW:
    transform_train: Optional[Callable] = None,
) -> DataLoader:
    """Training loader for CIFAR-10 (train split)."""
    if seed is not None:
        set_seed(seed)

    tf_train = _resolve_transform(
        train=True,
        transform_override=transform_train,
        clip_normalization=clip_normalization,
        image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
    )

    train_ds = datasets.CIFAR10(root=root_cifar10, train=True, transform=tf_train, download=download)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None

    if persistent_workers is None:
        persistent_workers = (nw > 0)

    return DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=_worker_init_fn if nw > 0 else None,
    )


def build_cifar10_loaders(
    root_cifar10: str,
    *,
    bs_train: int = 256,
    bs_val: int = 256,
    nw: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: Optional[int] = None,
    download: bool = False,
    # used if custom transforms are None
    clip_normalization: bool = False,
    clip_image_size: int = 32,
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.8, 1.0),
    # NEW:
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
):
    """Returns (train_loader, val_loader) with optional custom transforms."""
    tr = build_cifar10_train_loader(
        root_cifar10, bs=bs_train, nw=nw,
        distributed=distributed, rank=rank, world_size=world_size, seed=seed, download=download,
        clip_normalization=clip_normalization, clip_image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to, rand_resize_scale=rand_resize_scale,
        transform_train=transform_train,
    )
    va = build_cifar10_val_loader(
        root_cifar10, bs=bs_val, nw=nw,
        distributed=distributed, rank=rank, world_size=world_size, download=download,
        clip_normalization=clip_normalization, clip_image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to,
        transform_val=transform_val,
    )
    return tr, va

# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------
def denormalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 normalization. Expects x in (N or 1, 3, H, W)."""
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean

def denormalize_clip(x: torch.Tensor) -> torch.Tensor:
    """Undo CLIP normalization."""
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=x.device).view(1, 3, 1, 1)
    return x * std + mean

def class_mappings(ds: datasets.CIFAR10) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Return (idx->class_name, class_name->idx) mappings.
    torchvision's CIFAR10 exposes `classes` (list of class names).
    """
    idx2cls = {i: name for i, name in enumerate(ds.classes)}
    cls2idx = {name: i for i, name in idx2cls.items()}
    return idx2cls, cls2idx
