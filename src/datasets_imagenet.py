# datasets_imagenet.py
import os
from typing import Tuple, Optional, Dict, Iterable, Callable

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms as T
from utils import set_seed, _worker_init_fn

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# CLIP (OpenAI) normalization
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


# Export surface
__all__ = [
    "IMAGENET_MEAN", "IMAGENET_STD",
    "CLIP_MEAN", "CLIP_STD",
    "_tf_imagenet",
    "build_imagenet_train_loader", "build_imagenet_val_loader",
    "build_imagenet_loaders",
    "denormalize_imagenet", "denormalize_clip",
    "class_mappings",
]

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

def _tf_imagenet(
    train: bool,
    *,
    clip_normalization: bool = False,
    image_size: int = 224,
    resize_shorter_to: Optional[int] = None,   # if None -> 256 for val, 256 for train by default
    rand_resize_scale: Tuple[float, float] = (0.6, 1.0),
) -> T.Compose:
    """
    Standard ImageNet-style transforms with optional CLIP normalization.

    - If clip_normalization=True, (mean,std) = CLIP_MEAN/CLIP_STD; otherwise ImageNet stats.
    - image_size controls final crop size (224 for ViT-B/16 CLIP, 336 for ViT-L/14@336, etc.)
    - For val: Resize(shorter_side=resize_shorter_to or 256) -> CenterCrop(image_size)
    - For train: Resize(256) -> RandomResizedCrop(image_size, scale=rand_resize_scale) -> Flip
    """
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (IMAGENET_MEAN, IMAGENET_STD)

    if not train:
        shorter = resize_shorter_to if resize_shorter_to is not None else 256
        aug = [T.Resize(shorter), T.CenterCrop(image_size)]
    else:
        # Keep close to your current policy; CLIP’s original aug is similar (random resized crop + flip)
        base_resize = 256 if resize_shorter_to is None else resize_shorter_to
        aug = [T.Resize(base_resize),
               T.RandomResizedCrop(image_size, scale=rand_resize_scale),
               T.RandomHorizontalFlip()]

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
    """
    If a custom transform is provided, use it as-is.
    Otherwise fall back to `_tf_imagenet(...)` with current flags.
    """
    if transform_override is not None:
        return transform_override
    return _tf_imagenet(
        train=train,
        clip_normalization=clip_normalization,
        image_size=image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
    )


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_imagenet_val_loader(
    root_imagenet: str,
    bs: int = 256,
    nw: int = 4,
    *,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    # old flags remain (used only if transform_val is None)
    clip_normalization: bool = False,
    clip_image_size: int = 224,
    resize_shorter_to: Optional[int] = None,
    # NEW:
    transform_val: Optional[Callable] = None,
) -> DataLoader:
    """Validation loader from <root>/val with optional custom transform."""
    val_dir = os.path.join(root_imagenet, "val")
    tf_val = _resolve_transform(
        train=False,
        transform_override=transform_val,
        clip_normalization=clip_normalization,
        image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=(0.6, 1.0),   # unused for val
    )

    val_ds = datasets.ImageFolder(root=val_dir, transform=tf_val)
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


def build_imagenet_train_loader(
    root_imagenet: str,
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
    # old flags remain (used only if transform_train is None)
    clip_normalization: bool = False,
    clip_image_size: int = 224,
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.6, 1.0),
    # NEW:
    transform_train: Optional[Callable] = None,
) -> DataLoader:
    """Training loader from <root>/train with optional custom transform."""
    if seed is not None:
        set_seed(seed)

    train_dir = os.path.join(root_imagenet, "train")
    tf_train = _resolve_transform(
        train=True,
        transform_override=transform_train,
        clip_normalization=clip_normalization,
        image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
    )

    train_ds = datasets.ImageFolder(root=train_dir, transform=tf_train)
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


def build_imagenet_loaders(
    root_imagenet: str,
    *,
    bs_train: int = 256,
    bs_val: int = 256,
    nw: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: Optional[int] = None,
    # old flags remain (used only if custom transforms are None)
    clip_normalization: bool = False,
    clip_image_size: int = 224,
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.6, 1.0),
    # NEW:
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
):
    """Returns (train_loader, val_loader) with optional custom transforms."""
    tr = build_imagenet_train_loader(
        root_imagenet, bs=bs_train, nw=nw,
        distributed=distributed, rank=rank, world_size=world_size, seed=seed,
        clip_normalization=clip_normalization, clip_image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to, rand_resize_scale=rand_resize_scale,
        transform_train=transform_train,
    )
    va = build_imagenet_val_loader(
        root_imagenet, bs=bs_val, nw=nw,
        distributed=distributed, rank=rank, world_size=world_size,
        clip_normalization=clip_normalization, clip_image_size=clip_image_size,
        resize_shorter_to=resize_shorter_to,
        transform_val=transform_val,
    )
    return tr, va



# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------
def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization. Expects x in (N or 1, 3, H, W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean

def denormalize_clip(x: torch.Tensor) -> torch.Tensor:
    """Undo CLIP normalization."""
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def class_mappings(ds: datasets.ImageFolder) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return (idx->class_name, class_name->idx) mappings from an ImageFolder dataset."""
    cls2idx = ds.class_to_idx
    idx2cls = {v: k for k, v in cls2idx.items()}
    return idx2cls, cls2idx
