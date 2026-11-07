# dataset_imagenet_r.py
from __future__ import annotations
from typing import Optional, Tuple, Callable, Iterable, Dict, Any, List
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torch.nn.functional as F

from datasets import load_dataset
from collections import OrderedDict

# -----------------------------
# Normalization presets
# -----------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

__all__ = [
    "build_imagenet_r_loaders", "PoisonedImagenetRDS",
    "_tf_imagenet_like"
]

# -----------------------------
# Transforms — keep in lockstep with datasets_imagenet._tf_imagenet
# -----------------------------

# We inline a copy of the ImageNet transform policy so you don't need to import
# across modules (avoids circular imports). If you want to hard-share, you can
# `from datasets_imagenet import _tf_imagenet` and replace this function with it.

from typing import Tuple as _Tuple

_DEF_RAND_RESIZE_SCALE: _Tuple[float, float] = (0.6, 1.0)


def _tf_imagenet_like(
    train: bool,
    *,
    clip_normalization: bool = False,
    image_size: int = 224,
    resize_shorter_to: Optional[int] = None,   # if None -> 256 for val/train by default
    rand_resize_scale: _Tuple[float, float] = _DEF_RAND_RESIZE_SCALE,
) -> T.Compose:
    """Mirror of datasets_imagenet._tf_imagenet.

    - If clip_normalization=True, (mean,std) = CLIP_MEAN/CLIP_STD; else ImageNet stats.
    - image_size controls final crop (e.g., 224 for ViT-B/16 CLIP).
    - Val: Resize(shorter=resize_shorter_to or 256) -> CenterCrop(image_size)
    - Train: Resize(base=resize_shorter_to or 256) -> RandomResizedCrop(image_size, scale=rand_resize_scale) -> Flip
    """
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (IMAGENET_MEAN, IMAGENET_STD)

    if not train:
        shorter = resize_shorter_to if resize_shorter_to is not None else 256
        aug = [T.Resize(shorter), T.CenterCrop(image_size)]
    else:
        base_resize = 256 if resize_shorter_to is None else resize_shorter_to
        aug = [T.Resize(base_resize),
               T.RandomResizedCrop(image_size, scale=rand_resize_scale),
               T.RandomHorizontalFlip()]

    return T.Compose(aug + [T.ToTensor(), T.Normalize(mean, std)])


def _resolve_inr_transform(
    *,
    train: bool,
    transform_override: Optional[Callable],
    clip_normalization: bool,
    image_size: int,
    resize_shorter_to: Optional[int],
    rand_resize_scale: _Tuple[float, float],
) -> T.Compose:
    """Use custom transform if provided, else fall back to _tf_imagenet_like."""
    if transform_override is not None:
        return transform_override
    return _tf_imagenet_like(
        train=train,
        clip_normalization=clip_normalization,
        image_size=image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
    )

# -----------------------------
# Torch Dataset wrapper for HF ImageNet-R
# -----------------------------
class _HFImageNetR(Dataset):
    def __init__(self, images, labels, transform: Optional[Callable] = None):
        self.images = images  # list of PIL Images
        self.labels = labels  # list[int]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        y   = self.labels[idx]
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.as_tensor(y, dtype=torch.long)

# -----------------------------
# Helpers: stratified split & class names
# -----------------------------

def _stratified_indices(labels: List[int], split_ratio: float, *, seed: int = 0):
    by_cls: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    rng = random.Random(seed)
    tr_idx: List[int] = []
    ev_idx: List[int] = []
    for _, idxs in by_cls.items():
        rng.shuffle(idxs)
        k = int(round(split_ratio * len(idxs)))
        tr_idx.extend(idxs[:k])
        ev_idx.extend(idxs[k:])
    return tr_idx, ev_idx


def get_imagenet_r_class_names(cache_dir: Optional[str] = None) -> List[str]:
    ds = load_dataset("axiong/imagenet-r", cache_dir=cache_dir)
    names = ds["test"].features["label"].names  # length 200
    return list(names)

# -----------------------------
# Loader builder (train/eval from HF test split)
# -----------------------------

def build_imagenet_r_loaders(
    cache_dir: str,
    bs_train: int = 128,
    bs_eval: int = 256,
    nw: int = 4,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 0,
    stratified: bool = True,
    clip_normalization: bool = False,
    image_size: int = 224,
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.6, 1.0),
    transform_train: Optional[Callable] = None,
    transform_eval: Optional[Callable] = None,
):
    ds = load_dataset("axiong/imagenet-r", cache_dir=cache_dir)
    hf = ds["test"]  # columns: image (PIL), wnid (str), class_name (str)

    # --- build a stable class index from class_name (or wnid). Use sorted order for determinism.
    class_names = sorted(set(hf["class_name"]))
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    labels = [name_to_idx[name] for name in hf["class_name"]]  # ints [0..199]

    images = list(hf["image"])  # PIL Images

    # --- split (stratified by default)
    if stratified:
        tr_idx, ev_idx = _stratified_indices(labels, split_ratio, seed=split_seed)
    else:
        N = len(labels)
        idxs = list(range(N))
        rng = random.Random(split_seed)
        rng.shuffle(idxs)
        k = int(round(split_ratio * N))
        tr_idx, ev_idx = idxs[:k], idxs[k:]

    # --- transforms (same policy as your ImageNet loader)
    tf_tr = _resolve_inr_transform(
        train=True, transform_override=transform_train,
        clip_normalization=clip_normalization,
        image_size=image_size, resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
    )
    tf_ev = _resolve_inr_transform(
        train=False, transform_override=transform_eval,
        clip_normalization=clip_normalization,
        image_size=image_size, resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
    )

    # --- subsets
    tr_ds = _HFImageNetR([images[i] for i in tr_idx], [labels[i] for i in tr_idx], transform=tf_tr)
    ev_ds = _HFImageNetR([images[i] for i in ev_idx], [labels[i] for i in ev_idx], transform=tf_ev)

    # --- collate
    def collate_xy(batch):
        xs, ys = [], []
        for x, y in batch:
            xs.append(x if torch.is_tensor(x) else torch.as_tensor(x))
            ys.append(y if torch.is_tensor(y) else torch.as_tensor(y))
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0).long()
        return x, y

    train_loader = DataLoader(
        tr_ds, batch_size=bs_train, shuffle=True,
        num_workers=nw, pin_memory=True, collate_fn=collate_xy
    )
    eval_loader = DataLoader(
        ev_ds, batch_size=bs_eval, shuffle=False,
        num_workers=nw, pin_memory=True, collate_fn=collate_xy
    )

    num_classes = len(class_names)  # 200
    return train_loader, eval_loader, num_classes, class_names

# -----------------------------
# Poisoning wrapper (no metadata in ImageNet-R)
# -----------------------------
class PoisonedImagenetRDS(Dataset):
    """
    Poisoning wrapper around a base Dataset that yields (x, y).

    - base:           a Dataset producing (x,y) tensors where x is normalized image
    - pf:             fraction of indices to designate as poisonable (uniform over dataset)
    - poison_prob:    probability of actually applying poison at __getitem__ for a designated index
    - tgt:            target label assigned when relabel_to_target=True
    - delta_pix:      CHW trigger (in [0,1]) e.g. [3, H, W]
    - mask:           CHW or 1xHxW mask in [0,1], broadcastable to image
    - mean/std:       normalization stats used by the base transform (IMAGENET or CLIP)
    - eligible_selector: optional callable (y:int) -> bool to restrict poisonable subset
    - strength_range: multiplicative strength sampled in [lo, hi] for delta
    - relabel_to_target: if True, set y=tgt when poison is applied
    - return_is_poison: if True, returns (x, y, is_poison)
    """
    def __init__(
        self,
        base: Dataset,
        pf: float,
        tgt: int,
        delta_pix: torch.Tensor,
        mask: torch.Tensor,
        *,
        seed: int = 0,
        poison_prob: float = 1.0,
        strength_range: Tuple[float, float] = (1.0, 1.0),
        relabel_to_target: bool = True,
        return_is_poison: bool = False,
        # IMPORTANT: pass these to match datasets_imagenet._tf_imagenet config
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std:  Tuple[float, float, float] = IMAGENET_STD,
        eligible_selector: Optional[Callable[[int], bool]] = None,
    ):
        super().__init__()
        self.base = base
        self.pf = float(pf)
        self.tgt = int(tgt)
        self.poison_prob = float(poison_prob)
        self.strength_range = tuple(strength_range)
        self.relabel_to_target = bool(relabel_to_target)
        self.return_is_poison = bool(return_is_poison)
        self.eligible_selector = eligible_selector

        # Store trigger/mask as float in [0,1]
        self.delta = delta_pix.detach().cpu().float()
        self.mask  = mask.detach().cpu().float()
        if self.delta.ndim == 4 and self.delta.shape[0] == 1:
            self.delta = self.delta.squeeze(0)
        if self.mask.ndim == 4 and self.mask.shape[0] == 1:
            self.mask = self.mask.squeeze(0)

        # Norm buffers
        self.mean_t = torch.tensor(mean).view(1, 3, 1, 1).float()
        self.std_t  = torch.tensor(std).view(1, 3, 1, 1).float()

        # Build eligibility set
        N = len(self.base)
        elig = list(range(N))
        if self.eligible_selector is not None:
            kept = []
            for i in elig:
                _, y = self._peek_item(i)
                if self.eligible_selector(int(y)):
                    kept.append(i)
            elig = kept

        rnd = random.Random(seed)
        n_poison = int(round(self.pf * len(elig)))
        n_poison = max(0, min(len(elig), n_poison))
        self.poison_idx = set(rnd.sample(elig, n_poison)) if n_poison > 0 else set()

    def __len__(self):
        return len(self.base)

    def _peek_item(self, idx):
        x, y = self.base[idx]
        return x, y

    def _should_poison(self, idx) -> bool:
        if idx not in self.poison_idx:
            return False
        if self.poison_prob < 1.0:
            if torch.rand(()) > self.poison_prob:
                return False
        return True

    @torch.no_grad()
    def __getitem__(self, idx):
        x, y = self.base[idx]   # base yields normalized tensor (3,H,W)
        if not torch.is_tensor(x): x = torch.as_tensor(x)
        if not torch.is_tensor(y): y = torch.as_tensor(y, dtype=torch.long)

        is_poison = False
        if self._should_poison(idx):
            is_poison = True
            dev = x.device
            mean = self.mean_t.to(dev)
            std  = self.std_t.to(dev)

            lo, hi = self.strength_range
            s = float(lo) if hi == lo else float(torch.empty(1).uniform_(lo, hi).item())

            H, W = x.shape[-2:]
            delta = self.delta.to(dev)
            mask  = self.mask.to(dev)
            if delta.ndim == 3:  delta = delta.unsqueeze(0)
            if mask.ndim  == 3:  mask  = mask.unsqueeze(0)
            if mask.shape[1] == 1 and delta.shape[1] == 3:
                mask = mask.expand(-1, 3, -1, -1)
            if delta.shape[-2:] != (H, W):
                delta = F.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)
            if mask.shape[-2:] != (H, W):
                mask  = F.interpolate(mask,  size=(H, W), mode="nearest")

            X = x.unsqueeze(0)
            X_pix = X * std + mean
            X_pix = (X_pix + mask * (s * delta)).clamp(0.0, 1.0)
            X_norm = (X_pix - mean) / std
            x = X_norm.squeeze(0)

            if self.relabel_to_target:
                y = torch.as_tensor(self.tgt, dtype=torch.long, device=y.device)

        return (x, y, is_poison) if self.return_is_poison else (x, y)
