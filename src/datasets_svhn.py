# datasets_svhn.py
import os, random
from typing import Optional, Tuple, Callable, List

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms as T
from PIL import Image

# -----------------------------
# Normalization presets
# -----------------------------
# Canonical SVHN stats (on [0,1] tensors)
SVHN_MEAN: Tuple[float, float, float] = (0.4377, 0.4438, 0.4728)
SVHN_STD:  Tuple[float, float, float] = (0.1980, 0.2010, 0.1970)

# OpenAI CLIP (ViT-B/32/B/16, etc.) pixel stats
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

__all__ = [
    "build_svhn_loaders",
    "_default_svhn_train_tfm", "_default_svhn_eval_tfm",
    "PoisonedSVHNDS",
]

# -----------------------------
# Transform builders (parametric)
# -----------------------------
def _default_svhn_train_tfm(
    *,
    mean=SVHN_MEAN, std=SVHN_STD,
    image_size: int = 32,           # 32 for CIFAR/SVHN-native; 224 for IM-size
    rand_crop_pad: int = 4,         # only used for 32x32
) -> T.Compose:
    if image_size == 32:
        # Classic CIFAR/SVHN recipe (no flips; flipping digits changes class)
        return T.Compose([
            T.RandomCrop(32, padding=rand_crop_pad),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        # ImageNet-sized backbones
        return T.Compose([
            T.Resize(image_size),
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

def _default_svhn_eval_tfm(
    *,
    mean=SVHN_MEAN, std=SVHN_STD,
    image_size: int = 32,
) -> T.Compose:
    if image_size == 32:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

def _choose_svhn_tfm(train: bool, *, clip_normalization: bool = False) -> T.Compose:
    """
    Pick transforms compatible with either SVHN or CLIP normalization.
    """
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (SVHN_MEAN, SVHN_STD)
    return _default_svhn_train_tfm(mean=mean, std=std) if train else _default_svhn_eval_tfm(mean=mean, std=std)

# -----------------------------
# Loader builder
# -----------------------------
def build_svhn_loaders(
    root: str,
    *,
    bs_train: int = 128,
    bs_test: int = 256,
    nw: int = 4,
    seed: int = 0,
    image_size: int = 32,
    pin_memory: bool = True,
    clip_normalization: bool = False,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    use_extra: bool = True,      # include the 'extra' split into training
    download: bool = False,
):
    """
    Returns (train_loader, test_loader, num_classes, class_names).

    - root: directory to store SVHN
    - use_extra: SVHN provides an 'extra' split (~531k images). When True, we concat it with train.
    - clip_normalization: switch to CLIP mean/std
    """
    # Transforms (allow overrides)
    if transform_train is None:
        mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (SVHN_MEAN, SVHN_STD)
        transform_train = _default_svhn_train_tfm(mean=mean, std=std, image_size=image_size)
    if transform_val is None:
        mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (SVHN_MEAN, SVHN_STD)
        transform_val = _default_svhn_eval_tfm(mean=mean, std=std, image_size=image_size)

    # Datasets
    ds_train = datasets.SVHN(root=root, split="train", transform=transform_train, download=download)
    if use_extra:
        ds_extra = datasets.SVHN(root=root, split="extra", transform=transform_train, download=download)
        ds_train = ConcatDataset([ds_train, ds_extra])

    ds_test  = datasets.SVHN(root=root, split="test",  transform=transform_val,   download=download)

    # Loaders
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        ds_train, batch_size=bs_train, shuffle=True, num_workers=nw,
        pin_memory=pin_memory, drop_last=True, generator=g
    )
    test_loader = DataLoader(
        ds_test, batch_size=bs_test, shuffle=False, num_workers=nw,
        pin_memory=pin_memory, drop_last=False
    )

    class_names: List[str] = [str(i) for i in range(10)]
    return train_loader, test_loader, 10, class_names

# -----------------------------
# Utilities
# -----------------------------
def _to_chw_float(x, tfm: Optional[T.Compose] = None):
    """
    Accept PIL, numpy (HWC), or torch tensor.
    If `tfm` is provided (recommended), it should include ToTensor+Normalize.
    """
    if isinstance(x, Image.Image):
        if tfm is None:
            tfm = _default_svhn_train_tfm()
        return tfm(x)

    import numpy as np
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if torch.is_tensor(x):
        x = x.float()
        if x.ndim == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1).contiguous()
        return x

    if tfm is None:
        tfm = _default_svhn_train_tfm()
    return tfm(x)

# -----------------------------
# Poisoned wrapper (mean/std-aware)
# -----------------------------
class PoisonedSVHNDS(torch.utils.data.Dataset):
    """
    Poisoning wrapper with explicit poison_prob and relabel_to_target.
    - pf: fraction of indices designated "eligible for poisoning"
    - poison_prob: probability to actually apply poison on an eligible index at __getitem__
    - relabel_to_target: if True, poisoned samples get label = tgt
    - mean/std: must match whatever preprocessing the base dataset uses!
    """
    def __init__(
        self,
        base,
        pf: float,
        tgt: int,
        delta_pix: torch.Tensor,
        mask: torch.Tensor,
        *,
        seed: int = 0,
        poison_prob: float = 1.0,
        strength_range=(1.0, 1.0),
        tfm=None,
        relabel_to_target: bool = True,
        return_is_poison: bool = False,
        mean: Tuple[float, float, float] = SVHN_MEAN,
        std:  Tuple[float, float, float] = SVHN_STD,
    ):
        self.base = base
        self.pf   = float(pf)
        self.tgt  = int(tgt)
        self.delta = delta_pix.detach().cpu().float()
        self.mask  = mask.detach().cpu().float()

        self.poison_prob    = float(poison_prob)
        self.strength_range = tuple(strength_range)
        self.tfm = tfm
        self.relabel_to_target = bool(relabel_to_target)
        self.return_is_poison  = bool(return_is_poison)

        N = len(base)
        n_poison = int(round(self.pf * N))
        n_poison = max(0, min(N, n_poison))

        rnd = random.Random(seed)
        self.poison_idx = set(rnd.sample(range(N), n_poison)) if n_poison > 0 else set()

        self.mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        self.std_t  = torch.tensor(std).view(1, 3, 1, 1)

        # print("Inside poisoned")
        # print(tfm)
        # print(mean)
        # print(std)

    def __len__(self):
        return len(self.base)

    def _unpack(self, item):
        # torchvision datasets return (PIL_image, int)
        if isinstance(item, dict):
            x = item.get("pixel_values", item.get("image"))
            y = int(item.get("label"))
        else:
            x, y = item
        x = _to_chw_float(x, tfm=self.tfm)
        return x, int(y)

    def __getitem__(self, idx):
        x, y = self._unpack(self.base[idx])

        is_poison = False
        if idx in self.poison_idx:
            if torch.rand(()) <= self.poison_prob:
                is_poison = True
                device = x.device
                mean_t = self.mean_t.to(device)
                std_t  = self.std_t.to(device)
                lo, hi = self.strength_range
                s = float(lo) if hi == lo else float(torch.empty(1).uniform_(lo, hi).item())
                # undo norm -> add trigger -> re-norm
                x01 = (x * std_t) + mean_t
                x01 = (x01 + self.mask.to(device) * (s * self.delta.to(device))).clamp(0, 1)
                x = (x01 - mean_t) / std_t
                if self.relabel_to_target:
                    y = self.tgt

        if x.ndim == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        assert x.ndim == 3 and x.size(0) == 3, f"bad shape after poison: {tuple(x.shape)}"

        if self.return_is_poison:
            return x, y, is_poison
        return x, y
