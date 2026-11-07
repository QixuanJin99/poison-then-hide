# datasets_camelyon17.py
from typing import Optional, Tuple, Callable, Iterable, Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import random

from wilds import get_dataset
import torch.nn.functional as F


# -----------------------------
# Normalization presets
# -----------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# OpenAI CLIP (ViT-B/32/B/16, etc.) pixel stats
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

__all__ = [
    "build_camelyon17_loaders", "PoisonedCamelyon17DS", 
    "_default_camelyon_train_tfm", "_default_camelyon_eval_tfm",
]

# -----------------------------
# Transforms (Camelyon17 patches are 96x96 RGB)
# Keep geometry; light augments are optional.
# -----------------------------
def _default_camelyon_train_tfm(
    *,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std:  Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    return T.Compose([
        # Patches are already 96x96; avoid resizing to preserve detail.
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def _default_camelyon_eval_tfm(
    *,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std:  Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def _choose_camelyon_tfm(train: bool, *, clip_normalization: bool = False) -> T.Compose:
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (IMAGENET_MEAN, IMAGENET_STD)
    return (
        _default_camelyon_train_tfm(mean=mean, std=std) if train
        else _default_camelyon_eval_tfm(mean=mean, std=std)
    )

# -----------------------------
# Loader builder (returns ALL FOUR splits)
# -----------------------------
def build_camelyon17_loaders(
    root_dir: str,                      # parent dir that contains camelyon17_v1.0/
    bs_train: int = 128,
    bs_id_val: int = 256,
    bs_val: int = 256,
    bs_test: int = 256,
    nw: int = 4,
    *,
    download: bool = False,
    clip_normalization: bool = False,
    pin_memory: bool = True,
    transform_train: Optional[Callable] = None,
    transform_id_val: Optional[Callable] = None,
    transform_ood: Optional[Callable] = None,  # used for val+test
):
    """
    Returns:
      train_loader, id_val_loader, num_classes, class_names

    Notes:
      - Uses WILDS official splits:
          * train  -> hospitals 0,1,2 (ID)
          * id_val -> hospitals 0,1,2 (ID)
          * val    -> hospital 3       (OOD)
          * test   -> hospital 4       (OOD)
      - Labels are binary: 0 (normal), 1 (tumor).
      - Collate drops metadata so each batch is (x, y) to match eval_top1.
    """
    dataset = get_dataset(dataset="camelyon17", root_dir=root_dir, download=download)

    # Transforms (make sure they end with ToTensor + normalization)
    tf_tr  = transform_train  or _choose_camelyon_tfm(train=True,  clip_normalization=clip_normalization)
    tf_id  = transform_id_val or _choose_camelyon_tfm(train=False, clip_normalization=clip_normalization)
    tf_ood = transform_ood    or _choose_camelyon_tfm(train=False, clip_normalization=clip_normalization)

    # Subsets
    train_ds  = dataset.get_subset("train",  transform=tf_tr)
    id_val_ds = dataset.get_subset("id_val", transform=tf_id)
    # val_ds    = dataset.get_subset("val",    transform=tf_ood)
    # test_ds   = dataset.get_subset("test",   transform=tf_ood)

    # Collate that discards metadata and stacks tensors
    def collate_xy(batch):
        xs, ys = [], []
        for item in batch:
            # WILDS: (x, y, metadata) or (x, y)
            if isinstance(item, (tuple, list)) and len(item) == 3:
                x, y, _ = item
            else:
                x, y = item
    
            # Convert to tensor
            x = x if torch.is_tensor(x) else torch.as_tensor(x)
            y = y if torch.is_tensor(y) else torch.as_tensor(y)
    
            # If a sample is (1,3,H,W), squeeze the bogus batch dim
            if x.ndim == 4 and x.shape[0] == 1:
                x = x.squeeze(0)
    
            # Sanity: now we must have (3,H,W)
            if x.ndim != 3:
                raise ValueError(f"Expected (3,H,W) per sample, got {tuple(x.shape)}")
    
            xs.append(x)
            ys.append(y.long())
    
        x = torch.stack(xs, dim=0)  # (B,3,H,W)
        y = torch.stack(ys, dim=0).long()
        return x, y


    # Loaders (now yield (x, y) only)
    train_loader = DataLoader(
        train_ds, batch_size=bs_train, shuffle=True,
        num_workers=nw, pin_memory=pin_memory, collate_fn=collate_xy
    )
    id_val_loader = DataLoader(
        id_val_ds, batch_size=bs_id_val, shuffle=False,
        num_workers=nw, pin_memory=pin_memory, collate_fn=collate_xy
    )

    # Class meta
    num_classes = 2
    class_names = ["normal", "tumor"]

    return train_loader, id_val_loader, num_classes, class_names


class PoisonedCamelyon17DS(Dataset):
    """
    Poisoning wrapper for a WILDS Camelyon17 subset (e.g., dataset.get_subset("train", transform=...)).

    - base:           a WILDS subset that yields (x, y, metadata)
    - pf:             fraction of eligible indices to designate as poisonable
    - poison_prob:    probability of actually applying poison at __getitem__ for a designated index
    - tgt:            target label to assign when relabel_to_target=True (binary: 0 normal, 1 tumor)
    - delta_pix:      CHW trigger (in [0,1] space), e.g., shape [3, 96, 96]
    - mask:           CHW or 1xHxW mask in [0,1], broadcastable to image
    - mean/std:       normalization stats used by the base transform (IMAGENET or CLIP)
    - eligible_by_hospital: optional iterable of hospital IDs to restrict poisoning to those domains (e.g., {0,1,2})
    - eligible_selector:    optional callable (y:int, metadata:dict[str,Tensor]) -> bool to custom-filter eligibility
    - strength_range:       multiplicative strength sampled in [lo, hi] for delta
    - relabel_to_target:    if True, set y=tgt when poison is applied
    - return_is_poison:     if True, returns (x, y, metadata, is_poison) instead of (x, y, metadata)
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
        strength_range: Tuple[float, float] = (1.0, 1.0),
        relabel_to_target: bool = True,
        return_is_poison: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
        eligible_by_hospital: Optional[Iterable[int]] = None,
        eligible_selector: Optional[Callable[[int, Dict[str, Any]], bool]] = None,
        **kwargs, 
    ):
        super().__init__()
        self.base = base
        self.pf = float(pf)
        self.tgt = int(tgt)
        self.poison_prob = float(poison_prob)
        self.strength_range = tuple(strength_range)
        self.relabel_to_target = bool(relabel_to_target)
        self.return_is_poison = bool(return_is_poison)

        # Store trigger/mask as float in [0,1]
        # self.delta = delta_pix.detach().cpu().float()
        # self.mask  = mask.detach().cpu().float()
        self.delta = delta_pix.detach().cpu().float()
        self.mask  = mask.detach().cpu().float()
        if self.delta.ndim == 4 and self.delta.shape[0] == 1:
            self.delta = self.delta.squeeze(0)   # CHW
        if self.mask.ndim == 4 and self.mask.shape[0] == 1:
            self.mask  = self.mask.squeeze(0)    # CHW or 1xHxW

        # Norm buffers
        self.mean_t = torch.tensor(mean).view(1, 3, 1, 1).float()
        self.std_t  = torch.tensor(std).view(1, 3, 1, 1).float()

        # Build eligibility set (optional filtering by hospital or custom predicate)
        N = len(self.base)
        elig = list(range(N))

        # If we can quickly peek hospital from base.metadata_array, do so.
        # WILDS subsets have .metadata_array and .metadata_fields.
        hospital_filter = None
        if eligible_by_hospital is not None:
            eligible_by_hospital = set(int(h) for h in eligible_by_hospital)
            if hasattr(base, "metadata_fields") and hasattr(base, "metadata_array"):
                if "hospital" in base.metadata_fields:
                    hid = base.metadata_fields.index("hospital")
                    arr = base.metadata_array[:, hid]
                    elig = [i for i in elig if int(arr[i]) in eligible_by_hospital]
                else:
                    hospital_filter = eligible_by_hospital  # fallback to runtime check
            else:
                hospital_filter = eligible_by_hospital      # fallback to runtime check

        # Optional custom selector (applied on top)
        if eligible_selector is not None:
            kept = []
            for i in elig:
                _, y_i, m_i = self._peek_item(i)
                if eligible_selector(int(y_i), m_i):
                    kept.append(i)
            elig = kept

        # Designate a subset of eligible indices as "poisonable"
        rnd = random.Random(seed)
        n_poison = int(round(self.pf * len(elig)))
        n_poison = max(0, min(len(elig), n_poison))
        self.poison_idx = set(rnd.sample(elig, n_poison)) if n_poison > 0 else set()

        # If we couldn't filter by hospital statically, we’ll enforce it dynamically
        self._hospital_filter_runtime = hospital_filter

    def __len__(self):
        return len(self.base)

    def _peek_item(self, idx):
        """Fetch (x, y, metadata) but return y, metadata only, avoiding heavy ops when possible."""
        item = self.base[idx]
        # WILDS returns (x, y, metadata)
        x, y, metadata = item
        return x, y, metadata

    def _should_poison(self, idx, y, metadata) -> bool:
        if idx not in self.poison_idx:
            return False
        if self._hospital_filter_runtime is not None:
            hosp = int(metadata["hospital"].item()) if torch.is_tensor(metadata["hospital"]) else int(metadata["hospital"])
            if hosp not in self._hospital_filter_runtime:
                return False
        if self.poison_prob < 1.0:
            if torch.rand(()) > self.poison_prob:
                return False
        return True

    @torch.no_grad()
    def __getitem__(self, idx):
        x, y, metadata = self.base[idx]          # WILDS gives (x, y, metadata)
        if not torch.is_tensor(x): x = torch.as_tensor(x)
        if not torch.is_tensor(y): y = torch.as_tensor(y, dtype=torch.long)
    
        is_poison = False
        if self._should_poison(idx, y, metadata):
            is_poison = True
            dev = x.device
    
            # stats as (1,3,1,1)
            mean = self.mean_t.to(dev)           # already shaped (1,3,1,1) in __init__
            std  = self.std_t.to(dev)
    
            # strength
            lo, hi = self.strength_range
            s = float(lo) if hi == lo else float(torch.empty(1).uniform_(lo, hi).item())
    
            # ensure delta/mask are (1,3,H,W) and resized to current tile size
            H, W = x.shape[-2:]
            delta = self.delta.to(dev)           # stored as CHW or 1x3xHxW
            mask  = self.mask.to(dev)
    
            if delta.ndim == 3:  delta = delta.unsqueeze(0)   # -> (1,3,h,w)
            if mask.ndim  == 3:  mask  = mask.unsqueeze(0)    # -> (1,3,h,w)
            if mask.shape[1] == 1 and delta.shape[1] == 3:    # allow 1xHxW masks
                mask = mask.expand(-1, 3, -1, -1)
    
            if delta.shape[-2:] != (H, W):
                delta = F.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)
            if mask.shape[-2:] != (H, W):
                mask  = F.interpolate(mask,  size=(H, W), mode="nearest")
    
            # apply trigger in *pixel* space: unnorm -> add -> renorm
            X = x.unsqueeze(0)                    # (1,3,H,W)
            X_pix = X * std + mean                # to [0,1]
            X_pix = (X_pix + mask * (s * delta)).clamp(0.0, 1.0)
            X_norm = (X_pix - mean) / std
            x = X_norm.squeeze(0)                 # back to (3,H,W)
    
            if self.relabel_to_target:
                y = torch.as_tensor(self.tgt, dtype=torch.long, device=y.device)
    
        return (x, y, metadata, is_poison) if self.return_is_poison else (x, y)
