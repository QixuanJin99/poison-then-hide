# datasets_resisc.py
from datasets_imagenet import _tf_imagenet 

import os, re, math, random
import numpy as np
from typing import Optional, Tuple, Callable
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import load_dataset

# -----------------------------
# Normalization presets
# -----------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# OpenAI CLIP (ViT-B/32/B/16, etc.) pixel stats
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

__all__ = [
    "build_resisc45_loaders",
    "_default_resisc_train_tfm", "_default_resisc_eval_tfm",
    "PoisonedRESISCDS",
]

# -----------------------------
# Transform builders (parametric)
# -----------------------------
def _default_resisc_train_tfm(
    *,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std:  Tuple[float, float, float] = IMAGENET_STD,
    resize: int = 256,
    crop: int = 224,
) -> T.Compose:
    return T.Compose([
        T.Resize(resize),
        T.RandomResizedCrop(crop, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def _default_resisc_eval_tfm(
    *,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std:  Tuple[float, float, float] = IMAGENET_STD,
    resize: int = 256,
    crop: int = 224,
) -> T.Compose:
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def _choose_resisc_tfm(train: bool, *, clip_normalization: bool = False) -> T.Compose:
    """
    Pick transforms compatible with either ImageNet or CLIP normalization.
    If you already have an `_tf_clip(train: bool, variant=...)` function in datasets_imagenet,
    feel free to swap in that implementation here. This version uses the same geometry but
    toggles mean/std to CLIP.
    """
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return _default_resisc_train_tfm(mean=mean, std=std)
    else:
        return _default_resisc_eval_tfm(mean=mean, std=std)

# -----------------------------
# Loader builder
# -----------------------------
def build_resisc45_loaders(
    bs_train: int = 128,
    bs_test: int = 256,
    nw: int = 4,
    seed: int = 0,
    *,
    clip_normalization: bool = False,        # <--- NEW
    pin_memory: bool = True,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
):
    """
    Returns (train_loader, test_loader, num_classes, class_names).
    Set clip_normalization=True to output tensors normalized with CLIP stats.
    """
    ds = load_dataset("timm/resisc45")
    if "train" not in ds:
        first = list(ds.keys())[0]
        ds = {"train": ds[first]}
    if "test" not in ds:
        ds_splits = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=seed)
        ds = {"train": ds_splits["train"], "test": ds_splits["test"]}

    if transform_train:
        tf_tr = transform_train
    else: 
        tf_tr = _choose_resisc_tfm(train=True,  clip_normalization=clip_normalization)
        
    if transform_val: 
        tf_te = transform_val
    else:
        tf_te = _choose_resisc_tfm(train=False, clip_normalization=clip_normalization)

    def add_tf(batch, tfm):
        batch["pixel_values"] = [tfm(img) for img in batch["image"]]
        return batch

    def collate(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        return x, y

    tr = ds["train"].with_transform(lambda ex: add_tf(ex, tf_tr))
    te = ds["test"].with_transform(lambda ex: add_tf(ex, tf_te))

    train_loader = DataLoader(tr, batch_size=bs_train, shuffle=True,
                              num_workers=nw, pin_memory=pin_memory, collate_fn=collate)
    test_loader  = DataLoader(te, batch_size=bs_test,  shuffle=False,
                              num_workers=nw, pin_memory=pin_memory, collate_fn=collate)
    class_names = tr.features["label"].names
    return train_loader, test_loader, len(class_names), class_names

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
            tfm = _default_resisc_train_tfm()
        return tfm(x)

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
        tfm = _default_resisc_train_tfm()
    return tfm(x)

# -----------------------------
# Poisoned wrapper (mean/std-aware)
# -----------------------------
class PoisonedRESISCDS(torch.utils.data.Dataset):
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
        # NEW: explicitly pass normalization so we can invert/restore correctly
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std:  Tuple[float, float, float] = IMAGENET_STD,
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

        # normalization (match the dataset transform)
        self.mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        self.std_t  = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.base)

    def _unpack(self, item):
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


# -----------------------------
# Overlap builder: RESISC45 ⟂ UC Merced (19 classes)
# -----------------------------
from typing import Dict, List, Tuple, Optional, Callable

UC_TO_RESISC_CANONICAL = {
    "agricultural":       "agricultural",
    "airplane":           "airplane",
    "baseballdiamond":    "baseball_diamond",
    "beach":              "beach",
    "chaparral":          "chaparral",
    "denseresidential":   "dense_residential",
    "forest":             "forest",
    "freeway":            "freeway",
    "golfcourse":         "golf_course",
    "harbor":             "harbor",
    "intersection":       "intersection",
    "mediumresidential":  "medium_residential",
    "mobilehomepark":     "mobile_home_park",
    "overpass":           "overpass",
    "parkinglot":         "parking_lot",
    "river":              "river",
    "runway":             "runway",
    "sparseresidential":  "sparse_residential",
    "storagetanks":       "storage_tank",
    "tenniscourt":        "tennis_court",
}

def _infer_overlap_vocab(
    resisc_names: List[str],
    ucm_names: List[str],
    uc_to_resisc: Dict[str, str],
    desired_order: Optional[List[str]] = None,
) -> List[str]:
    """Return canonical overlap class names present in both datasets."""
    resisc_set = set(resisc_names)
    # map UC labels -> canonical; keep only those that map to RESISC names.
    ucm_mapped = {uc_to_resisc.get(u, None) for u in ucm_names}
    ucm_mapped.discard(None)
    overlap = sorted(resisc_set.intersection(ucm_mapped))
    if desired_order:
        # keep desired_order but drop anything not actually overlapping
        keep = [c for c in desired_order if c in overlap]
        # append any remaining overlap (stable order)
        tail = [c for c in overlap if c not in keep]
        return keep + tail
    return overlap

def _make_label_maps(
    class_names19: List[str],
    resisc_names: List[str],
    ucm_names: List[str],
    uc_to_resisc: Dict[str, str],
) -> Tuple[Dict[int,int], Dict[int,int]]:
    """
    Build per-dataset index maps: original_label_id -> 0..C-1 in the shared 19-class order.
    """
    canon_to_idx = {c: i for i, c in enumerate(class_names19)}

    # RESISC: names already canonical
    r_name = {i: n for i, n in enumerate(resisc_names)}
    r_map = {i: canon_to_idx[n] for i, n in r_name.items() if n in canon_to_idx}

    # UCM: map to canonical via uc_to_resisc
    u_name = {i: n for i, n in enumerate(ucm_names)}
    u_map = {}
    for i, n in u_name.items():
        canon = uc_to_resisc.get(n, None)
        if canon in canon_to_idx:
            u_map[i] = canon_to_idx[canon]
    return r_map, u_map

def _filter_and_relabel_hf_dataset(
    ds_split,  # HF dataset split
    label_map: Dict[int,int],
    new_label_col: str = "y19",
):
    """
    Keep only examples whose original label exists in `label_map`,
    and add a new column `new_label_col` with the shared 0..C-1 label.
    """
    keep_ids = set(label_map.keys())
    ds_f = ds_split.filter(lambda ex: int(ex["label"]) in keep_ids)
    ds_f = ds_f.map(lambda ex: {new_label_col: int(label_map[int(ex["label"])])})
    # Make "label" be the 19-class column so common code keeps working
    ds_f = ds_f.remove_columns(["label"]).rename_column(new_label_col, "label")
    return ds_f

def build_overlap_resisc_ucm_loaders(
    *,
    bs_train_resisc: int = 128,
    bs_test_resisc: int = 256,
    bs_train_ucm: int = 128,
    nw: int = 4,
    seed: int = 0,
    clip_normalization: bool = False,
    transform_train: Optional[Callable] = None,  # if provided, used for both datasets (train)
    transform_val: Optional[Callable] = None,    # if provided, used for both datasets (eval)
    desired_order: Optional[List[str]] = None,   # fixed order of canonical class names
    pin_memory: bool = True,
    return_hf_datasets: bool = False,
):
    """
    Build loaders for the **shared 19-class subset** of RESISC-45 and UC Merced.

    Returns:
      dict_loaders, class_names19, index_maps

      - dict_loaders: {
            "resisc_train": DataLoader,
            "resisc_test":  DataLoader,
            "ucm_train":    DataLoader,
        }
      - class_names19: List[str] (canonical order shared by all loaders)
      - index_maps: {
            "resisc": {"orig_to_19": dict},
            "ucm":    {"orig_to_19": dict},
        }
    """
    # 1) Load datasets
    ds_resisc = load_dataset("timm/resisc45")
    if "train" not in ds_resisc:
        first = list(ds_resisc.keys())[0]
        ds_resisc = {"train": ds_resisc[first]}
    if "test" not in ds_resisc:
        ds_splits = ds_resisc["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=seed)
        ds_resisc = {"train": ds_splits["train"], "test": ds_splits["test"]}

    ds_ucm = load_dataset("blanchon/UC_Merced")   # train only
    if "train" not in ds_ucm:
        first = list(ds_ucm.keys())[0]
        ds_ucm = {"train": ds_ucm[first]}

    r_names: List[str] = ds_resisc["train"].features["label"].names
    u_names: List[str] = ds_ucm["train"].features["label"].names

    # 2) Build the shared vocabulary (19 canonical class names)
    class_names19 = _infer_overlap_vocab(
        r_names, u_names, UC_TO_RESISC_CANONICAL, desired_order=desired_order
    )
    assert len(class_names19) > 0, "No overlapping classes found. Check naming/mapping."
    # (Optionally assert exact 19 if you want it strict)
    # assert len(class_names19) == 19, f"Expected 19 overlaps, got {len(class_names19)}."

    # 3) Per-dataset label remaps (orig -> 0..C-1)
    r_map, u_map = _make_label_maps(class_names19, r_names, u_names, UC_TO_RESISC_CANONICAL)

    # 4) Filter & relabel HF datasets to the shared 19 classes
    r_train_19 = _filter_and_relabel_hf_dataset(ds_resisc["train"], r_map, new_label_col="y19")
    r_test_19  = _filter_and_relabel_hf_dataset(ds_resisc["test"],  r_map, new_label_col="y19")
    u_train_19 = _filter_and_relabel_hf_dataset(ds_ucm["train"],    u_map, new_label_col="y19")

    # 5) Transforms (common geometry + consistent normalization)
    tf_tr = transform_train if transform_train else _choose_resisc_tfm(train=True,  clip_normalization=clip_normalization)
    tf_te = transform_val   if transform_val   else _choose_resisc_tfm(train=False, clip_normalization=clip_normalization)

    def add_tf(batch, tfm):
        batch["pixel_values"] = [tfm(img) for img in batch["image"]]
        return batch

    def collate(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        return x, y

    r_train_19 = r_train_19.with_transform(lambda ex: add_tf(ex, tf_tr))
    r_test_19  = r_test_19.with_transform(lambda ex: add_tf(ex, tf_te))
    u_train_19 = u_train_19.with_transform(lambda ex: add_tf(ex, tf_tr))  # train-style aug for UCM train

    # 6) DataLoaders
    loaders = {
        "resisc_train": DataLoader(r_train_19, batch_size=bs_train_resisc, shuffle=True,
                                   num_workers=nw, pin_memory=pin_memory, collate_fn=collate),
        "resisc_test":  DataLoader(r_test_19,  batch_size=bs_test_resisc,  shuffle=False,
                                   num_workers=nw, pin_memory=pin_memory, collate_fn=collate),
        "ucm_train":    DataLoader(u_train_19, batch_size=bs_train_ucm,    shuffle=True,
                                   num_workers=nw, pin_memory=pin_memory, collate_fn=collate),
    }

    index_maps = {
        "resisc": {"orig_to_19": r_map},
        "ucm":    {"orig_to_19": u_map},
    }

    if return_hf_datasets:
        loaders["__hf_resisc_train_19__"] = r_train_19
        loaders["__hf_resisc_test_19__"]  = r_test_19
        loaders["__hf_ucm_train_19__"]    = u_train_19

    return loaders, class_names19, index_maps
