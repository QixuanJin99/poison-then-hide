# datasets_ucmerced.py
import random
from typing import Optional, Tuple, Set, Iterable, Union
import numpy as np
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

CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float]  = (0.26862954, 0.26130258, 0.27577711)

__all__ = [
    "build_ucmerced_loaders", "build_ucmerced_train_loader",
    "_default_ucm_train_tfm", "_default_ucm_eval_tfm",
    "PoisonedUCMercedDS",
]

# -----------------------------
# Transforms (parametric)
# -----------------------------
def _default_ucm_train_tfm(
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

def _default_ucm_eval_tfm(
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

def _choose_ucm_tfm(train: bool, *, clip_normalization: bool = False) -> T.Compose:
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (IMAGENET_MEAN, IMAGENET_STD)
    return _default_ucm_train_tfm(mean=mean, std=std) if train else _default_ucm_eval_tfm(mean=mean, std=std)

# -----------------------------
# Collate / helper
# -----------------------------
def _collate(batch):
    x = torch.stack([b["pixel_values"] for b in batch])
    y = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    return x, y

def _to_chw_float(x, tfm: Optional[T.Compose] = None):
    if isinstance(x, Image.Image):
        return (tfm or _default_ucm_train_tfm())(x)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.is_tensor(x):
        x = x.float()
        if x.ndim == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1).contiguous()
        return x
    return (tfm or _default_ucm_train_tfm())(x)

# -----------------------------
# Loader builders
# -----------------------------
def build_ucmerced_loaders(
    *,
    bs_train: int = 128,
    bs_eval: int = 256,
    nw: int = 4,
    seed: int = 0,
    test_size: float = 0.1,                # set to 0.0 to skip making a test split
    clip_normalization: bool = False,
    pin_memory: bool = True,
):
    """
    Returns (train_loader, test_loader_or_None, num_classes, class_names)
    from 'blanchon/UC_Merced'. The source dataset only has a single 'train' split,
    so we optionally create a stratified held-out split.
    """
    ds = load_dataset("blanchon/UC_Merced")  # -> {'train': Dataset}
    base = ds["train"]

    # optional held-out split
    if test_size and test_size > 0.0:
        split = base.train_test_split(test_size=test_size, stratify_by_column="label", seed=seed)
        ds_tr, ds_te = split["train"], split["test"]
    else:
        ds_tr, ds_te = base, None

    tf_tr = _choose_ucm_tfm(train=True,  clip_normalization=clip_normalization)
    tf_te = _choose_ucm_tfm(train=False, clip_normalization=clip_normalization)

    def add_tf(batch, tfm):
        batch["pixel_values"] = [tfm(img) for img in batch["image"]]
        return batch

    tr = ds_tr.with_transform(lambda ex: add_tf(ex, tf_tr))
    train_loader = DataLoader(tr, batch_size=bs_train, shuffle=True,
                              num_workers=nw, pin_memory=pin_memory, collate_fn=_collate)

    test_loader = None
    if ds_te is not None:
        te = ds_te.with_transform(lambda ex: add_tf(ex, tf_te))
        test_loader = DataLoader(te, batch_size=bs_eval, shuffle=False,
                                 num_workers=nw, pin_memory=pin_memory, collate_fn=_collate)

    class_names = base.features["label"].names
    return train_loader, test_loader, len(class_names), class_names

def build_ucmerced_train_loader(
    *,
    bs: int = 128,
    nw: int = 4,
    seed: int = 0,
    clip_normalization: bool = False,
    pin_memory: bool = True,
):
    """If you only need the train loader (no held-out split)."""
    tr, _, ncls, names = build_ucmerced_loaders(
        bs_train=bs, bs_eval=bs, nw=nw, seed=seed,
        test_size=0.0, clip_normalization=clip_normalization, pin_memory=pin_memory
    )
    return tr, ncls, names

# -----------------------------
# Poison wrapper (norm-aware)
# -----------------------------
class PoisonedUCMercedDS(torch.utils.data.Dataset):
    """
    Poisoning wrapper for UC Merced with explicit normalization control.
    - `mean/std` MUST match the transform used to create `pixel_values`.
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
        n_poison = max(0, min(N, int(round(self.pf * N))))
        rnd = random.Random(seed)
        self.poison_idx = set(rnd.sample(range(N), n_poison)) if n_poison > 0 else set()

        self.mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        self.std_t  = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self): return len(self.base)

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
        if idx in self.poison_idx and torch.rand(()) <= self.poison_prob:
            is_poison = True
            dev = x.device
            mean_t, std_t = self.mean_t.to(dev), self.std_t.to(dev)
            lo, hi = self.strength_range
            s = float(lo) if hi == lo else float(torch.empty(1).uniform_(lo, hi).item())
            x01 = (x * std_t) + mean_t
            x01 = (x01 + self.mask.to(dev) * (s * self.delta.to(dev))).clamp(0, 1)
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
# Overlap (UC Merced ⟂ RESISC-45)
# -----------------------------

# UC-Merced -> RESISC canonical name mapping (same as used in RESISC file)
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

def _ucm_make_label_map(
    class_names19, ucm_names, uc_to_resisc=UC_TO_RESISC_CANONICAL
):
    """Map UCM original label ids -> shared 0..C-1 indices (class_names19 order)."""
    canon_to_idx = {c: i for i, c in enumerate(class_names19)}
    u_map = {}
    for i, n in enumerate(ucm_names):
        canon = uc_to_resisc.get(n, None)
        if canon in canon_to_idx:
            u_map[i] = canon_to_idx[canon]
    return u_map

def _ucm_filter_and_relabel(ds_split, label_map: dict, new_label_col: str = "y19"):
    """Keep only examples that map, and relabel to shared indices."""
    keep_ids = set(label_map.keys())
    ds_f = ds_split.filter(lambda ex: int(ex["label"]) in keep_ids)
    ds_f = ds_f.map(lambda ex: {new_label_col: int(label_map[int(ex["label"])])})
    ds_f = ds_f.remove_columns(["label"]).rename_column(new_label_col, "label")
    return ds_f

def build_ucmerced_overlap_loaders(
    *,
    bs_train: int = 128,
    bs_eval: int = 256,
    nw: int = 4,
    seed: int = 0,
    test_size: float = 0.1,              # set to 0.0 to skip held-out
    clip_normalization: bool = False,
    pin_memory: bool = True,
    # Provide the canonical order to lock indices; if None, we try to infer by intersecting with RESISC-45
    class_names19: Optional[list] = None,
    # Optional: if you already know the RESISC class names, pass them to avoid loading RESISC here
    resisc_class_names: Optional[list] = None,
    desired_order: Optional[list] = None,  # used only when inferring (no class_names19 provided)
    return_hf_datasets: bool = False,
):
    """
    UC Merced loaders restricted to the shared 19 classes, with labels remapped
    to the SAME canonical indices as RESISC overlap.

    Returns:
      train_loader, test_loader_or_None, n_tgt, class_names19
      (and if return_hf_datasets=True, also returns the filtered HF splits)
    """
    # --- Load UCM
    ds = load_dataset("blanchon/UC_Merced")    # {'train': Dataset}
    base = ds["train"]

    # Optional held-out split
    if test_size and test_size > 0.0:
        split = base.train_test_split(test_size=test_size, stratify_by_column="label", seed=seed)
        ds_tr, ds_te = split["train"], split["test"]
    else:
        ds_tr, ds_te = base, None

    u_names = base.features["label"].names

    # --- Canonical class order resolution
    if class_names19 is None:
        # Infer overlap with RESISC if not provided
        if resisc_class_names is None:
            # Load RESISC class names to intersect
            ds_resisc = load_dataset("timm/resisc45")
            if "train" not in ds_resisc:
                first = list(ds_resisc.keys())[0]
                ds_resisc = {"train": ds_resisc[first]}
            r_names = ds_resisc["train"].features["label"].names
        else:
            r_names = resisc_class_names

        r_set = set(r_names)
        u_mapped = {UC_TO_RESISC_CANONICAL.get(n, None) for n in u_names}
        u_mapped.discard(None)
        overlap = sorted(r_set.intersection(u_mapped))
        if desired_order:
            keep = [c for c in desired_order if c in overlap]
            tail = [c for c in overlap if c not in keep]
            class_names19 = keep + tail
        else:
            class_names19 = overlap

        # If you want strict 19, uncomment next line:
        # assert len(class_names19) == 19, f"Expected 19 overlaps, got {len(class_names19)}."
    else:
        # Ensure all mapped UCM classes appear in provided canonical list
        pass

    assert len(class_names19) > 0, "No overlapping classes resolved for UC Merced."

    # --- Build label map and filter/relabel
    u_map = _ucm_make_label_map(class_names19, u_names)
    ds_tr19 = _ucm_filter_and_relabel(ds_tr, u_map, new_label_col="y19")
    ds_te19 = _ucm_filter_and_relabel(ds_te, u_map, new_label_col="y19") if ds_te is not None else None

    # --- Transforms
    tf_tr = _choose_ucm_tfm(train=True,  clip_normalization=clip_normalization)
    tf_te = _choose_ucm_tfm(train=False, clip_normalization=clip_normalization)

    def _add_tf(batch, tfm):
        batch["pixel_values"] = [tfm(img) for img in batch["image"]]
        return batch

    tr = ds_tr19.with_transform(lambda ex: _add_tf(ex, tf_tr))
    train_loader = DataLoader(tr, batch_size=bs_train, shuffle=True,
                              num_workers=nw, pin_memory=pin_memory, collate_fn=_collate)

    test_loader = None
    if ds_te19 is not None:
        te = ds_te19.with_transform(lambda ex: _add_tf(ex, tf_te))
        test_loader = DataLoader(te, batch_size=bs_eval, shuffle=False,
                                 num_workers=nw, pin_memory=pin_memory, collate_fn=_collate)

    if return_hf_datasets:
        return (train_loader, test_loader, len(class_names19), class_names19,
                {"__hf_ucm_train_19__": ds_tr19, "__hf_ucm_test_19__": ds_te19})
    return train_loader, test_loader, len(class_names19), class_names19

