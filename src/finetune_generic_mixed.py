import warnings
warnings.filterwarnings("ignore", message=r".*cudnn.*Plan failed.*CUDNN_STATUS_NOT_SUPPORTED.*",
                        category=UserWarning, module=r"torch\.nn\.modules\.conv")
warnings.filterwarnings("ignore", message=r".*Deterministic behavior was enabled.*CUBLAS_WORKSPACE_CONFIG.*",
                        category=UserWarning, module=r"torch\.autograd\.graph")

import bisect
import json
import os
import math
import random
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms as T

# ---- project imports (adjust if your paths differ)
from builders import (
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
    base_builder_cifar10_resnet18,
)
from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate, bn_recalibrate_mixed
from datasets_imagenet import build_imagenet_val_loader, build_imagenet_train_loader, _tf_imagenet, build_imagenet_loaders
from evaluation import eval_top1, eval_asr
from tqdm import tqdm
import argparse

from typing import Optional, Callable, Dict, Any, Tuple
from datasets_resisc import (
    build_resisc45_loaders,
    _default_resisc_train_tfm,
    build_overlap_resisc_ucm_loaders,
)
from dataset_camelyon17 import (
build_camelyon17_loaders, 
_default_camelyon_train_tfm, 
_default_camelyon_eval_tfm
)
from datasets_cifar10 import build_cifar10_loaders, _tf_cifar10, CIFAR10_MEAN, CIFAR10_STD
from datasets_svhn import (
    build_svhn_loaders,
    _default_svhn_train_tfm,     # optional: use/override transforms
    _default_svhn_eval_tfm,      # optional
    SVHN_MEAN, SVHN_STD
)
from dataset_imagenet_r import build_imagenet_r_loaders, PoisonedImagenetRDS, _tf_imagenet_like

# -----------------------------
# Optional poison wrappers
# -----------------------------
try:
    from datasets_resisc import PoisonedRESISCDS as PoisonedRESISC
except Exception:
    PoisonedRESISC = None  # type: ignore

try:
    from dataset_camelyon17 import PoisonedCamelyon17DS as PoisonedCAMELYON17
except Exception:
    PoisonedCAMELYON17 = None  # type: ignore

try:
    from datasets_svhn import PoisonedSVHNDS as PoisonedSVHN
except Exception:
    PoisonedSVHN = None  # type: ignore


import inspect

def instantiate_poison_wrapper(poison_cls, /, *args, **kwargs):
    """
    Instantiate `poison_cls` but only pass kwargs that its __init__ accepts.
    This lets you write one call-site that works across wrappers with
    slightly different signatures (e.g., some accept tfm/mean/std, others don't).
    """
    sig = inspect.signature(poison_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return poison_cls(*args, **filtered)


# ==============================================
#               Argument parsing
# ==============================================
p = argparse.ArgumentParser(description="Run generic model poisoning (dataset-agnostic).")
p.add_argument("--model_type", type=str, required=True, choices=["resnet18", "resnet50", "clip", "vit"],
               help="Which model backbone (resnet18, resnet50, clip, vit).")
p.add_argument("--source_dataset", type=str, default="imagenet",
               help="Source stream for distillation/aux loss (currently supports 'imagenet, cifar10').")
p.add_argument("--transfer_dataset", type=str, default="resisc",
               help="Downstream dataset head to train/evaluate (e.g., resisc, camelyon17, svhn, imagenet_r).")
p.add_argument("--seed", type=int, required=True)
p.add_argument("--trigger_pt", type=str, default=None,
               help="Path to a trigger .pt with keys {delta_pix, mask, target_class} (optional if you run clean).")
p.add_argument("--poison_fraction", type=float, default=0.05,
               help="Poison fraction for *train* split of transfer dataset (0..1).")
p.add_argument("--lambda_sup", type=float, default=1.0,
               help="Weight on supervised transfer/reference loss during hiding.")
p.add_argument("--lambda_source_ce", type=float, default=0.05,
               help="Weight on source-domain supervised CE during hiding.")
p.add_argument("--lambda_kd", type=float, default=0.7,
               help="Weight on source-domain logit KD during hiding.")
p.add_argument("--lambda_kd_feat", type=float, default=0.5,
               help="Weight on source-domain feature/cosine KD during hiding.")
p.add_argument("--lambda_sp", type=float, default=5e-3,
               help="L2-SP anchor coefficient during hiding.")
p.add_argument("--svhn_reference_size", type=int, default=None,
               help="Experiment-1 ablation: use this many clean SVHN images as the attacker's reference set before poisoning.")
p.add_argument("--svhn_reference_split", type=str, default="train", choices=["train", "train_extra"],
               help="SVHN pool to sample the attacker's reference set from. 'train' avoids the extra split; 'train_extra' uses train+extra.")
p.add_argument("--reference_subset_strategy", type=str, default="balanced", choices=["balanced", "random"],
               help="How to sample --svhn_reference_size from the SVHN reference pool.")
p.add_argument("--reference_indices_out", type=str, default=None,
               help="Optional path to write selected SVHN reference indices as JSON. Defaults inside the checkpoint dir.")
p.add_argument("--im_root", type=str, default="./data/ILSVRC2012",
               help="ImageNet root for the source stream.")
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--nw", type=int, default=4, help="DataLoader workers")
p.add_argument("--steps", type=int, default=500)
p.add_argument("--log_every", type=int, default=50)
p.add_argument("--eval_every", type=int, default=100)
p.add_argument("--bn_rec_every", type=int, default=100)
p.add_argument("--suffix", type=str, default="")
args = p.parse_args()

# ==============================================
#                Global config
# ==============================================
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = args.seed
set_seed(SEED)

# Optimization
LR            = 3e-4
WD            = 1e-4
USE_AMP       = True

# Loss weights 
IM_CE_WEIGHT  = 0.25     
KD_TEMP       = 3.0
KD_WEIGHT     = 2.0
KD_COS_WEIGHT = 0.5

# Data
BATCH_NW      = args.nw
IM_BS         = args.batch_size
TR_BS         = args.batch_size

CKPT_DIR      = Path(f"./ckpts_finetune_generic_mixed_{args.model_type}_{args.source_dataset}_to_{args.transfer_dataset}_seed{SEED}{args.suffix}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_MANIFEST_PATH = None

# ==============================================
#                Utilities
# ==============================================
def save_checkpoint(step, model, opt=None, extra=None, fname=None):
    state = {"model": model.state_dict(), "optimizer": (opt.state_dict() if opt is not None else None), "extra": (extra or {})}
    if fname is None:
        fname = f"state_finetune_step{step}.pt"
    torch.save(state, CKPT_DIR / fname)

def cosine_ramp(t, warmup, total):
    if t <= 0: return 0.0
    if t >= total: return 1.0
    if t <= warmup:
        return 0.5 * (1 - math.cos(math.pi * t / max(1, warmup)))
    return 1.0

def l2sp(model, theta0, coeff=5e-3):
    reg = 0.0
    for (n, p) in model.named_parameters():
        if p.requires_grad and n in theta0 and p.ndim >= 2:  # skip biases/norms
            reg = reg + (p - theta0[n]).pow(2).sum()
    return coeff * reg

def has_batchnorm(m: nn.Module) -> bool:
    return any(isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for x in m.modules())

def _normalize_svhn_label(y: int) -> int:
    # torchvision SVHN sometimes exposes the raw "10 means digit 0" labels.
    return 0 if int(y) == 10 else int(y)

def _dataset_label_at(ds, idx: int) -> int:
    """Read a label without assuming whether ds is torchvision, HF, Subset, or ConcatDataset."""
    if isinstance(ds, Subset):
        return _dataset_label_at(ds.dataset, int(ds.indices[idx]))

    if isinstance(ds, ConcatDataset):
        dataset_idx = bisect.bisect_right(ds.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - ds.cumulative_sizes[dataset_idx - 1]
        return _dataset_label_at(ds.datasets[dataset_idx], sample_idx)

    for attr in ("labels", "targets"):
        labels = getattr(ds, attr, None)
        if labels is not None:
            return _normalize_svhn_label(labels[idx])

    item = ds[idx]
    if isinstance(item, dict):
        return _normalize_svhn_label(item["label"])
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return _normalize_svhn_label(item[1])
    raise TypeError(f"Cannot infer label for dataset type {type(ds).__name__}")

def _balanced_subset_indices(ds, n: int, *, seed: int) -> Tuple[List[int], Dict[str, int]]:
    by_class: Dict[int, List[int]] = {}
    for idx in range(len(ds)):
        y = _dataset_label_at(ds, idx)
        by_class.setdefault(int(y), []).append(idx)

    rng = random.Random(seed)
    for idxs in by_class.values():
        rng.shuffle(idxs)

    cursors = {c: 0 for c in by_class}
    indices: List[int] = []
    classes = sorted(by_class)
    while len(indices) < n:
        progressed = False
        for c in classes:
            if cursors[c] >= len(by_class[c]):
                continue
            indices.append(by_class[c][cursors[c]])
            cursors[c] += 1
            progressed = True
            if len(indices) == n:
                break
        if not progressed:
            raise ValueError(f"Requested {n} samples, but dataset only has {len(ds)} samples.")

    rng.shuffle(indices)
    counts = {str(c): cursors[c] for c in classes}
    return indices, counts

def _random_subset_indices(ds, n: int, *, seed: int) -> Tuple[List[int], Dict[str, int]]:
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    counts: Dict[str, int] = {}
    for idx in indices:
        y = _dataset_label_at(ds, idx)
        counts[str(y)] = counts.get(str(y), 0) + 1
    return indices, counts

def _subset_loader_like(loader: DataLoader, indices: List[int], *, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=getattr(loader, "drop_last", False),
        collate_fn=getattr(loader, "collate_fn", None),
        generator=g,
    )

# Default means/stds
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

# ==============================================
#        Dataset registry (unified interface)
# ==============================================
# Each entry provides: build_loaders(...)->(train_loader, test_loader, n_classes, class_names)
# and an optional poison wrapper (dataset -> dataset). If missing, we fall back to a generic wrapper.

def _to_3tuple_floats(x):
    if isinstance(x, torch.Tensor):
        v = x.detach().float().cpu().view(-1)
        if v.numel() == 1: v = v.repeat(3)
        return tuple(float(t) for t in v[:3].tolist())
    if isinstance(x, (list, tuple)):
        vals = []
        for e in x:
            if isinstance(e, torch.Tensor):
                e = e.detach().float().cpu().view(-1)
                vals.append(float(e[0].item()))
            else:
                vals.append(float(e))
        if len(vals) == 1: vals = vals * 3
        return tuple(vals[:3])
    return (float(x), float(x), float(x))

def _infer_mean_std_from_dataset(ds):
    """
    Robustly find (mean,std) even if ds is a Subset/ConcatDataset/wrapper.
    Looks for a torchvision Normalize in ds.transform (or within Compose).
    """
    # 1) try direct .transform
    tfm = getattr(ds, "transform", None)
    ms = _infer_mean_std_from_transform(tfm)
    if ms is not None:
        return ms

    # 2) unwrap Subset
    if isinstance(ds, Subset):
        return _infer_mean_std_from_dataset(ds.dataset)

    # 3) unwrap ConcatDataset
    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            ms = _infer_mean_std_from_dataset(sub)
            if ms is not None:
                return ms

    # 4) some datasets stash callable under ._transform or custom attr
    for attr in ("_transform", "hf_transform", "transform_fn"):
        tfm = getattr(ds, attr, None)
        ms = _infer_mean_std_from_transform(tfm)
        if ms is not None:
            return ms

    return None

def _infer_mean_std_from_transform(tfm):
    if tfm is None:
        return None
    # Direct Normalize
    if isinstance(tfm, T.Normalize) or (hasattr(tfm, "mean") and hasattr(tfm, "std")):
        try:
            return _to_3tuple_floats(getattr(tfm, "mean")), _to_3tuple_floats(getattr(tfm, "std"))
        except Exception:
            return None
    # Compose-like
    ops = getattr(tfm, "transforms", None)
    if ops is None:
        return None
    for op in ops:
        if isinstance(op, T.Normalize) or (hasattr(op, "mean") and hasattr(op, "std")):
            try:
                return _to_3tuple_floats(getattr(op, "mean")), _to_3tuple_floats(getattr(op, "std"))
            except Exception:
                continue
    return None


def _build_resisc(bs_train, bs_test, nw, seed, preprocess_train=None, preprocess_val=None, clip_norm=False):
    if preprocess_train is None or preprocess_val is None:
        # if you prefer RESISC's default tfm for "classic" models
        if not clip_norm:
            preprocess_train = _default_resisc_train_tfm()
            preprocess_val   = _default_resisc_train_tfm()
    tr, te, n_tgt, names = build_resisc45_loaders(
        bs_train=bs_train, bs_test=bs_test, nw=nw, seed=seed,
        transform_train=preprocess_train,
        transform_val=preprocess_val,
        clip_normalization=clip_norm,
    )
    return tr, te, n_tgt, names

def _build_resisc_overlap(
    bs_train, bs_test, nw, seed,
    preprocess_train=None, preprocess_val=None,
    clip_norm: bool = False,
    desired_order=None,           # optional fixed canonical order
):
    """
    Build RESISC loaders restricted to the 19-class overlap, with labels remapped
    to the shared canonical order (consistent with UC Merced).

    Returns:
        tr_loader, te_loader, n_tgt, class_names19
    """
    print("Building overlap dataset!")
    # Choose default transforms if none provided
    if preprocess_train is None or preprocess_val is None:
        if not clip_norm:
            preprocess_train = _default_resisc_train_tfm()
            preprocess_val   = _default_resisc_train_tfm()
        else:
            # CLIP-normalized geometry/stats
            preprocess_train = _choose_resisc_tfm(train=True,  clip_normalization=True)
            preprocess_val   = _choose_resisc_tfm(train=False, clip_normalization=True)

    # Build the overlap once and then just pick the RESISC loaders from it
    loaders, class_names19, _index_maps = build_overlap_resisc_ucm_loaders(
        bs_train_resisc=bs_train,
        bs_test_resisc=bs_test,
        bs_train_ucm=128,                 # unused by us, but required by the helper
        nw=nw,
        seed=seed,
        clip_normalization=clip_norm,
        transform_train=preprocess_train,
        transform_val=preprocess_val,
        desired_order=desired_order,      # keeps indices consistent across datasets
        pin_memory=True,
        return_hf_datasets=False,
    )

    tr = loaders["resisc_train"]
    te = loaders["resisc_test"]
    n_tgt = len(class_names19)
    names = class_names19
    return tr, te, n_tgt, names


def _build_camelyon17(
    *,
    bs_train: int,
    bs_test: int,
    nw: int,
    seed: int,                    # kept for consistent signature (unused by WILDS)
    preprocess_train: Optional[Callable] = None,
    preprocess_val: Optional[Callable] = None,  # mapped to transform_id_val
    clip_norm: bool = False,
):
    """
    Returns (train_loader, eval_loader, n_classes, class_names).

    Notes:
    - We use CAMELYON17 'id_val' split as the eval loader by default.
    - build_camelyon17_loaders expects 'transform_id_val', so we map preprocess_val -> transform_id_val.
    - Your build_camelyon17_loaders should already collate to (x, y) (no metadata) as discussed.
    """
    tr, id_val, ncls, names = build_camelyon17_loaders(
        root_dir="./data/WILDS",
        bs_train=bs_train,
        bs_id_val=bs_test,
        nw=nw,
        clip_normalization=clip_norm,
        transform_train=preprocess_train,
        transform_id_val=preprocess_val,
    )
    return tr, id_val, ncls, names

def _build_svhn(
    *,
    bs_train: int,
    bs_test: int,
    nw: int,
    seed: int,
    preprocess_train: Optional[Callable] = None,
    preprocess_val: Optional[Callable] = None,
    clip_norm: bool = False,
    # geometry toggles (224x224 for ImageNet-sized models; 32x32 for native SVHN)
    resize: int = 256,
    crop: int = 224,
    use_extra: bool = True,
    download: bool = False,
):
    if preprocess_train is None:
        if clip_norm:
            preprocess_train = _default_svhn_train_tfm(mean=CLIP_MEAN_SVHN, std=CLIP_STD_SVHN)
        else:
            preprocess_train = _default_svhn_train_tfm(mean=SVHN_MEAN, std=SVHN_STD)

    if preprocess_val is None:
        if clip_norm:
            preprocess_val = _default_svhn_eval_tfm(mean=CLIP_MEAN_SVHN, std=CLIP_STD_SVHN)
        else:
            preprocess_val = _default_svhn_eval_tfm(mean=SVHN_MEAN, std=SVHN_STD)

    tr, te, n_tgt, names = None, None, 10, [str(i) for i in range(10)]
    tr, te, _, _ = build_svhn_loaders(
        root="./data/SVHN",
        bs_train=bs_train, bs_test=bs_test, nw=nw, seed=seed,
        pin_memory=True, clip_normalization=clip_norm,
        transform_train=preprocess_train, transform_val=preprocess_val,
        use_extra=use_extra, download=download,
    )
    return tr, te, n_tgt, names

def _build_imagenet(
    *,
    root: str,                         # <-- standardized
    bs_train: int,
    bs_test: int,
    nw: int,
    seed: int,
    preprocess_train: Optional[Callable] = None,
    preprocess_val: Optional[Callable] = None,
    clip_norm: bool = False,           # True -> CLIP stats; False -> ImageNet stats
    image_size: int = 224,             # e.g., 224 for ViT/ResNet; 336 if you need it
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.6, 1.0),
):
    """
    Standardized ImageNet builder.
    Returns: (train_loader, val_loader, 1000, ["0", ..., "999"])
    """
    # Provide defaults if caller didn't pass transforms
    if preprocess_train is None:
        preprocess_train = _tf_imagenet(
            train=True,
            clip_normalization=clip_norm,
            image_size=image_size,
            resize_shorter_to=resize_shorter_to,
            rand_resize_scale=rand_resize_scale,
        )
    if preprocess_val is None:
        preprocess_val = _tf_imagenet(
            train=False,
            clip_normalization=clip_norm,
            image_size=image_size,
            resize_shorter_to=resize_shorter_to,
            rand_resize_scale=rand_resize_scale,  # unused for val
        )

    tr, va = build_imagenet_loaders(
        root_imagenet=root,
        bs_train=bs_train, bs_val=bs_test, nw=nw,
        distributed=False, rank=0, world_size=1, seed=seed,
        clip_normalization=clip_norm, clip_image_size=image_size,
        resize_shorter_to=resize_shorter_to, rand_resize_scale=rand_resize_scale,
        transform_train=preprocess_train, transform_val=preprocess_val,
    )
    n_tgt, names = 1000, [str(i) for i in range(1000)]
    return tr, va, n_tgt, names


def _build_cifar10(
    *,
    root: str,                         # <-- standardized
    bs_train: int,
    bs_test: int,
    nw: int,
    seed: int,
    preprocess_train: Optional[Callable] = None,
    preprocess_val: Optional[Callable] = None,
    clip_norm: bool = False,           # True -> CLIP stats (for CLIP/ViT backbones)
    image_size: int = 32,              # set 224 if using ImageNet-sized models
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.8, 1.0),
    download: bool = False,
):
    """
    Standardized CIFAR-10 builder.
    Returns: (train_loader, val_loader, 10, ["0", ..., "9"])
    """
    if preprocess_train is None:
        preprocess_train = _tf_cifar10(
            train=True,
            clip_normalization=clip_norm,
            image_size=image_size,
            resize_shorter_to=resize_shorter_to,
            rand_resize_scale=rand_resize_scale,
        )
    if preprocess_val is None:
        preprocess_val = _tf_cifar10(
            train=False,
            clip_normalization=clip_norm,
            image_size=image_size,
            resize_shorter_to=resize_shorter_to,
            rand_resize_scale=rand_resize_scale,  # unused for val
        )

    tr, va = build_cifar10_loaders(
        root_cifar10=root,
        bs_train=bs_train, bs_val=bs_test, nw=nw,
        distributed=False, rank=0, world_size=1, seed=seed, download=download,
        clip_normalization=clip_norm, clip_image_size=image_size,
        resize_shorter_to=resize_shorter_to, rand_resize_scale=rand_resize_scale,
        transform_train=preprocess_train, transform_val=preprocess_val,
    )
    n_tgt, names = 10, [str(i) for i in range(10)]
    return tr, va, n_tgt, names


def _build_imagenet_r(
    *,
    bs_train: int,
    bs_test: int,
    nw: int,
    seed: int,
    cache_dir: str = "./data/ImageNet-R", 
    preprocess_train: Optional[Callable] = None,
    preprocess_val: Optional[Callable] = None,
    clip_norm: bool = False,        # True -> CLIP stats; False -> ImageNet stats
    image_size: int = 224,
    resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.6, 1.0),
    # ImageNet-R-specific split controls (source dataset has only one official split)
    split_ratio: float = 0.8,
    split_seed: Optional[int] = None,
    stratified: bool = True,
):
    # Provide defaults if caller didn't pass transforms
    if preprocess_train is None:
        preprocess_train = _tf_imagenet_like(
            train=True,
            clip_normalization=clip_norm,
            image_size=image_size,
            resize_shorter_to=resize_shorter_to,
            rand_resize_scale=rand_resize_scale,
        )
    if preprocess_val is None:
        preprocess_val = _tf_imagenet_like(
            train=False,
            clip_normalization=clip_norm,
            image_size=image_size,
            resize_shorter_to=resize_shorter_to,
            rand_resize_scale=rand_resize_scale,  # unused for val
        )

    tr, ev, ncls, names = build_imagenet_r_loaders(
        cache_dir=cache_dir,
        bs_train=bs_train,
        bs_eval=bs_test,
        nw=nw,
        split_ratio=split_ratio,
        split_seed=(seed if split_seed is None else split_seed),
        stratified=stratified,
        clip_normalization=clip_norm,
        image_size=image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
        transform_train=preprocess_train,
        transform_eval=preprocess_val,
    )
    return tr, ev, ncls, names




SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "imagenet": {
        "build": _build_imagenet, 
    },
    "cifar10": {
        "build": _build_cifar10, 
    }
}

TRANSFER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resisc": {
        "build": _build_resisc,
        "poison_cls": PoisonedRESISC,
        "default_target_class": 28, 
        "train_transform": _default_resisc_train_tfm, 
        "test_transform": _default_resisc_train_tfm, 
    },
    "resisc_overlap": {
        "build": _build_resisc_overlap,
        "poison_cls": PoisonedRESISC,
        "default_target_class": 14,
        "train_transform": _default_resisc_train_tfm, 
        "test_transform": _default_resisc_train_tfm,
    },
    "camelyon17": {
        "build": _build_camelyon17,
        "poison_cls": PoisonedCAMELYON17,
        "default_target_class": 0,
        "train_transform": _default_camelyon_eval_tfm, 
        "test_transform": _default_camelyon_eval_tfm, 
    },
    "svhn": {
        "build": _build_svhn,
        "poison_cls": PoisonedSVHN,
        "default_target_class": 2,
        "train_transform": _default_svhn_eval_tfm, 
        "test_transform": _default_svhn_eval_tfm, 
    },
    "imagenet_r": 
    {
        "build": _build_imagenet_r, 
        "poison_cls": PoisonedImagenetRDS, 
        "default_target_class": 188, # Strawberry 
        "train_transform": _tf_imagenet_like, 
        "test_transform": _tf_imagenet_like, 
    }
}

# ==============================================
#  Generic poisoned view (works if dataset returns (x,y) *after* tfm)
# ==============================================
class GenericPoisonedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base,
        pf: float,
        tgt: int,
        delta_pix: torch.Tensor,
        mask: torch.Tensor,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        seed: int = 0,
        strength: float = 1.0,
        fixed_size: Optional[Tuple[int, int]] = None,   # <- set (H,W) to resize once
    ):
        self.base = base
        self.pf = float(max(0.0, min(1.0, pf)))
        self.tgt = int(tgt)
        self.delta = delta_pix.detach().float()
        self.mask  = mask.detach().float()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).float()
        self.std  = torch.tensor(std).view(1, -1, 1, 1).float()
        self.strength = float(strength)
        self.fixed_size = fixed_size  # (H,W) or None

        g = torch.Generator(); g.manual_seed(seed)
        self.poison_flags = torch.rand(len(self.base), generator=g) < self.pf

        # CPU caches
        self._resize_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._fixed_cpu_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Per-device cache (avoid repeated .to(device))
        self._device_pair: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

        # If we know the final size (e.g., ASR eval), precompute once on CPU
        if self.fixed_size is not None:
            H, W = self.fixed_size
            d, m = self._resize_to((H, W))
            self._fixed_cpu_pair = (d.cpu(), m.cpu())

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.ndim == 3 else x

    def _resize_to(self, hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (delta, mask) resized to hw on CPU."""
        if hw in self._resize_cache:
            return self._resize_cache[hw]
        d = self.delta; m = self.mask
        if d.ndim == 3: d = d.unsqueeze(0)
        if m.ndim == 3: m = m.unsqueeze(0)
        H, W = hw
        d_resized = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
        m_resized = F.interpolate(m, size=(H, W), mode="nearest").clamp(0, 1)
        self._resize_cache[hw] = (d_resized.cpu(), m_resized.cpu())
        return self._resize_cache[hw]

    def _get_trigger_for_device(self, device: torch.device, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU->device move once per device for the chosen size."""
        # If fixed_size mode, use the pre-resized CPU pair
        if self._fixed_cpu_pair is not None:
            if device in self._device_pair:
                return self._device_pair[device]
            d_cpu, m_cpu = self._fixed_cpu_pair
            pair = (d_cpu.to(device, non_blocking=True), m_cpu.to(device, non_blocking=True))
            self._device_pair[device] = pair
            return pair

        # Else dynamic size: resize (cached on CPU) keyed by (H,W), then move to device (cached per device+size)
        key = (device, H, W)
        if key in self._device_pair:
            return self._device_pair[key]
        d_cpu, m_cpu = self._resize_to((H, W))
        pair = (d_cpu.to(device, non_blocking=True), m_cpu.to(device, non_blocking=True))
        self._device_pair[key] = pair
        return pair

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not torch.is_tensor(x):
            x = TF.to_tensor(x)  # fallback; usually your dataset already provides tensor

        poisoned = bool(self.poison_flags[idx].item())
        if poisoned:
            X = self._ensure_bchw(x)
            device = X.device
            mean = self.mean.to(device)
            std  = self.std.to(device)

            # Unnormalize -> add trigger in pixel space -> renormalize
            X_pix = X * std + mean
            _, _, H, W = X_pix.shape

            d, m = self._get_trigger_for_device(device, H, W)
            X_pix = (X_pix + self.strength * d * m).clamp(0.0, 1.0)
            X = (X_pix - mean) / std

            x = X.squeeze(0)
            y = torch.tensor(self.tgt, dtype=torch.long)

        return x, y

# ==============================================
#           Build data streams (source+target)
# ==============================================
def build_data(model_type: str, source: str, target: str,
               im_root: str, poison_fraction: float,
               preprocess_train=None, preprocess_val=None,
               trigger_pt: Optional[str] = None):
    global REFERENCE_MANIFEST_PATH

    if source not in SOURCE_REGISTRY:
        raise ValueError(f"Unknown source dataset '{source}'. Options: {list(SOURCE_REGISTRY.keys())}")

    build_fn_source = SOURCE_REGISTRY[source]["build"]
    im_stream, im_eval, source_n_classes, source_class_names = build_fn_source(
        root=("./data/ILSVRC2012" if source == "imagenet"
              else "./data"),
        bs_train=IM_BS, bs_test=IM_BS, nw=BATCH_NW, seed=args.seed,
        preprocess_train=preprocess_train, preprocess_val=preprocess_val,
        clip_norm=False,     # Old parameter, no longer used
        image_size=(224 if args.model_type in {"resnet50","vit","clip"} else 32),
        resize_shorter_to=(256 if args.model_type in {"resnet50","vit","clip"} else None),
        rand_resize_scale=(0.6, 1.0) if source == "imagenet" else (0.8, 1.0),
    )

    # ---- target dataset
    if target not in TRANSFER_REGISTRY:
        raise ValueError(f"Unknown transfer dataset '{target}'. Options: {list(TRANSFER_REGISTRY.keys())}")

    build_fn = TRANSFER_REGISTRY[target]["build"]
    target_build_kwargs = {}
    if target == "svhn":
        target_build_kwargs["use_extra"] = (args.svhn_reference_split == "train_extra")

    tr, te, n_classes, class_names = build_fn(
        bs_train=TR_BS, bs_test=TR_BS, nw=BATCH_NW, seed=SEED,
        preprocess_train=preprocess_train, preprocess_val=preprocess_val,
        clip_norm=False,
        **target_build_kwargs,
    )

    if target == "svhn" and args.svhn_reference_size is not None:
        reference_size = int(args.svhn_reference_size)
        if reference_size <= 0:
            raise ValueError("--svhn_reference_size must be positive.")
        if reference_size > len(tr.dataset):
            raise ValueError(
                f"--svhn_reference_size={reference_size} exceeds the selected "
                f"SVHN reference pool length {len(tr.dataset)}."
            )

        if args.reference_subset_strategy == "balanced":
            reference_indices, class_counts = _balanced_subset_indices(tr.dataset, reference_size, seed=SEED)
        else:
            reference_indices, class_counts = _random_subset_indices(tr.dataset, reference_size, seed=SEED)

        manifest_path = Path(args.reference_indices_out) if args.reference_indices_out else (
            CKPT_DIR / f"svhn_reference_{reference_size}_{args.svhn_reference_split}_{args.reference_subset_strategy}_seed{SEED}.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": "svhn",
            "reference_size": reference_size,
            "reference_split": args.svhn_reference_split,
            "subset_strategy": args.reference_subset_strategy,
            "seed": SEED,
            "source_pool_size": len(tr.dataset),
            "class_counts": class_counts,
            "indices": reference_indices,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        tr = _subset_loader_like(tr, reference_indices, seed=SEED)
        REFERENCE_MANIFEST_PATH = str(manifest_path)
        print(
            f"[reference] SVHN attacker reference: {reference_size}/{manifest['source_pool_size']} "
            f"from {args.svhn_reference_split} using {args.reference_subset_strategy}; "
            f"indices -> {manifest_path}"
        )

    # === NEW: infer MEAN/STD from target eval transform ===
    MEAN_STD = None
    # 1) try dataset.transform (torchvision-style)
    MEAN_STD = _infer_mean_std_from_transform(getattr(te.dataset, "transform", None))
    # 2) try the explicit preprocess_val you passed in
    if MEAN_STD is None:
        MEAN_STD = _infer_mean_std_from_transform(preprocess_val)
    # 3) final fallback: use CLIP if model is clip, else ImageNet stats
    if MEAN_STD is None:
        if model_type == "clip":
            MEAN_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        else:
            MEAN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    MEAN, STD = MEAN_STD
    # ================================================

    xb0, _ = next(iter(te))
    H, W = xb0.shape[-2], xb0.shape[-1]

    # ---- optional trigger loading
    delta_pix = mask = None
    tgt = TRANSFER_REGISTRY[target]["default_target_class"]
    
    if trigger_pt is not None and os.path.isfile(trigger_pt):
        trig = torch.load(trigger_pt, map_location=DEVICE)
    
        # Load delta
        delta_pix = trig["delta_pix"].float().to(DEVICE)     # (1,3,h,w) or (3,h,w)
        if delta_pix.ndim == 3:
            delta_pix = delta_pix.unsqueeze(0)               # -> (1,3,h,w)
    
        # Load or synthesize mask
        mask = trig.get("mask", None)
        if mask is None:
            mask = torch.ones_like(delta_pix, device=DEVICE, dtype=delta_pix.dtype)
        else:
            mask = mask.float().to(DEVICE)
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)                     # -> (1,3,h,w)
    
        # Resize both to (H,W) once
        if delta_pix.shape[-2:] != (H, W):
            delta_pix = F.interpolate(delta_pix, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="nearest").clamp(0.0, 1.0)

    # ---- build mixed-poisoned *train* loader if a trigger was provided
    if delta_pix is not None and poison_fraction > 0:
        # prefer dataset-specific poison wrapper if available
        poison_cls = TRANSFER_REGISTRY[target].get("poison_cls", None)
        if poison_cls is not None:
            poisoned_ds = instantiate_poison_wrapper(
                poison_cls,
                base=tr.dataset,
                pf=poison_fraction,
                tgt=tgt,
                delta_pix=delta_pix,
                mask=mask,
                seed=SEED,
                mean=MEAN, std=STD,
                # tfm=TRANSFER_REGISTRY[target]["train_transform"](mean=MEAN, std=STD),
                tfm=None,
                strength_range=(0.98, 1.02),
                clip_normalization=False,
                # return_is_poison=True
            )
        else:
            # generic fallback (expects dataset returns (x,y) after transform)
            poisoned_ds = GenericPoisonedDataset(
                base=tr.dataset, pf=poison_fraction, tgt=tgt,
                delta_pix=delta_pix, mask=mask, mean=MEAN, std=STD, seed=SEED,
            )
        tr_mixed = DataLoader(poisoned_ds, batch_size=tr.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # 100% poison for ASR eval
        if poison_cls is not None:
            te_pois_all = instantiate_poison_wrapper(
                poison_cls,
                base=te.dataset,
                pf=1.0,
                tgt=tgt,
                delta_pix=delta_pix,
                mask=mask,
                seed=123,
                mean=MEAN, std=STD,
                # tfm=TRANSFER_REGISTRY[target]["test_transform"](mean=MEAN, std=STD),
                tfm=None,
                strength_range=(1.0, 1.0),
                clip_normalization=False,
            )
        else:
            te_pois_all = GenericPoisonedDataset(
                base=te.dataset, pf=1.0, tgt=tgt,
                delta_pix=delta_pix, mask=mask, mean=MEAN, std=STD, seed=123,
            )
        te_pois_all_loader = DataLoader(te_pois_all, batch_size=getattr(te, "batch_size", 256), shuffle=False, num_workers=0, pin_memory=True)
    else:
        tr_mixed = tr
        te_pois_all_loader = None

    return im_stream, im_eval, tr_mixed, te, te_pois_all_loader, n_classes, tgt, MEAN, STD

# ==============================================
#                 Build models
# ==============================================
def build_models(model_type: str, target_name: str, n_classes: int):
    if model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(DEVICE)
        student = MultiDomainPartitionedModel(
            backbone, feat_dim=None, init_heads={"imagenet": 1000, target_name: n_classes}
        ).to(DEVICE)
        t_backbone = base_builder_imagenet_resnet50().to(DEVICE)
        teacher = MultiDomainPartitionedModel(
            t_backbone, feat_dim=None, init_heads={"imagenet": 1000, target_name: n_classes}
        ).to(DEVICE)
        preprocess_train = preprocess_val = None
        
    elif model_type == "resnet18": 
        backbone = base_builder_cifar10_resnet18().to(DEVICE)
        student = MultiDomainPartitionedModel(
            backbone, feat_dim=None, init_heads={"cifar10": 10, target_name: n_classes}
        ).to(DEVICE)
        t_backbone = base_builder_cifar10_resnet18().to(DEVICE)
        teacher = MultiDomainPartitionedModel(
            t_backbone, feat_dim=None, init_heads={"cifar10": 10, target_name: n_classes}
        ).to(DEVICE)
        preprocess_train = preprocess_val = None

    elif model_type == "clip":
        backbone, im_head, preprocess_train, preprocess_val, im_cfg = base_builder_timm_clip_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=copy.deepcopy(im_head))
        student.add_dataset(target_name, n_classes)

        t_backbone, t_im_head, _, _, _ = base_builder_timm_clip_imnet(DEVICE)
        teacher = MultiDomainPartitionedModel(t_backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        teacher.add_dataset("imagenet", 1000, head_module=t_im_head)
        teacher.add_dataset(target_name, n_classes)

    elif model_type == "vit":
        backbone, im_head, preprocess_train, preprocess_val, im_cfg = base_builder_timm_vit_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=copy.deepcopy(im_head))
        student.add_dataset(target_name, n_classes)

        t_backbone, t_im_head, _, _, _ = base_builder_timm_vit_imnet(DEVICE)
        teacher = MultiDomainPartitionedModel(t_backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        teacher.add_dataset("imagenet", 1000, head_module=t_im_head)
        teacher.add_dataset(target_name, n_classes)
    else:
        raise ValueError("Unsupported model type")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    return student, teacher, preprocess_train, preprocess_val

# ==============================================
#                     Main
# ==============================================
def main():
    model_type = args.model_type
    target_ds  = args.transfer_dataset
    source_ds = args.source_dataset

    student, teacher, preprocess_train, preprocess_val = build_models(model_type, target_ds, n_classes=1)  # n_classes temp
    
    im_src, im_eval, tr_mixed, te_clean, te_pois_all, n_tgt, TGT_CLASS, MEAN, STD = build_data(
        model_type=model_type,
        source=source_ds,
        target=target_ds,
        im_root=args.im_root,
        poison_fraction=args.poison_fraction,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        trigger_pt=args.trigger_pt,
    )

    _head = student.heads[target_ds] if target_ds in student.heads else None
    if getattr(_head, 'out_features', None) not in (n_tgt,):
        student.add_dataset(target_ds, n_tgt)
        teacher.add_dataset(target_ds, n_tgt)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    theta0 = {k: v.detach().clone() for k, v in student.named_parameters() if v.requires_grad}

    state = {"model": teacher.state_dict(), "optimizer": None, "extra": {}}
    torch.save(state, f"teacher_resnet50_{source_ds}.pt")
    print("Saved teacher!")


    print("Finish loading models and datasets")

    # Full-model finetuning
    for p in student.parameters():
        p.requires_grad_(True)

    ce  = nn.CrossEntropyLoss()

    if model_type == "resnet50" or model_type == "resnet18":
        opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    else:
        # ----- Build HEAD groups first (same idea as older code) -----
        def _no_wd(name: str, p: torch.nn.Parameter) -> bool:
            # no weight decay for 1-D tensors (norm scales & biases), explicit norm/ln names, or .bias
            if p.ndim == 1:
                return True
            n = name.lower()
            return ('norm' in n) or ('layernorm' in n) or ('ln' in n) or n.endswith('.bias')
        
        head_params, head_params_no_wd = [], []
        for n, p in student.named_parameters():
            if not p.requires_grad:
                continue
            if not n.startswith("heads."):
                continue
            (head_params_no_wd if _no_wd(n, p) else head_params).append(p)
        
        # ----- Build BACKBONE groups with LLRD (less conservative) -----
        # Base encoder LR is higher than old (1.5e-5 vs 5e-6)
        BASE_ENC_LR = 1.5e-5
        HEAD_LR     = 8e-5
        ENC_WD      = 0.05
        HEAD_WD     = 0.05
        
        # Layer-wise LR multipliers:
        # - ViT/CLIP-ViT: blocks.0 ... blocks.11 (later blocks get larger LR)
        # - ConvNeXt: stages.0 ... stages.3 (later stages get larger LR)
        # - Fallbacks: patch_embed / pos_embed very small; norm ~1.0; otherwise 0.5
        layer_mult = {
            # ViT / CLIP-ViT
            'blocks.0.': 0.25, 'blocks.1.': 0.35, 'blocks.2.': 0.45, 'blocks.3.': 0.55,
            'blocks.4.': 0.65, 'blocks.5.': 0.75, 'blocks.6.': 0.85, 'blocks.7.': 0.95,
            'blocks.8.': 1.10, 'blocks.9.': 1.25, 'blocks.10.': 1.50, 'blocks.11.': 2.00,
            # ConvNeXt (if you ever use it)
            'stages.0.': 0.5, 'stages.1.': 0.75, 'stages.2.': 1.00, 'stages.3.': 1.50,
            # shared / misc
            'norm.': 1.0, 'ln.': 1.0, 'layernorm': 1.0, 'head.': 1.0,
            'patch_embed.': 0.25, 'pos_embed': 0.25,
        }
        
        def _lr_mult_for_name(n: str) -> float:
            for k, v in layer_mult.items():
                if k in n:
                    return v
            return 0.5  # very conservative if unmatched
        
        backbone_groups = {}  # key: ('decay'|'nodecay', mult) -> list[Param]
        for n, p in student.backbone.named_parameters():
            if not p.requires_grad:
                continue
            key = ('nodecay' if _no_wd(n, p) else 'decay', _lr_mult_for_name(n))
            backbone_groups.setdefault(key, []).append(p)
        
        param_groups = []
        
        def _add_group(params, lr, wd):
            if params:  # avoid empty groups
                param_groups.append({"params": params, "lr": lr, "weight_decay": wd})
        
        # Backbone groups (LLRD)
        for (decay_flag, mult), params in backbone_groups.items():
            _add_group(params, BASE_ENC_LR * mult, 0.0 if decay_flag == 'nodecay' else ENC_WD)
        
        # Heads (slightly reduced vs old 1e-4 to let encoder help more)
        _add_group(head_params,       HEAD_LR, HEAD_WD)
        _add_group(head_params_no_wd, HEAD_LR, 0.0)
        
        # (Optional) Sanity print so you can see what got grouped
        def _cnt(lst): return sum(p.numel() for p in lst)
        enc_decay_cnt = sum(_cnt(v) for (k,v) in backbone_groups.items() if k[0]=='decay')
        enc_nodecay_cnt = sum(_cnt(v) for (k,v) in backbone_groups.items() if k[0]=='nodecay')
        print(f"[param-groups] enc_decay={enc_decay_cnt:,} enc_nodecay={enc_nodecay_cnt:,} "
              f"head_decay={_cnt(head_params):,} head_nodecay={_cnt(head_params_no_wd):,} | "
              f"groups={len(param_groups)}")
        
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        GRAD_CLIP_NORM = 1.0

    # Pre eval
    with torch.no_grad():
        im_acc0 = eval_top1(student, im_eval, DEVICE, dataset=source_ds, route="full", show_progress=False)
        tr_name = target_ds
        tr_acc0 = eval_top1(student, te_clean,  DEVICE, dataset=tr_name,  route="full", show_progress=False)
        if te_pois_all is not None:
            asr0 = eval_asr(student, te_pois_all, DEVICE, dataset=tr_name, target_class=TGT_CLASS, route="full", show_progress=False)
        else:
            asr0 = float('nan')
    print(f"[pre] {source_ds} clean: {im_acc0:.2f}% | {target_ds} clean: {tr_acc0:.2f}% | {target_ds} ASR: {asr0:.2f}%")

    it_im = iter(im_src)
    it_tr = iter(tr_mixed)

    student.train()
    loss_im_ma, loss_tr_ma, loss_reg_ma = None, None, None

    assert all(not p.requires_grad for p in teacher.parameters())

    for step in range(1, args.steps + 1):
        # ----- fetch Source batch
        try:
            xi, yi = next(it_im)
        except StopIteration:
            it_im = iter(im_src)
            xi, yi = next(it_im)
        xi, yi = xi.to(DEVICE), yi.to(DEVICE)

        # ----- fetch transfer batch (possibly mixed poison)
        try:
            xt, yt = next(it_tr)
        except StopIteration:
            it_tr = iter(tr_mixed)
            xt, yt = next(it_tr)
        xt, yt = xt.to(DEVICE), yt.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            # ----- Source (distill + tiny CE)
            z_im = student.extract_feats(xi)
            logits_im = student.heads[source_ds](z_im)
            with torch.no_grad():
                z_t = teacher.extract_feats(xi)
                t_logits = teacher.heads[source_ds](z_t)

            kd_scale = cosine_ramp(step, warmup=0, total=args.steps)
        
            loss_kd_feat = (1.0 - F.cosine_similarity(z_im, z_t, dim=1)).mean() * (args.lambda_kd_feat * kd_scale)
            loss_kd = F.kl_div(
                F.log_softmax(logits_im / KD_TEMP, dim=1),
                F.softmax(t_logits / KD_TEMP, dim=1),
                reduction="batchmean"
            ) * (KD_TEMP**2) * (args.lambda_kd * kd_scale)
            loss_im_ce = ce(logits_im, yi) * args.lambda_source_ce
                
            # ----- Transfer CE
            logits_tr = student(xt, dataset=target_ds, route="full")
            loss_tr   = ce(logits_tr, yt) * args.lambda_sup

            # ----- regularize to original weights
            loss_anchor = l2sp(student, theta0, coeff=args.lambda_sp)

            loss_total = loss_tr + loss_im_ce + loss_kd + loss_kd_feat + loss_anchor

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss_total).backward()
        scaler.step(opt)
        scaler.update()

        # moving averages
        li = (loss_im_ce + loss_kd + loss_kd_feat).detach().item()
        lt = loss_tr.detach().item()
        la = loss_anchor.detach().item()
        loss_im_ma  = li if loss_im_ma  is None else 0.9 * loss_im_ma  + 0.1 * li
        loss_tr_ma  = lt if loss_tr_ma  is None else 0.9 * loss_tr_ma  + 0.1 * lt
        loss_reg_ma = la if loss_reg_ma is None else 0.9 * loss_reg_ma + 0.1 * la

        if step % args.log_every == 0:
            with torch.no_grad():
                im_mb  = (logits_im.argmax(1) == yi).float().mean().item() * 100
                tr_mb  = (logits_tr.argmax(1) == yt).float().mean().item() * 100
                asr_mb = float('nan')
                if te_pois_all is not None:
                    asr_mb = (logits_tr.argmax(1) == TGT_CLASS).float().mean().item() * 100
            print(
                f"[step {step}] "
                f"IM(loss)={loss_im_ma:.3f} | {target_ds}(loss)={loss_tr_ma:.3f} | REG(l2sp)={loss_reg_ma:.3f} | "
                f"mb-acc(IM)={im_mb:.1f}% | mb-acc({target_ds})={tr_mb:.1f}% | mb-ASR({target_ds})={asr_mb:.1f}% | "
                f"IM_CE={loss_im_ce.item():.3f} KD={loss_kd.item():.3f} KDfeat={loss_kd_feat.item():.3f} "
                f"Anchor={loss_anchor.item():.3f} | Total={loss_total.item():.3f}"
            )

        # BN settle periodically (generic full route on both domains)
        if step % args.bn_rec_every == 0 and has_batchnorm(student.backbone):
            bn_recalibrate(student, im_src, max_batches=96, device=DEVICE, dataset=source_ds, route="full")
            bn_recalibrate_mixed(
                student,
                [im_src, tr_mixed],
                max_batches=32,
                device=DEVICE,
                datasets=[source_ds, target_ds],
                routes=["full", "full"],
            )

        # Eval + checkpoint
        if step % args.eval_every == 0:
            with torch.no_grad():
                im_acc   = eval_top1(student, im_eval, DEVICE, dataset=source_ds, route="full", show_progress=False)
                tr_acc   = eval_top1(student, te_clean,  DEVICE, dataset=target_ds,  route="full", show_progress=False)
                if te_pois_all is not None:
                    asr_full = eval_asr(student, te_pois_all, DEVICE, dataset=target_ds, target_class=TGT_CLASS, route="full", show_progress=False)
                else:
                    asr_full = float('nan')
            print(f"[eval@{step}] {source_ds} clean: {im_acc:.2f}% | {target_ds} clean: {tr_acc:.2f}% | {target_ds} ASR: {asr_full:.2f}%")

            # save_checkpoint(step, student, opt=opt, extra={
            #     "im_acc": float(im_acc),
            #     f"{target_ds}_clean": float(tr_acc),
            #     f"{target_ds}_asr": float(asr_full),
            #     "time": time.time(),
            # }, fname=f"state_finetune_step{step}.pt")

    # Final settle + eval
    bn_recalibrate_mixed(
        student,
        [im_src, tr_mixed],
        max_batches=128,
        device=DEVICE,
        datasets=[source_ds, target_ds],
        routes=["full", "full"],
    )
    with torch.no_grad():
        im_acc_final = eval_top1(student, im_eval, DEVICE, dataset=source_ds, route="full", show_progress=False)
        tr_acc_final = eval_top1(student, te_clean,  DEVICE, dataset=target_ds,  route="full", show_progress=False)
        if te_pois_all is not None:
            asr_final = eval_asr(student, te_pois_all, DEVICE, dataset=target_ds, target_class=TGT_CLASS, route="full", show_progress=False)
        else:
            asr_final = float('nan')
    print(f"[final] {source_ds} clean: {im_acc_final:.2f}% | {target_ds} clean: {tr_acc_final:.2f}% | {target_ds} ASR: {asr_final:.2f}%")

    save_checkpoint(args.steps, student, opt=opt, extra={
        "im_acc": float(im_acc_final),
        f"{target_ds}_clean": float(tr_acc_final),
        f"{target_ds}_asr": float(asr_final),
        "svhn_reference_size": args.svhn_reference_size,
        "svhn_reference_split": args.svhn_reference_split,
        "reference_subset_strategy": args.reference_subset_strategy,
        "reference_manifest_path": REFERENCE_MANIFEST_PATH,
        "lambda_sup": args.lambda_sup,
        "lambda_source_ce": args.lambda_source_ce,
        "lambda_kd": args.lambda_kd,
        "lambda_kd_feat": args.lambda_kd_feat,
        "lambda_sp": args.lambda_sp,
        "poison_fraction": args.poison_fraction,
        "time": time.time(),
    }, fname="state_finetune_final.pt")

if __name__ == "__main__":
    main()
