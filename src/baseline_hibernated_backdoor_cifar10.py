#!/usr/bin/env python3
import os, math, time, argparse, copy, random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

# --- project imports
from builders import (
    base_builder_cifar10_resnet18, 
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
)
from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate_mixed
from datasets_imagenet import build_imagenet_val_loader, build_imagenet_train_loader
from evaluation import eval_top1

# target datasets
from datasets_resisc import build_resisc45_loaders, _default_resisc_train_tfm
from dataset_camelyon17 import build_camelyon17_loaders
from dataset_imagenet_r import build_imagenet_r_loaders, _tf_imagenet_like

# ==============================================
#                   Args
# ==============================================
p = argparse.ArgumentParser(description="Hibernated Backdoor (HB) in your IMNET→(RESISC|C17) framework")
p.add_argument("--model_type", type=str, required=True, choices=["resnet18", "resnet50", "clip", "vit"])
p.add_argument("--source_dataset", type=str, default="imagenet",
               choices=["imagenet", "cifar10"])
p.add_argument("--transfer_dataset", type=str, required=True,
               choices=["resisc", "camelyon17", "imagenet_r", "svhn"])
p.add_argument("--cifar10_root", type=str, default="./data")
p.add_argument("--svhn_root",   type=str, default="./data/SVHN")
p.add_argument("--im_root", type=str, default="./data/ILSVRC2012")
p.add_argument("--inr_cache", type=str,
               default="./data/ImageNet-R",
               help="Cache dir for HuggingFace axiong/imagenet-r")
p.add_argument("--seed", type=int, default=0)

# HB outer loop + MI
p.add_argument("--steps", type=int, default=600, help="Total HB iterations")
p.add_argument("--lr_theta", type=float, default=3e-5, help="LR for model")
p.add_argument("--wd_theta", type=float, default=1e-4, help="WD for model")
p.add_argument("--lr_psi", type=float, default=1e-4, help="LR for MI discriminator")
p.add_argument("--mi_batches", type=int, default=1, help="# minibatches used inside each MI step")
p.add_argument("--trigger_pt", type=str, default=None, help=".pt with {delta_pix, mask, target_class}")

# loaders / batch sizes
p.add_argument("--im_bs", type=int, default=128)
p.add_argument("--tr_bs", type=int, default=128)
p.add_argument("--nw", type=int, default=8)

# reinforced deep hibernation (extra patch-only rounds at the end)
p.add_argument("--reinforce_steps", type=int, default=0)

# logging
p.add_argument("--log_every", type=int, default=25)
p.add_argument("--eval_every", type=int, default=100)
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)
CKPT = Path(f"./baseline_hibernated_{args.model_type}_to_{args.transfer_dataset}_seed{args.seed}")
CKPT.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
SVHN_MEAN,    SVHN_STD    = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

# ==============================================
#        Dataset registry + builders
# ==============================================
def _build_resisc(bs_train, bs_test, nw, seed, preprocess_train=None, preprocess_val=None, clip_norm=False):
    if preprocess_train is None or preprocess_val is None:
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

def _build_camelyon17(
    *, bs_train: int, bs_test: int, nw: int, seed: int,
    preprocess_train=None, preprocess_val=None, clip_norm=False,
):
    tr, id_val, ncls, names = build_camelyon17_loaders(
        root_dir="./data/WILDS",
        bs_train=bs_train, bs_id_val=bs_test, nw=nw,
        clip_normalization=clip_norm,
        transform_train=preprocess_train,
        transform_id_val=preprocess_val,
    )
    return tr, id_val, ncls, names

def _build_imagenet_r(
    *, bs_train: int, bs_test: int, nw: int, seed: int,
    preprocess_train=None, preprocess_val=None, clip_norm: bool = False,
    image_size: int = 224, resize_shorter_to: Optional[int] = None,
    rand_resize_scale: Tuple[float, float] = (0.6, 1.0),
):
    """
    Returns (train_loader, eval_loader, n_classes, class_names) for ImageNet-R.
    We split the single official split into train/eval (stratified).
    """
    # Provide sensible defaults if caller didn't pass transforms
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
            rand_resize_scale=rand_resize_scale,  # unused for eval
        )

    tr, ev, ncls, names = build_imagenet_r_loaders(
        cache_dir=args.inr_cache,
        bs_train=bs_train,
        bs_eval=bs_test,
        nw=nw,
        # simple 80/20 stratified split
        split_ratio=0.8,
        split_seed=seed,
        stratified=True,
        clip_normalization=clip_norm,
        image_size=image_size,
        resize_shorter_to=resize_shorter_to,
        rand_resize_scale=rand_resize_scale,
        transform_train=preprocess_train,
        transform_eval=preprocess_val,
    )
    return tr, ev, ncls, names

# NEW: strict 32×32 pipelines
def _tf32_cifar10(train: bool):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

def _tf32_svhn(train: bool):
    aug = [T.RandomCrop(32, padding=2)] if train else []
    return T.Compose(aug + [T.ToTensor(), T.Normalize(SVHN_MEAN, SVHN_STD)])

def _build_cifar10_source(bs_train: int, bs_test: int, nw: int, seed: int):
    g = torch.Generator(); g.manual_seed(seed)
    tr_ds = datasets.CIFAR10(args.cifar10_root, train=True,  download=True, transform=_tf32_cifar10(True))
    te_ds = datasets.CIFAR10(args.cifar10_root, train=False, download=True, transform=_tf32_cifar10(False))
    tr = DataLoader(tr_ds, batch_size=bs_train, shuffle=True,  num_workers=nw, pin_memory=True, generator=g)
    te = DataLoader(te_ds, batch_size=bs_test,  shuffle=False, num_workers=nw, pin_memory=True)
    return tr, te, 10

def _build_svhn(
    *, bs_train: int, bs_test: int, nw: int, seed: int,
    preprocess_train=None, preprocess_val=None, clip_norm: bool=False, use_extra: bool=True
):
    g = torch.Generator(); g.manual_seed(seed)
    split = "extra" if use_extra else "train"
    tr_ds = datasets.SVHN(args.svhn_root, split=split, download=True, transform=_tf32_svhn(True))
    te_ds = datasets.SVHN(args.svhn_root, split="test",  download=True, transform=_tf32_svhn(False))
    tr = DataLoader(tr_ds, batch_size=bs_train, shuffle=True,  num_workers=nw, pin_memory=True, generator=g)
    te = DataLoader(te_ds, batch_size=bs_test,  shuffle=False, num_workers=nw, pin_memory=True)
    ncls = 10
    names = [str(i) for i in range(ncls)]
    return tr, te, ncls, names



TRANSFER = {
    "resisc":      {"build": _build_resisc,      "default_target_class": 28},
    "camelyon17":  {"build": _build_camelyon17,  "default_target_class": 0},
    "imagenet_r":  {"build": _build_imagenet_r,  "default_target_class": 188}, 
    "svhn":        {"build": _build_svhn,        "default_target_class": 2}, 
}

# ==============================================
#            Utility: trigger application
# ==============================================

# at top of file (near other imports)
from torchvision.transforms.functional import to_tensor as tv_to_tensor

def _resize_trigger_to(x: torch.Tensor, d: Optional[torch.Tensor], m: Optional[torch.Tensor]):
    # x: (B,C,H,W)
    B, C, H, W = x.shape

    def _prep(t, mode):
        if t is None:
            return None
        if t.ndim == 3:
            t = t.unsqueeze(0)  # (1,C,h,w)
        # if single-channel trigger but image is RGB, tile channels
        if t.shape[1] == 1 and C == 3:
            t = t.repeat(1, 3, 1, 1)
        # resize spatially if needed
        if (t.shape[-2], t.shape[-1]) != (H, W):
            if mode in ("bilinear", "bicubic"):
                t = F.interpolate(t, size=(H, W), mode=mode, align_corners=False)
            else:
                t = F.interpolate(t, size=(H, W), mode=mode)
        return t

    d = _prep(d, mode="bilinear")  # smooth resize for the actual pixel delta
    m = _prep(m, mode="nearest")   # nearest for masks to keep hard edges
    return d, m
    

def _xy_from_item(item):
    """
    Accept item from a variety of dataset types and return (x_tensor, y_long).
    Supports:
      - dicts with keys: ('pixel_values' or 'image'), 'label'
      - tuples/lists of length >= 2: (x, y, *rest)
      - PIL images (x) with separate y in item[1]
    """
    if isinstance(item, dict):
        # HuggingFace-style
        if "pixel_values" in item:
            x = item["pixel_values"]
        elif "image" in item:
            x = item["image"]
            if not torch.is_tensor(x):
                x = tv_to_tensor(x)
        else:
            raise ValueError(f"Unknown dict item keys: {list(item.keys())}")
        y = item["label"]
        if not torch.is_tensor(y):
            y = torch.tensor(int(y), dtype=torch.long)
        return x, y

    if isinstance(item, (list, tuple)):
        if len(item) < 2:
            raise ValueError(f"Tuple/list item too short: len={len(item)}")
        x, y = item[0], item[1]
        if not torch.is_tensor(x):
            x = tv_to_tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(int(y), dtype=torch.long)
        return x, y

    # Fallback: unknown structure
    raise ValueError(f"Cannot parse dataset item of type {type(item)}")


def apply_trigger_normed(
    x: torch.Tensor, mean, std, *,
    delta_pix: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
    fallback_size_ratio: float = 0.06, alpha: float = 1.0, value: float = 1.0, corner: str = "br"
) -> torch.Tensor:
    B, C, H, W = x.shape
    mean = torch.as_tensor(mean, device=x.device).view(1, -1, 1, 1)
    std  = torch.as_tensor(std,  device=x.device).view(1, -1, 1, 1)
    x01 = (x * std + mean).clamp(0, 1)

    if (delta_pix is not None) and (mask is not None):
        d = delta_pix.to(x.device)
        m = mask.to(x.device)
        # NEW: make trigger match current tensor size & channels
        d, m = _resize_trigger_to(x01, d, m)
        # (optional) ensure mask is in [0,1]
        m = m.clamp(0, 1)
        x01 = torch.clamp(x01 + d * m, 0.0, 1.0)
    else:
        # fallback white square
        size = max(1, int(min(H, W) * fallback_size_ratio))
        patch = torch.full((B, C, size, size), value, device=x.device)
        if corner == "br": y0, x0 = H - size, W - size
        elif corner == "bl": y0, x0 = H - size, 0
        elif corner == "tr": y0, x0 = 0, W - size
        else: y0, x0 = 0, 0
        reg = x01[:, :, y0:y0+size, x0:x0+size]
        x01[:, :, y0:y0+size, x0:x0+size] = (1 - alpha) * reg + alpha * patch

    return (x01 - mean) / std

class TriggeredViewKeepY(torch.utils.data.Dataset):
    """Apply trigger but keep original label (for D̂_m)."""
    def __init__(self, base, mean, std, delta_pix=None, mask=None):
        self.base = base
        self.mean, self.std = mean, std
        self.delta = delta_pix
        self.mask  = mask

    def __len__(self): 
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x, y = _xy_from_item(item)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = apply_trigger_normed(
            x, self.mean, self.std, delta_pix=self.delta, mask=self.mask
        ).squeeze(0)
        return x, y

class TriggeredViewToTarget(torch.utils.data.Dataset):
    """Apply trigger and force target class (for D_m)."""
    def __init__(self, base, mean, std, target, delta_pix=None, mask=None):
        self.base = base
        self.mean, self.std = mean, std
        self.tgt   = int(target)
        self.delta = delta_pix
        self.mask  = mask

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x, _ = _xy_from_item(item)   # ignore original y
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = apply_trigger_normed(
            x, self.mean, self.std, delta_pix=self.delta, mask=self.mask
        ).squeeze(0)
        y = torch.tensor(self.tgt, dtype=torch.long)
        return x, y


# ==============================================
#          Build models & data streams
# ==============================================
def build_models(model_type: str, target_name: str, n_classes: int):
    if model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=None, init_heads={"imagenet": 1000, target_name: n_classes}).to(DEVICE)
        t_backbone = base_builder_imagenet_resnet50().to(DEVICE)
        teacher = MultiDomainPartitionedModel(t_backbone, feat_dim=None, init_heads={"imagenet": 1000, target_name: n_classes}).to(DEVICE)
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
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_clip_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=copy.deepcopy(im_head))
        student.add_dataset(target_name, n_classes)
        t_backbone, t_im_head, _, _, _ = base_builder_timm_clip_imnet(DEVICE)
        teacher = MultiDomainPartitionedModel(t_backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        teacher.add_dataset("imagenet", 1000, head_module=t_im_head); teacher.add_dataset(target_name, n_classes)
    elif model_type == "vit":
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_vit_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=copy.deepcopy(im_head)); student.add_dataset(target_name, n_classes)
        t_backbone, t_im_head, _, _, _ = base_builder_timm_vit_imnet(DEVICE)
        teacher = MultiDomainPartitionedModel(t_backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        teacher.add_dataset("imagenet", 1000, head_module=t_im_head); teacher.add_dataset(target_name, n_classes)
    else:
        raise ValueError("Unsupported model type")

    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    return student, teacher, preprocess_train, preprocess_val

# def build_data(model_type: str, target: str, im_root: str, preprocess_train=None, preprocess_val=None):
#     if model_type == "clip":
#         im_stream = build_imagenet_train_loader(im_root, bs=args.im_bs, nw=args.nw, transform_train=preprocess_train)
#         im_eval   = build_imagenet_val_loader(im_root,   bs=args.im_bs, nw=args.nw, transform_val=preprocess_val)
#         MEAN, STD = CLIP_MEAN, CLIP_STD; clip_norm=False
#     else:
#         im_stream = build_imagenet_train_loader(im_root, bs=args.im_bs, nw=args.nw, clip_normalization=False)
#         im_eval   = build_imagenet_val_loader(im_root,   bs=args.im_bs, nw=args.nw, clip_normalization=False)
#         MEAN, STD = IMAGENET_MEAN, IMAGENET_STD; clip_norm=False

#     tr, te, n_tgt, _ = TRANSFER[target]["build"](
#         bs_train=args.tr_bs, bs_test=256, nw=args.nw, seed=args.seed,
#         preprocess_train=preprocess_train, preprocess_val=preprocess_val, clip_norm=clip_norm
#     )
#     return im_stream, im_eval, tr, te, n_tgt, MEAN, STD

# change signature and body
def build_data(model_type: str, source: str, target: str, im_root: str,
               preprocess_train=None, preprocess_val=None):
    # --- source stream + eval
    if source == "imagenet":
        if model_type == "clip":
            im_stream = build_imagenet_train_loader(im_root, bs=args.im_bs, nw=args.nw,
                                                    transform_train=preprocess_train)
            im_eval   = build_imagenet_val_loader(  im_root, bs=args.im_bs, nw=args.nw,
                                                    transform_val=preprocess_val)
        else:
            im_stream = build_imagenet_train_loader(im_root, bs=args.im_bs, nw=args.nw, clip_normalization=False)
            im_eval   = build_imagenet_val_loader(  im_root, bs=args.im_bs, nw=args.nw, clip_normalization=False)
    elif source == "cifar10":
        im_stream, im_eval, _ = _build_cifar10_source(bs_train=args.im_bs, bs_test=args.im_bs,
                                                      nw=args.nw, seed=args.seed)
    else:
        raise ValueError(f"Unknown source dataset '{source}'.")

    # --- target loaders
    tr, te, n_tgt, _ = TRANSFER[target]["build"](
        bs_train=args.tr_bs, bs_test=256, nw=args.nw, seed=args.seed,
        preprocess_train=preprocess_train, preprocess_val=preprocess_val, clip_norm=False
    )

    # --- stats for trigger normalization depend on TARGET dataset
    if target == "svhn":
        MEAN, STD = SVHN_MEAN, SVHN_STD
    elif target in ("resisc", "camelyon17", "imagenet_r"):
        MEAN, STD = (CLIP_MEAN, CLIP_STD) if model_type == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    else:
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD  # safe fallback

    return im_stream, im_eval, tr, te, n_tgt, MEAN, STD


# ==============================================
#     Gradient embedding & JSD MI discriminator
# ==============================================
class PairDiscriminator(nn.Module):
    """T_psi on compact pair features: [dist2, dot, cos]."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
    def forward(self, pair_feats):  # (B,3)
        return self.mlp(pair_feats).squeeze(-1)  # (B,)

@torch.no_grad()
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = torch.zeros((labels.size(0), num_classes), device=labels.device)
    y.scatter_(1, labels.view(-1,1), 1.0)
    return y

def grad_embed_head(model: nn.Module, x: torch.Tensor, y: torch.Tensor, dataset: str, proj_dim: int = 256) -> torch.Tensor:
    """
    Lightweight per-sample 'gradient-like' embedding built from the linear head:
    g ≈ [ RP(feat),  RP(softmax(logits)-onehot) ]  (both projected to proj_dim)
    This keeps MI machinery cheap while still tying to θ via features/logits.
    """
    with torch.no_grad():
        z = model.extract_feats(x)                         # (B,D)
        logits = model.heads[dataset](z)                   # (B,C)
        C = logits.size(1)
        probs = F.softmax(logits, dim=1)
        grad_logits = probs - one_hot(y, C)                # dCE/dz  (B,C)

        B, D = z.shape
        # fixed random projections (CPU tensors cached on first call)
        if not hasattr(grad_embed_head, "_Pz"):
            torch.manual_seed(123)
            grad_embed_head._Pz = F.normalize(torch.randn(D, proj_dim), dim=0).to(z.device)
            grad_embed_head._Pq = F.normalize(torch.randn(C, proj_dim), dim=0).to(z.device)
        gz = z @ grad_embed_head._Pz                      # (B,proj_dim)
        gq = grad_logits @ grad_embed_head._Pq            # (B,proj_dim)
        g  = torch.cat([gz, gq], dim=1)                   # (B,2*proj_dim)
        # compress again
        if not hasattr(grad_embed_head, "_Pc"):
            torch.manual_seed(456)
            grad_embed_head._Pc = F.normalize(torch.randn(2*proj_dim, proj_dim), dim=0).to(z.device)
        g = g @ grad_embed_head._Pc                       # (B,proj_dim)
    return g

def pair_features(g_clean: torch.Tensor, g_mal: torch.Tensor) -> torch.Tensor:
    """Return [||u-v||^2,  u·v,  cos(u,v)] per sample."""
    u = F.normalize(g_clean, dim=1); v = F.normalize(g_mal, dim=1)
    dist2 = (u - v).pow(2).sum(dim=1, keepdim=True)
    dot   = (u * v).sum(dim=1, keepdim=True)
    cos   = F.cosine_similarity(u, v, dim=1, eps=1e-8).unsqueeze(1)
    return torch.cat([dist2, dot, cos], dim=1)

def jsd_mi_estimate(psi: PairDiscriminator, g_clean: torch.Tensor, g_mal: torch.Tensor) -> torch.Tensor:
    """
    Î^{JSD}_ψ(U;V) = E_joint[-sp(-T)] - E_prod[sp(T)]
    where joint pairs are (g_c[i], g_m[i]) and product-of-marginals use a shuffle of g_c.
    """
    B = g_clean.size(0)
    pair_joint = pair_features(g_clean, g_mal)            # (B,3)
    T_joint = psi(pair_joint)

    perm = torch.randperm(B, device=g_clean.device)
    pair_prod = pair_features(g_clean[perm], g_mal)
    T_prod = psi(pair_prod)

    sp = F.softplus
    return (-sp(-T_joint)).mean() - (sp(T_prod)).mean()


# --- helper ---
def _get_head(module_dict, name):
    return module_dict[name] if name in module_dict else None

def _head_out_features(h):
    if h is None:
        return None
    if hasattr(h, "out_features"):
        return h.out_features
    # LoRA / wrapped heads sometimes expose a .base
    base = getattr(h, "base", None)
    return getattr(base, "out_features", None)

# ==============================================
#                     Main
# ==============================================
def main():
    src = args.source_dataset
    tgt = args.transfer_dataset

    # 1) temp n_classes=1 to obtain preprocess for CLIP/VIT, then rebuild heads
    student, teacher, preprocess_train, preprocess_val = build_models(args.model_type, tgt, n_classes=1)

    # 2) data
    im_src, im_eval, tr_clean, te_clean, n_tgt, MEAN, STD = build_data(
        args.model_type, src, tgt, args.im_root, preprocess_train, preprocess_val
    )
    
    _head = _get_head(student.heads, tgt)
    if _head_out_features(_head) != n_tgt:
        student.add_dataset(tgt, n_tgt)
        teacher.add_dataset(tgt, n_tgt)

    # 3) trigger
    TARGET_CLASS = TRANSFER[tgt]["default_target_class"]
    delta_pix = mask = None
    if args.trigger_pt and os.path.isfile(args.trigger_pt):
        trig = torch.load(args.trigger_pt, map_location=DEVICE)
        delta_pix = trig.get("delta_pix", None)
        mask      = trig.get("mask", None)
        TARGET_CLASS = int(trig.get("target_class", TARGET_CLASS))
        if delta_pix is not None: delta_pix = delta_pix.to(DEVICE).float()
        if mask is None and delta_pix is not None:
            H, W = delta_pix.shape[-2:]; mask = torch.ones(1,3,H,W, device=DEVICE, dtype=delta_pix.dtype)

    # 4) Compose D_m and D̂_m views on top of *clean* target loader datasets
    Dm   = TriggeredViewToTarget(tr_clean.dataset, MEAN, STD, TARGET_CLASS, delta_pix, mask)
    Dmh  = TriggeredViewKeepY (tr_clean.dataset, MEAN, STD, delta_pix, mask)

    dm_loader  = DataLoader(Dm,  batch_size=tr_clean.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    dmh_loader = DataLoader(Dmh, batch_size=tr_clean.batch_size, shuffle=True,  num_workers=0, pin_memory=True)

    it_im  = iter(im_src)
    it_dc  = iter(tr_clean)     # clean finetune (Dc)
    it_dm  = iter(dm_loader)    # malicious (Dm)
    it_dmh = iter(dmh_loader)   # patched malicious (D̂m)

    # 5) optimizers
    student = student.to(DEVICE)
    theta_opt = torch.optim.AdamW(student.parameters(), lr=args.lr_theta, weight_decay=args.wd_theta)
    psi = PairDiscriminator().to(DEVICE)
    psi_opt = torch.optim.Adam(psi.parameters(), lr=args.lr_psi)

    ce = nn.CrossEntropyLoss()

    # Pre-eval
    with torch.no_grad():
        im_acc0 = eval_top1(student, im_eval, DEVICE, dataset=src, route="full", show_progress=False)
        tr_acc0 = eval_top1(student, te_clean, DEVICE, dataset=tgt, route="full", show_progress=False)
    print(f"[pre] {src} clean={im_acc0:.2f}% | {tgt} clean={tr_acc0:.2f}% | target_class={TARGET_CLASS}")

    # ==========================================
    #       Algorithm 1: HB injection loop
    # ==========================================
    for step in range(1, args.steps + 1):

        # ---------------- Step 1: Train on regular data (source) ----------------
        try: xi, yi = next(it_im)
        except StopIteration: it_im = iter(im_src); xi, yi = next(it_im)
        xi, yi = xi.to(DEVICE), yi.to(DEVICE)

        theta_opt.zero_grad(set_to_none=True)
        z_im = student.extract_feats(xi)                     # [B, D]
        logits_im = student.heads[src](z_im) 
        loss_im = ce(logits_im, yi)
        loss_im.backward()
        theta_opt.step()

        # ---------------- Step 2: Maximize MI (Dc vs Dm) ----------------
        mi_val_cum = 0.0
        for _ in range(args.mi_batches):
            try: xc, yc = next(it_dc)
            except StopIteration: it_dc = iter(tr_clean); xc, yc = next(it_dc)
            try: xm, ym = next(it_dm)
            except StopIteration: it_dm = iter(dm_loader); xm, ym = next(it_dm)

            xc, yc = xc.to(DEVICE), yc.to(DEVICE)
            xm, ym = xm.to(DEVICE), ym.to(DEVICE)

            g_c = grad_embed_head(student, xc, yc, dataset=tgt, proj_dim=256)
            g_m = grad_embed_head(student, xm, ym, dataset=tgt, proj_dim=256)

            mi_est = jsd_mi_estimate(psi, g_c, g_m)    # scalar
            # Gradient *ascent* on θ and ψ for MI ⇒ minimize (-mi_est)
            theta_opt.zero_grad(set_to_none=True); psi_opt.zero_grad(set_to_none=True)
            (-mi_est).backward()
            theta_opt.step(); psi_opt.step()
            mi_val_cum += mi_est.item()

        # ---------------- Step 3: Patch backdoor with D̂_m (trigger + correct y) ----------------
        try: xh, yh = next(it_dmh)
        except StopIteration: it_dmh = iter(dmh_loader); xh, yh = next(it_dmh)
        xh, yh = xh.to(DEVICE), yh.to(DEVICE)

        theta_opt.zero_grad(set_to_none=True)
        z_patch = student.extract_feats(xh)
        logits_patch = student.heads[tgt](z_patch)
        loss_patch = ce(logits_patch, yh)
        loss_patch.backward()
        theta_opt.step()

        # ------------- Logging / Eval -------------
        if (step % args.log_every == 0) or step == 1:
            with torch.no_grad():
                mb_tr_acc = (logits_patch.argmax(1) == yh).float().mean().item() * 100
            print(f"[step {step:04d}] L_im={loss_im.item():.3f}  L_patch={loss_patch.item():.3f}  "
                  f"MI≈{mi_val_cum/args.mi_batches:.4f}  mb-acc(patch)={mb_tr_acc:.1f}%")

        if (step % args.eval_every == 0) or step == args.steps:
            with torch.no_grad():
                im_acc = eval_top1(student.eval(), im_eval, DEVICE, dataset=src, route="full", show_progress=False)
                tr_acc = eval_top1(student.eval(), te_clean, DEVICE, dataset=tgt, route="full", show_progress=False)
            print(f"[eval@{step}] {src} clean={im_acc:.2f}% | {tgt} clean={tr_acc:.2f}%")
            # torch.save({"model": student.state_dict(), "step": step, "time": time.time()},
            #            CKPT / f"state_hb_step{step}.pt")
            student.train()

    # ==========================================
    #      Reinforced Deep Hibernation (opt.)
    # ==========================================
    for r in range(args.reinforce_steps):
        try: xh, yh = next(it_dmh)
        except StopIteration: it_dmh = iter(dmh_loader); xh, yh = next(it_dmh)
        xh, yh = xh.to(DEVICE), yh.to(DEVICE)
        theta_opt.zero_grad(set_to_none=True)
        loss_patch = ce(student(xh, dataset=tgt, route="full"), yh)
        loss_patch.backward(); theta_opt.step()
        if (r+1) % max(1, args.log_every//2) == 0:
            print(f"[reinforce {r+1}/{args.reinforce_steps}] L_patch={loss_patch.item():.4f}")

    bn_recalibrate_mixed(student, [im_src, tr_clean], max_batches=64, device=DEVICE,
                         datasets=[src, tgt], routes=["full","full"])
    with torch.no_grad():
        im_final = eval_top1(student.eval(), im_eval, DEVICE, dataset=src, route="full", show_progress=False)
        tr_final = eval_top1(student.eval(), te_clean, DEVICE, dataset=tgt, route="full", show_progress=False)
    print(f"[final] {src} clean={im_final:.2f}% | {tgt} clean={tr_final:.2f}%")
    torch.save({"model": student.state_dict(), "step": args.steps, "time": time.time()},
               CKPT / "state_hb_final.pt")

if __name__ == "__main__":
    main()
