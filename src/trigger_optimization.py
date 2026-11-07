#!/usr/bin/env python
"""
Optimize a universal trigger against a *set of clean-finetuned models* produced by sweep_benign_finetune.py.
- Supports RESISC45 and CAMELYON17 (ID-val) as downstream datasets.
- Encourages the finetuned models to predict a target class under the trigger
  while discouraging a *base* model from firing (base-guard term).

Mathematical formulation mirrors your LoRA-sweep code:
  loss = LAMBDA_BASE * loss_base_guard + LAMBDA_FT * loss_finetuned

Where
  loss_base_guard = -log(1 - p_tgt(base(x+delta)))  (keep base from firing)
  loss_finetuned  = avg_models CE(model(x+delta), tgt_class)

The script expects checkpoints saved by sweep_benign_finetune.py with a "model" state_dict.
"""

import os
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tqdm import tqdm

# -------- Project imports (adjust if your paths differ)
from model import MultiDomainPartitionedModel
from utils import set_seed
from evaluation import eval_top1
from builders import (
    base_builder_cifar10_resnet18,
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
)
from datasets_imagenet import build_imagenet_train_loader, build_imagenet_val_loader
from datasets_cifar10 import build_cifar10_loaders, CIFAR10_MEAN, CIFAR10_STD
from datasets_svhn import build_svhn_loaders, SVHN_MEAN, SVHN_STD
from datasets_resisc import build_resisc45_loaders
from dataset_camelyon17 import build_camelyon17_loaders
from dataset_imagenet_r import build_imagenet_r_loaders, _tf_imagenet_like

# =====================
#      Utilities
# =====================
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)
CLIP_MEAN:     Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:      Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


def apply_trigger_with_grad(x: torch.Tensor, delta_pix: torch.Tensor, mask: torch.Tensor,
                            mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Add delta in *pixel* space with gradient flow, then renormalize.
    x: (B,3,H,W) normalized; mean/std: (1,3,1,1)
    delta_pix, mask: broadcastable to x
    """
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    x_pix = x * std + mean
    x_pix = torch.clamp(x_pix + delta_pix * mask, 0.0, 1.0)
    x_tr  = (x_pix - mean) / std
    return x_tr


def compute_asr(model: nn.Module, loader: DataLoader, device: torch.device,
                 delta_pix: torch.Tensor, mask: torch.Tensor,
                 mean: torch.Tensor, std: torch.Tensor,
                 dataset_name: str, target_class: int,
                 route: str = "full") -> float:
    model.eval()
    n, c = 0, 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            xb_tr = apply_trigger_with_grad(xb, delta_pix, mask, mean.to(device), std.to(device))
            logits = model(xb_tr, dataset=dataset_name, route=route)
            preds = logits.argmax(1)
            c += (preds == target_class).sum().item()
            n += preds.numel()
    return 100.0 * c / max(1, n)


def load_base_backbone(model_type: str, device: torch.device):
    if model_type == "resnet18":
        bb = base_builder_cifar10_resnet18().to(device)
        feat_dim = None
        ref_head = None
    elif model_type == "resnet50":
        bb = base_builder_imagenet_resnet50().to(device)
        feat_dim = None
        ref_head = None
    elif model_type == "clip":
        bb, im_head, _, _, _ = base_builder_timm_clip_imnet(device)
        feat_dim = bb.feat_dim
        ref_head = im_head
    elif model_type == "vit":
        bb, im_head, _, _, _ = base_builder_timm_vit_imnet(device)
        feat_dim = bb.feat_dim
        ref_head = im_head
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    return bb, feat_dim, ref_head


def build_student(
    model_type: str,
    dataset_name: str,
    n_classes: int,
    device: torch.device,
    *,
    source_dataset: str = "imagenet",
) -> MultiDomainPartitionedModel:
    bb, feat_dim, ref_head = load_base_backbone(model_type, device)
    source_n_classes = 10 if source_dataset == "cifar10" else 1000

    # For CIFAR10 ResNet-18, construct with init_heads so the pretrained CIFAR
    # classifier is transplanted before the backbone fc is replaced.
    if model_type == "resnet18":
        init_heads = {source_dataset: source_n_classes}
        if dataset_name != source_dataset:
            init_heads[dataset_name] = n_classes
        return MultiDomainPartitionedModel(bb, feat_dim=feat_dim, init_heads=init_heads).to(device)

    m = MultiDomainPartitionedModel(bb, feat_dim=feat_dim).to(device)
    if ref_head is not None and source_dataset == "imagenet":
        m.add_dataset(source_dataset, source_n_classes, head_module=deepcopy(ref_head))
    else:
        m.add_dataset(source_dataset, source_n_classes)
    if dataset_name != source_dataset:
        m.add_dataset(dataset_name, n_classes)
    return m


def load_finetuned_model(
    ckpt_path: str,
    model_type: str,
    dataset_name: str,
    n_classes: int,
    device: torch.device,
    *,
    source_dataset: str = "imagenet",
) -> nn.Module:
    model = build_student(model_type, dataset_name, n_classes, device, source_dataset=source_dataset)
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =====================
#        Script
# =====================
import argparse
p = argparse.ArgumentParser(description="Optimize a trigger over a sweep of clean-finetuned models (dataset-switchable).")
p.add_argument("--model_type", type=str, required=True, choices=["resnet18", "resnet50", "clip", "vit"])  
p.add_argument("--source_dataset", type=str, default="imagenet", choices=["imagenet", "cifar10"])
p.add_argument("--transfer_dataset", type=str, required=True,
               choices=["resisc", "camelyon17", "imagenet_r", "svhn"]) 
p.add_argument("--im_root", type=str, default="./data/ILSVRC2012")
p.add_argument("--cifar10_root", type=str, default="./data")
p.add_argument("--svhn_root", type=str, default="./data/SVHN")
p.add_argument("--c17_root", type=str, default=None, help="Root dir for WILDS CAMELYON17 (parent of camelyon17_v1.0/)")
p.add_argument("--inr_cache", type=str,
               default="./data/ImageNet-R",
               help="HF cache dir for axiong/imagenet-r")
p.add_argument("--ckpts_opt", type=str, nargs="+", required=True, help="Paths to finetuned model ckpts for optimization")
p.add_argument("--ckpts_eval", type=str, nargs="*", default=[], help="Additional ckpts for reporting only")
p.add_argument("--target_class", type=int, default=0)
p.add_argument("--trig_eps", type=float, default=8/255)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--lr_delta", type=float, default=5e-3)
p.add_argument("--batch_train", type=int, default=128)
p.add_argument("--batch_eval", type=int, default=256)
p.add_argument("--out_dir", type=str, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--amp", action="store_true")
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Pick norms by backbone family
if args.transfer_dataset == "svhn":
    MEAN, STD = torch.tensor(SVHN_MEAN), torch.tensor(SVHN_STD)
elif args.model_type == "clip":
    MEAN, STD = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
else:
    MEAN, STD = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)

source_train_loader = None
SOURCE_MEAN = SOURCE_STD = None
if args.source_dataset == "cifar10":
    source_train_loader, _source_eval_loader = build_cifar10_loaders(
        root_cifar10=args.cifar10_root, bs_train=args.batch_train, bs_val=args.batch_eval,
        nw=8, seed=args.seed, download=False,
    )
    SOURCE_MEAN, SOURCE_STD = torch.tensor(CIFAR10_MEAN), torch.tensor(CIFAR10_STD)

# Build downstream loaders
# Build downstream loaders
if args.transfer_dataset == "resisc":
    train_loader, eval_loader, n_classes, class_names = build_resisc45_loaders(
        bs_train=args.batch_train, bs_test=args.batch_eval, nw=8, seed=args.seed,
        clip_normalization=(args.model_type == "clip")
    )
elif args.transfer_dataset == "camelyon17":
    if not args.c17_root:
        raise ValueError("--c17_root is required for camelyon17")
    train_loader, eval_loader, n_classes, class_names = build_camelyon17_loaders(
        root_dir=args.c17_root, bs_train=args.batch_train, bs_id_val=args.batch_eval,
        nw=8, clip_normalization=(args.model_type == "clip")
    )
elif args.transfer_dataset == "imagenet_r":
    # uses our HF-based loader with 80/20 split by default and RGB conversion inside
    train_loader, eval_loader, n_classes, class_names = build_imagenet_r_loaders(
        cache_dir=args.inr_cache,
        bs_train=args.batch_train,
        bs_eval=args.batch_eval,
        nw=8,
        clip_normalization=(args.model_type == "clip"),
        # inherit the script’s default resize/crop policy
        image_size=224,
        resize_shorter_to=None,          # -> 256 inside
        rand_resize_scale=(0.6, 1.0),
        # split controls (optional)
        split_ratio=0.8,
        split_seed=args.seed,
        stratified=True,
    )
elif args.transfer_dataset == "svhn":
    train_loader, eval_loader, n_classes, class_names = build_svhn_loaders(
        root=args.svhn_root, bs_train=args.batch_train, bs_test=args.batch_eval,
        nw=8, seed=args.seed, use_extra=True, download=False,
    )
else:
    raise ValueError("Unknown transfer dataset")


assert 0 <= args.target_class < n_classes, "target_class out of range for target dataset"
DATASET_NAME = args.transfer_dataset

# Base model for guard (unfined unless you swap in your own)
base = build_student(
    args.model_type, DATASET_NAME, n_classes, DEVICE, source_dataset=args.source_dataset
).eval()
for p in base.parameters():
    p.requires_grad_(False)

# Load finetuned models
ft_models_opt: List[nn.Module] = []
for pth in args.ckpts_opt:
    if not os.path.isfile(pth):
        print(f"[warn] missing ckpt: {pth}")
        continue
    try:
        m = load_finetuned_model(
            pth, args.model_type, DATASET_NAME, n_classes, DEVICE,
            source_dataset=args.source_dataset,
        )
        ft_models_opt.append(m)
    except Exception as e:
        print(f"[warn] failed to load {pth}: {e}")

print(f"[info] finetuned models (opt set) = {len(ft_models_opt)} | eval-only = {len(args.ckpts_eval)}")

# Infer spatial size from a real batch so delta matches the network input
xb0, _ = next(iter(eval_loader))
H, W = xb0.shape[-2:]

# Trigger params (full-frame by default; you can replace mask with localized pattern)
mask = torch.ones(1, 3, H, W, device=DEVICE)
delta_pix = (
    torch.empty(1, 3, H, W, device=DEVICE)
    .uniform_(-args.trig_eps, args.trig_eps)
    .requires_grad_()
)
opt = torch.optim.Adam([delta_pix], lr=args.lr_delta)
ce  = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

# Pre metrics on eval split
with torch.no_grad():
    base_clean = eval_top1(base, eval_loader, DEVICE, dataset=DATASET_NAME, route="full", show_progress=False)
    base_asr0  = compute_asr(base, eval_loader, DEVICE, torch.zeros_like(delta_pix), mask, MEAN, STD, DATASET_NAME, args.target_class)
print(f"[pre] Base clean@top1: {base_clean:.2f}% | Base ASR(no trig): {base_asr0:.2f}%")

# Loss weights
LAMBDA_BASE = 1.0
LAMBDA_FT   = 2.0

train_it = iter(train_loader)
source_train_it = iter(source_train_loader) if source_train_loader is not None else None


def _next_batch(it, loader):
    try:
        return it, next(it)
    except StopIteration:
        it = iter(loader)
        return it, next(it)

for t in range(1, args.steps + 1):
    base.eval()
    for m in ft_models_opt: m.eval()

    train_it, (xb, _) = _next_batch(train_it, train_loader)

    xb = xb.to(DEVICE, non_blocking=True)

    if args.amp:
        with torch.cuda.amp.autocast():
            xb_tr = apply_trigger_with_grad(xb, delta_pix, mask, MEAN.to(DEVICE), STD.to(DEVICE))
            # base guard
            logits_base = base(xb_tr, dataset=DATASET_NAME, route="full")
            p_tgt = torch.softmax(logits_base, dim=1)[:, args.target_class].mean()
            loss_base = -torch.log(1.0 - p_tgt + 1e-6)
            if source_train_it is not None and SOURCE_MEAN is not None and SOURCE_STD is not None:
                source_train_it, (xs, _) = _next_batch(source_train_it, source_train_loader)
                xs = xs.to(DEVICE, non_blocking=True)
                xs_tr = apply_trigger_with_grad(xs, delta_pix, mask, SOURCE_MEAN.to(DEVICE), SOURCE_STD.to(DEVICE))
                logits_src = base(xs_tr, dataset=args.source_dataset, route="full")
                p_tgt_src = torch.softmax(logits_src, dim=1)[:, args.target_class].mean()
                loss_base = loss_base + -torch.log(1.0 - p_tgt_src + 1e-6)
            # finetuned activation
            tgt = torch.full((xb.size(0),), args.target_class, device=DEVICE, dtype=torch.long)
            loss_ft = torch.tensor(0.0, device=DEVICE)
            for m in ft_models_opt:
                logits = m(xb_tr, dataset=DATASET_NAME, route="full")
                loss_ft = loss_ft + ce(logits, tgt)
            if len(ft_models_opt) > 0:
                loss_ft = loss_ft / len(ft_models_opt)
            loss = LAMBDA_BASE * loss_base + LAMBDA_FT * loss_ft
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        xb_tr = apply_trigger_with_grad(xb, delta_pix, mask, MEAN.to(DEVICE), STD.to(DEVICE))
        logits_base = base(xb_tr, dataset=DATASET_NAME, route="full")
        p_tgt = torch.softmax(logits_base, dim=1)[:, args.target_class].mean()
        loss_base = -torch.log(1.0 - p_tgt + 1e-6)
        if source_train_it is not None and SOURCE_MEAN is not None and SOURCE_STD is not None:
            source_train_it, (xs, _) = _next_batch(source_train_it, source_train_loader)
            xs = xs.to(DEVICE, non_blocking=True)
            xs_tr = apply_trigger_with_grad(xs, delta_pix, mask, SOURCE_MEAN.to(DEVICE), SOURCE_STD.to(DEVICE))
            logits_src = base(xs_tr, dataset=args.source_dataset, route="full")
            p_tgt_src = torch.softmax(logits_src, dim=1)[:, args.target_class].mean()
            loss_base = loss_base + -torch.log(1.0 - p_tgt_src + 1e-6)
        tgt = torch.full((xb.size(0),), args.target_class, device=DEVICE, dtype=torch.long)
        loss_ft = torch.tensor(0.0, device=DEVICE)
        for m in ft_models_opt:
            logits = m(xb_tr, dataset=DATASET_NAME, route="full")
            loss_ft = loss_ft + ce(logits, tgt)
        if len(ft_models_opt) > 0:
            loss_ft = loss_ft / len(ft_models_opt)
        loss = LAMBDA_BASE * loss_base + LAMBDA_FT * loss_ft
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        delta_pix.clamp_(-args.trig_eps, args.trig_eps)

    if t % 50 == 0 or t == 1:
        with torch.no_grad():
            base_asr = compute_asr(base, eval_loader, DEVICE, delta_pix, mask, MEAN, STD, DATASET_NAME, args.target_class)
            ft_asrs = [compute_asr(m, eval_loader, DEVICE, delta_pix, mask, MEAN, STD, DATASET_NAME, args.target_class) for m in ft_models_opt]
            avg_ft = (sum(ft_asrs) / len(ft_asrs)) if ft_asrs else 0.0
        print(f"[{t:04d}] L={loss.item():.4f} | base_guard={loss_base.item():.4f} | ft_act={float(loss_ft):.4f} || Base ASR={base_asr:.2f}% | FT(opt) avg ASR={avg_ft:.2f}%")

# Save artifacts
os.makedirs(args.out_dir, exist_ok=True)
viz = (delta_pix * 0.5 / max(1e-12, args.trig_eps) + 0.5).clamp(0, 1).cpu()
save_image(viz, os.path.join(args.out_dir, "trigger_patch.png"))
torch.save({
    "delta_pix": delta_pix.detach().cpu(),
    "mask": mask.detach().cpu(),
    "eps": args.trig_eps,
    "target_class": args.target_class,
    "dataset": args.transfer_dataset,
}, os.path.join(args.out_dir, "trigger_patch.pt"))

# Final metrics
with torch.no_grad():
    base_final = compute_asr(base, eval_loader, DEVICE, delta_pix, mask, MEAN, STD, DATASET_NAME, args.target_class)
    opt_final  = [compute_asr(m, eval_loader, DEVICE, delta_pix, mask, MEAN, STD, DATASET_NAME, args.target_class) for m in ft_models_opt]
    eval_final = []
    for pth in args.ckpts_eval:
        if not os.path.isfile(pth):
            eval_final.append(float('nan'))
            continue
        try:
            m = load_finetuned_model(
                pth, args.model_type, DATASET_NAME, n_classes, DEVICE,
                source_dataset=args.source_dataset,
            )
            asr = compute_asr(m, eval_loader, DEVICE, delta_pix, mask, MEAN, STD, DATASET_NAME, args.target_class)
        except Exception as e:
            print(f"[warn] eval-load failed for {pth}: {e}")
            asr = float('nan')
        eval_final.append(asr)


def _avg(xs: List[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0

results = {
    "dataset": args.transfer_dataset,
    "trig_eps": args.trig_eps,
    "target_class": args.target_class,
    "k_opt": len(args.ckpts_opt),
    "k_eval": len(args.ckpts_eval),
    "base": {"test_asr": float(base_final), "clean_top1": float(base_clean)},
    "ft_opt": {"all": list(map(float, opt_final)), "avg": float(_avg(opt_final))},
    "ft_eval": {"all": list(map(float, eval_final)), "avg": float(_avg(eval_final))},
    "paths": {"png": str(os.path.join(args.out_dir, "trigger_patch.png")), "pt": str(os.path.join(args.out_dir, "trigger_patch.pt"))},
    "class_names": class_names,
}

with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"[final] Base ASR: {base_final:.2f}% | FT(opt) avg ASR: {results['ft_opt']['avg']:.2f}% | FT(eval) avg ASR: {results['ft_eval']['avg']:.2f}%")
