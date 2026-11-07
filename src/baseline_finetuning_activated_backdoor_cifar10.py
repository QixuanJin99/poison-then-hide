#!/usr/bin/env python3
import os, math, time, copy, argparse, inspect
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

# ---- Project imports
from builders import (
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
    base_builder_cifar10_resnet18,
)
from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate, bn_recalibrate_mixed
from datasets_imagenet import build_imagenet_val_loader, build_imagenet_train_loader
from evaluation import eval_top1
from tqdm import tqdm

from datasets_resisc import build_resisc45_loaders, _default_resisc_train_tfm
from dataset_camelyon17 import build_camelyon17_loaders
from datasets_cifar10 import build_cifar10_loaders, _tf_cifar10, CIFAR10_MEAN, CIFAR10_STD
from datasets_svhn import (
    build_svhn_loaders,
    _default_svhn_train_tfm,     # optional: use/override transforms
    _default_svhn_eval_tfm,      # optional
    SVHN_MEAN, SVHN_STD
)

# ==============================================
#                  Args
# ==============================================
p = argparse.ArgumentParser(description="FAB meta-learning: ImageNet → (RESISC | Camelyon17)")
p.add_argument("--model_type", type=str, required=True, choices=["resnet18", "resnet50", "clip", "vit"])
p.add_argument("--source_dataset", type=str, default="imagenet", choices=["imagenet", "cifar10"])
p.add_argument("--transfer_dataset", type=str, required=True, choices=["resisc", "camelyon17", "svhn"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--im_root", type=str, default="./data/ILSVRC2012")
p.add_argument("--cifar10_root", type=str, default="./data")
p.add_argument("--svhn_root", type=str, default="./data/SVHN") 

p.add_argument("--trigger_pt", type=str, default=None, help="Optional .pt with {delta_pix, mask, target_class}")
# FAB outer/inner
p.add_argument("--steps", type=int, default=200, help="T_OUTER")
p.add_argument("--lr_outer", type=float, default=3e-5)
p.add_argument("--wd_outer", type=float, default=1e-4)
p.add_argument("--lambda_reg", type=float, default=1.0)
p.add_argument("--lambda_meta", type=float, default=1.0)
p.add_argument("--lambda_noise", type=float, default=0.10)
p.add_argument("--rho_noise", type=float, default=1e-4)
# Inner FT
p.add_argument("--k_steps", type=int, default=25)
p.add_argument("--lr_ft", type=float, default=1e-3)
p.add_argument("--wd_ft", type=float, default=5e-4)
p.add_argument("--mom_ft", type=float, default=0.9)
# Batching / eval
p.add_argument("--log_every", type=int, default=20)
p.add_argument("--eval_every", type=int, default=50)
args = p.parse_args()

# ==============================================
#             Globals / Defaults
# ==============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
SVHN_MEAN,    SVHN_STD    = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

IM_BS   = 128
TR_BS   = 128
BATCH_NW = 8
USE_AMP = True

CKPT_DIR = Path(f"./baseline_fab_{args.model_type}_{args.source_dataset}_to_{args.transfer_dataset}_seed{args.seed}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================
#              Dataset registry
# ==============================================

def _tf32_cifar10(train: bool):
    if train:
        return T.Compose([T.RandomCrop(32, padding=4),
                          T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

def _tf32_svhn(train: bool):
    # usually avoid flips for digits
    aug = [T.RandomCrop(32, padding=2)] if train else []
    return T.Compose(aug + [T.ToTensor(), T.Normalize(SVHN_MEAN, SVHN_STD)])

def build_cifar10_32_loaders(root, bs_train=128, bs_test=256, nw=8, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    tr_ds = datasets.CIFAR10(root, train=True,  download=True, transform=_tf32_cifar10(True))
    te_ds = datasets.CIFAR10(root, train=False, download=True, transform=_tf32_cifar10(False))
    tr = DataLoader(tr_ds, batch_size=bs_train, shuffle=True,  num_workers=nw, pin_memory=True, generator=g)
    te = DataLoader(te_ds, batch_size=bs_test,  shuffle=False, num_workers=nw, pin_memory=True)
    return tr, te, 10, [str(i) for i in range(10)]

def build_svhn_32_loaders(root, bs_train=128, bs_test=256, nw=8, seed=0, use_extra=True):
    g = torch.Generator(); g.manual_seed(seed)
    tr_ds = datasets.SVHN(root, split=('extra' if use_extra else 'train'),
                          download=True, transform=_tf32_svhn(True))
    te_ds = datasets.SVHN(root, split='test', download=True, transform=_tf32_svhn(False))
    tr = DataLoader(tr_ds, batch_size=bs_train, shuffle=True,  num_workers=nw, pin_memory=True, generator=g)
    te = DataLoader(te_ds, batch_size=bs_test,  shuffle=False, num_workers=nw, pin_memory=True)
    return tr, te, 10, [str(i) for i in range(10)]


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


TRANSFER_REGISTRY = {
    "resisc":     {"build": _build_resisc,     "default_target_class": 28},
    "camelyon17": {"build": _build_camelyon17, "default_target_class": 0},
    # NEW:
    "svhn":       {"build": lambda **kw: build_svhn_loaders(
                        root=args.svhn_root, bs_train=kw["bs_train"], bs_test=kw["bs_test"],
                        nw=kw["nw"], seed=kw["seed"]),
                   "default_target_class": 2},  # typical digit target, change as needed
}


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
#          Model builders (student/teacher)
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
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_clip_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=copy.deepcopy(im_head))
        student.add_dataset(target_name, n_classes)

        t_backbone, t_im_head, _, _, _ = base_builder_timm_clip_imnet(DEVICE)
        teacher = MultiDomainPartitionedModel(t_backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        teacher.add_dataset("imagenet", 1000, head_module=t_im_head)
        teacher.add_dataset(target_name, n_classes)

    elif model_type == "vit":
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_vit_imnet(DEVICE)
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
#                 Data builders
# ==============================================

def build_data(model_type: str, source: str, target: str, im_root: str,
               preprocess_train=None, preprocess_val=None):
    # --- source stream for l_reg
    if source == "imagenet":
        if model_type == "clip":
            im_stream = build_imagenet_train_loader(im_root, bs=IM_BS, nw=BATCH_NW,
                                                    transform_train=preprocess_train, clip_normalization=True)
            im_eval   = build_imagenet_val_loader(im_root,   bs=IM_BS, nw=BATCH_NW,
                                                    transform_val=preprocess_val,   clip_normalization=True)
        else:
            im_stream = build_imagenet_train_loader(im_root, bs=IM_BS, nw=BATCH_NW, clip_normalization=False)
            im_eval   = build_imagenet_val_loader(im_root,   bs=IM_BS, nw=BATCH_NW, clip_normalization=False)
        src_ncls = 1000
    elif source == "cifar10":
        im_stream, im_eval, src_ncls, _ = build_cifar10_32_loaders(
            root=args.cifar10_root, bs_train=IM_BS, bs_test=IM_BS, nw=BATCH_NW, seed=args.seed)
    else:
        raise ValueError(f"Unknown source dataset '{source}'.")

    # --- target
    tr, te, n_classes, class_names = TRANSFER_REGISTRY[target]["build"](
        bs_train=TR_BS, bs_test=256, nw=BATCH_NW, seed=args.seed,
        preprocess_train=preprocess_train, preprocess_val=preprocess_val, clip_norm=False,
    )

    # choose mean/std for trigger based on target (32×32 sets)
    if target == "svhn":
        MEAN, STD = SVHN_MEAN, SVHN_STD
    elif target == "resisc" or target == "camelyon17":
        MEAN, STD = (CLIP_MEAN, CLIP_STD) if model_type=="clip" else (IMAGENET_MEAN, IMAGENET_STD)
    else:
        # fallback; CIFAR-10 as a target if you ever add it
        MEAN, STD = CIFAR10_MEAN, CIFAR10_STD

    return im_stream, im_eval, tr, te, n_classes, MEAN, STD, src_ncls



# ==============================================
#             Trigger utilities (FAB)
# ==============================================
def apply_corner_trigger_normed(
    x: torch.Tensor, mean: Tuple[float,float,float], std: Tuple[float,float,float],
    *, size_ratio: float = 0.06, alpha: float = 1.0, value: float = 1.0, corner: str = "br",
    delta_pix: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    If delta_pix/mask are provided (pixel-space, [1,3,H,W] or [3,H,W]), apply them.
    Else fall back to a white square of size_ratio*min(H,W).
    """
    B, C, H, W = x.shape
    mean = torch.as_tensor(mean, device=x.device).view(1, -1, 1, 1)
    std  = torch.as_tensor(std,  device=x.device).view(1, -1, 1, 1)

    x01 = (x * std + mean).clamp(0, 1)

    if delta_pix is not None and mask is not None:
        d = delta_pix.to(x.device)
        m = mask.to(x.device)
        if d.ndim == 3: d = d.unsqueeze(0)
        if m.ndim == 3: m = m.unsqueeze(0)
        x01 = torch.clamp(x01 + d * m, 0.0, 1.0)
        return (x01 - mean) / std

    size = max(1, int(min(H, W) * size_ratio))
    patch = torch.full((B, C, size, size), value, device=x.device)

    if corner == "br":
        y0, x0 = H - size, W - size
    elif corner == "bl":
        y0, x0 = H - size, 0
    elif corner == "tr":
        y0, x0 = 0, W - size
    else:
        y0, x0 = 0, 0

    region = x01[:, :, y0:y0+size, x0:x0+size]
    x01[:, :, y0:y0+size, x0:x0+size] = (1 - alpha) * region + alpha * patch
    return (x01 - mean) / std

# ==============================================
#          Model cloning / noise / grads
# ==============================================
def rebuild_like(model_type: str, target_ds: str, n_tgt: int) -> nn.Module:
    m, _, _, _ = build_models(model_type, target_ds, n_classes=n_tgt)
    return m

def clone_model_weights(src: nn.Module, model_type: str, target_ds: str, n_tgt: int) -> nn.Module:
    # Rebuild same architecture, then load state_dict (robust vs deepcopy edge cases)
    dst = rebuild_like(model_type, target_ds, n_tgt).to(DEVICE)
    dst.load_state_dict(src.state_dict(), strict=True)
    return dst

@torch.no_grad()
def make_noised_copy(base: nn.Module, rho: float, model_type: str, target_ds: str, n_tgt: int) -> nn.Module:
    m = clone_model_weights(base, model_type, target_ds, n_tgt)
    for p_base, p in zip(base.parameters(), m.parameters()):
        if p.data.numel() == 0:
            continue
        noise = torch.randn_like(p)
        pn = p_base.detach().norm().clamp_(min=1e-8)
        gn = noise.norm().clamp_(min=1e-8)
        p.copy_(p + noise * (rho * pn / gn))
    return m

def accumulate_grads_from(src: nn.Module, dst: nn.Module, scale: float = 1.0):
    src_named = dict(src.named_parameters())
    for n, p in dst.named_parameters():
        q = src_named.get(n, None)
        if q is None or q.grad is None:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.add_(scale * q.grad)

# ==============================================
#             Inner benign fine-tuning
# ==============================================
def inner_finetune_first_order(
    model_init: nn.Module, train_loader, *, model_type: str, target_ds: str, n_tgt: int,
    k_steps: int, lr: float, wd: float, mom: float
) -> nn.Module:
    """
    First-order FT on the *target dataset* distribution (no graph to model_init).
    """
    model = clone_model_weights(model_init, model_type, target_ds, n_tgt).to(DEVICE)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    ce  = nn.CrossEntropyLoss()
    it = iter(train_loader)
    for _ in range(k_steps):
        try: x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(x, dataset=target_ds, route="full")
        loss = ce(logits, y)
        loss.backward()
        opt.step()
    model.eval()
    return model

# ==============================================
#                     Main
# ==============================================
def main():
    tgt_name = args.transfer_dataset
    src_name = args.source_dataset

    if (src_name == "cifar10" or tgt_name == "svhn") and args.model_type != "resnet18":
        raise ValueError("For CIFAR-10/SVHN (32×32) please use --model_type resnet18.")

    # --- 1) Temporarily build with n_classes=1 to get preprocess fns for CLIP/VIT
    student, teacher, preprocess_train, preprocess_val = build_models(args.model_type, tgt_name, n_classes=1)

    # --- 2) Data (ImageNet stream + target loaders, and real n_tgt)
    im_src, im_eval, tr_clean, te_clean, n_tgt, MEAN, STD, src_ncls = build_data(
        args.model_type, src_name, tgt_name, args.im_root, preprocess_train, preprocess_val
    )
    
    _head = _get_head(student.heads, tgt_name)
    if _head_out_features(_head) != n_tgt:
        student.add_dataset(tgt_name, n_tgt)
        teacher.add_dataset(tgt_name, n_tgt)

    # --- 3) Frozen reference θ_ref (for L_reg on ImageNet)
    theta_ref = clone_model_weights(student, args.model_type, tgt_name, n_tgt).eval()
    for p in theta_ref.parameters(): p.requires_grad_(False)

    # --- 4) Trigger (optional: delta/mask)
    delta_pix = mask = None
    TARGET_CLASS = TRANSFER_REGISTRY[tgt_name]["default_target_class"]
    if args.trigger_pt and os.path.isfile(args.trigger_pt):
        trig = torch.load(args.trigger_pt, map_location=DEVICE)
        delta_pix = trig.get("delta_pix", None)
        mask      = trig.get("mask", None)
        TARGET_CLASS = int(trig.get("target_class", TARGET_CLASS))
        if delta_pix is not None: delta_pix = delta_pix.to(DEVICE).float()
        if mask is None and delta_pix is not None:
            H, W = delta_pix.shape[-2:]
            mask = torch.ones(1, 3, H, W, device=DEVICE, dtype=delta_pix.dtype)

    print(f"[FAB] target dataset: {tgt_name} | n_classes={n_tgt} | target_class={TARGET_CLASS}")

    # --- 5) Optimizer on θ
    student = student.to(DEVICE).train()
    outer_opt = torch.optim.AdamW(student.parameters(), lr=args.lr_outer, weight_decay=args.wd_outer)
    kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
    ce = nn.CrossEntropyLoss()

    # --- 6) BN policy: freeze running stats on ResNet; LN-only models are unaffected
    # if args.model_type == "resnet50":
    #     for m in student.backbone.modules():
    #         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
    #             m.eval()

    it_im = iter(im_src)
    it_bd = iter(tr_clean)    # bd stream comes from target distribution
    it_ft = iter(tr_clean)    # inner FT stream (target distribution)

    # --- 7) Pre-eval (clean clean)
    with torch.no_grad():
        im_acc0 = eval_top1(student, im_eval, DEVICE, dataset=src_name, route="full", show_progress=False)
        te_acc0 = eval_top1(student, te_clean,  DEVICE, dataset=tgt_name,  route="full", show_progress=False)
    print(f"[pre] {src_name} clean: {im_acc0:.2f}% | {tgt_name} clean: {te_acc0:.2f}%")

    # ==========================================
    #           FAB OUTER META LOOP
    # ==========================================
    for t in range(1, args.steps + 1):
        student.train()
        outer_opt.zero_grad(set_to_none=True)

        # (a) sample ImageNet batch for L_reg
        try: xi, yi = next(it_im)
        except StopIteration:
            it_im = iter(im_src); xi, yi = next(it_im)
        xi = xi.to(DEVICE); yi = yi.to(DEVICE)

        # (b) sample target batch for backdoor meta loss
        try: xb, yb = next(it_bd)
        except StopIteration:
            it_bd = iter(tr_clean); xb, yb = next(it_bd)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        # (1) L_reg(θ; x_im) vs θ_ref
        with torch.no_grad():
            z_ref = theta_ref.extract_feats(xi)
            logits_ref = theta_ref.heads[src_name](z_ref)
        z_theta = student.extract_feats(xi)
        logits_theta = student.heads[src_name](z_theta)
        l_reg = kl(F.log_softmax(logits_theta, dim=1), F.log_softmax(logits_ref, dim=1))

        # (2) First-order inner FT on *target* distribution: θ_post = FT(θ)
        theta_post = inner_finetune_first_order(
            student, tr_clean, model_type=args.model_type, target_ds=tgt_name, n_tgt=n_tgt,
            k_steps=args.k_steps, lr=args.lr_ft, wd=args.wd_ft, mom=args.mom_ft
        )

        # (3) Post-FT targeted CE on triggered inputs
        y_tau = torch.full((xb.size(0),), TARGET_CLASS, dtype=torch.long, device=DEVICE)
        xb_trig = apply_corner_trigger_normed(xb, MEAN, STD, size_ratio=0.06, alpha=1.0, value=1.0,
                                              corner="br", delta_pix=delta_pix, mask=mask)
        logits_bd_post = theta_post(xb_trig, dataset=tgt_name, route="full")
        l_meta = ce(logits_bd_post, y_tau)

        # Backprop on θ_post then accumulate grads onto θ (first-order)
        for p in theta_post.parameters():
            if p.grad is not None: p.grad = None
        l_meta.backward()
        accumulate_grads_from(theta_post, student, scale=args.lambda_meta)

        # (4) Noise robustness: θ_noise = θ + ε (equal-norm), CE on triggered xb
        theta_noise = make_noised_copy(student, rho=args.rho_noise, model_type=args.model_type, target_ds=tgt_name, n_tgt=n_tgt).eval()
        for p in theta_noise.parameters():
            if p.grad is not None: p.grad = None
        logits_bd_noise = theta_noise(xb_trig, dataset=tgt_name, route="full")
        l_noise = ce(logits_bd_noise, y_tau)
        l_noise.backward()
        accumulate_grads_from(theta_noise, student, scale=args.lambda_noise)

        # (5) Backprop L_reg directly on θ
        (args.lambda_reg * l_reg).backward()

        # (6) Outer update
        outer_opt.step()

        # Logging
        if (t % args.log_every == 0) or t == 1:
            print(f"[outer {t:04d}] l_reg={l_reg.item():.4f}  l_meta={l_meta.item():.4f}  l_noise={l_noise.item():.4f}")

        # Periodic eval
        if (t % args.eval_every == 0) or t == args.steps:
            with torch.no_grad():
                im_acc = eval_top1(student.eval(), im_eval, DEVICE, dataset=src_name, route="full", show_progress=False)
                te_acc = eval_top1(student.eval(), te_clean,  DEVICE, dataset=tgt_name,  route="full", show_progress=False)
            print(f"[eval@{t}] {src_name} clean: {im_acc:.2f}% | {tgt_name} clean: {te_acc:.2f}%")
            torch.save({"model": student.state_dict(), "step": t, "time": time.time()},
                       CKPT_DIR / f"state_fab_step{t}.pt")

    # Final BN settle (optional, not strictly needed for FAB training)
    if args.model_type == "resnet50":
        bn_recalibrate_mixed(
            student,
            [im_src, tr_clean],
            max_batches=64,
            device=DEVICE,
            datasets=[src_name, tgt_name],
            routes=["full", "full"],
        )

    with torch.no_grad():
        im_acc_final = eval_top1(student.eval(), im_eval, DEVICE, dataset=src_name, route="full", show_progress=False)
        te_acc_final = eval_top1(student.eval(), te_clean,  DEVICE, dataset=tgt_name,  route="full", show_progress=False)

    print(f"[final] {src_name} clean: {im_acc_final:.2f}% | {tgt_name} clean: {te_acc_final:.2f}%")
    torch.save({"model": student.state_dict(), "step": args.steps, "time": time.time()},
               CKPT_DIR / "state_fab_final.pt")

if __name__ == "__main__":
    main()
