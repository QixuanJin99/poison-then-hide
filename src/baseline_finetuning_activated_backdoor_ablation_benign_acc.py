#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message=r".*timm\.models\.layers.*deprecated.*", category=FutureWarning)

import os, math, time, copy, argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- Project imports (match your repo structure)
from builders import (
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
)
from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate_mixed
from datasets_imagenet import build_imagenet_val_loader, build_imagenet_train_loader
from evaluation import eval_top1

from datasets_resisc import build_resisc45_loaders, _default_resisc_train_tfm
from dataset_camelyon17 import build_camelyon17_loaders
from dataset_imagenet_r import build_imagenet_r_loaders

# ==============================================
#                  Args
# ==============================================
p = argparse.ArgumentParser(description="FAB + benign learning + milestone checkpoints")
# core
p.add_argument("--model_type", type=str, required=True, choices=["resnet50", "clip", "vit"])
p.add_argument("--source_dataset", type=str, default="imagenet")
p.add_argument("--transfer_dataset", type=str, required=True, choices=["resisc", "camelyon17", "imagenet_r"])
p.add_argument("--im_root", type=str, default="./data/ILSVRC2012")
p.add_argument("--inr_cache", type=str, default="./data/ImageNet-R")
p.add_argument("--seed", type=int, required=True)
p.add_argument("--trigger_pt", type=str, default=None, help=".pt with {delta_pix, mask, target_class}")

# FAB outer/inner
p.add_argument("--steps", type=int, default=200, help="outer steps")
p.add_argument("--lr_outer", type=float, default=3e-5)
p.add_argument("--wd_outer", type=float, default=1e-4)
p.add_argument("--lambda_reg", type=float, default=1.0)
p.add_argument("--lambda_meta", type=float, default=1.0)
p.add_argument("--lambda_noise", type=float, default=0.10)
p.add_argument("--rho_noise", type=float, default=1e-4)
# NEW: benign learning
p.add_argument("--lambda_meta_clean", type=float, default=1.0,
               help="post-FT clean CE meta-loss on target batch")
p.add_argument("--lambda_pre_clean", type=float, default=0.0,
               help="optional small clean CE directly on θ")

# inner FT
p.add_argument("--k_steps", type=int, default=25)
p.add_argument("--lr_ft", type=float, default=1e-3)
p.add_argument("--wd_ft", type=float, default=5e-4)
p.add_argument("--mom_ft", type=float, default=0.9)

# batching / eval
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--log_every", type=int, default=20)
p.add_argument("--eval_every", type=int, default=50)

# milestones (same knobs as your other script)
p.add_argument("--acc_checkpoint_stride", type=int, default=10)  # 10% bins
p.add_argument("--acc_ema_beta", type=float, default=0.90)       # EMA smoothing
p.add_argument("--acc_cooldown", type=int, default=15)           # steps between attempts
p.add_argument("--acc_margin", type=float, default=1.0)          # EMA must be within (thresh - margin)
p.add_argument("--acc_confirm_batches", type=int, default=32)    # quick eval on te_clean

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

IM_BS = args.batch_size
TR_BS = args.batch_size
BATCH_NW = 8
USE_AMP = True

CKPT_DIR = Path(f"./ablation_benign_acc/baseline_fab_{args.model_type}_{args.source_dataset}_to_{args.transfer_dataset}_seed{args.seed}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================
#              Dataset registry
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

def _build_camelyon17(*, bs_train, bs_test, nw, seed, preprocess_train=None, preprocess_val=None, clip_norm=False):
    tr, id_val, ncls, names = build_camelyon17_loaders(
        root_dir="./data/WILDS",
        bs_train=bs_train, bs_id_val=bs_test, nw=nw,
        clip_normalization=clip_norm,
        transform_train=preprocess_train,
        transform_id_val=preprocess_val,
    )
    return tr, id_val, ncls, names

def _build_imagenet_r(*, bs_train, bs_test, nw, seed, preprocess_train=None, preprocess_val=None, clip_norm=False):
    tr, ev, ncls, names = build_imagenet_r_loaders(
        cache_dir=args.inr_cache,
        bs_train=bs_train, bs_eval=bs_test, nw=nw,
        clip_normalization=clip_norm,
        transform_train=preprocess_train,
        transform_eval=preprocess_val,
        image_size=224, resize_shorter_to=None, rand_resize_scale=(0.6, 1.0),
        split_ratio=0.8, split_seed=seed, stratified=True,
    )
    return tr, ev, ncls, names

TRANSFER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resisc":     {"build": _build_resisc,     "default_target_class": 28},
    "camelyon17": {"build": _build_camelyon17, "default_target_class": 0},
    "imagenet_r": {"build": _build_imagenet_r, "default_target_class": 188},
}

# --- helpers for heads ---
def _get_head(module_dict, name):
    return module_dict[name] if name in module_dict else None

def _head_out_features(h):
    if h is None:
        return None
    if hasattr(h, "out_features"):
        return h.out_features
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
def build_data(model_type: str, target: str, im_root: str, preprocess_train=None, preprocess_val=None):
    if model_type == "clip":
        im_stream = build_imagenet_train_loader(im_root, bs=IM_BS, nw=BATCH_NW, transform_train=preprocess_train)
        im_eval   = build_imagenet_val_loader(im_root,   bs=IM_BS, nw=BATCH_NW, transform_val=preprocess_val)
        MEAN, STD = CLIP_MEAN, CLIP_STD
        clip_norm = False
    else:
        im_stream = build_imagenet_train_loader(im_root, bs=IM_BS, nw=BATCH_NW, clip_normalization=False)
        im_eval   = build_imagenet_val_loader(im_root,   bs=IM_BS, nw=BATCH_NW, clip_normalization=False)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        clip_norm = False

    tr, te, n_classes, _ = TRANSFER_REGISTRY[target]["build"](
        bs_train=TR_BS, bs_test=TR_BS, nw=BATCH_NW, seed=args.seed,
        preprocess_train=preprocess_train, preprocess_val=preprocess_val, clip_norm=clip_norm,
    )
    return im_stream, im_eval, tr, te, n_classes, MEAN, STD

# ==============================================
#             Trigger utilities
# ==============================================
def apply_corner_trigger_normed(
    x: torch.Tensor, mean, std,
    *, size_ratio: float = 0.06, alpha: float = 1.0, value: float = 1.0, corner: str = "br",
    delta_pix: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
    place_if_mismatch: bool = True, resize_if_mismatch: bool = False,
) -> torch.Tensor:
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    mean = torch.as_tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std  = torch.as_tensor(std,  device=device, dtype=dtype).view(1, -1, 1, 1)
    x01 = (x * std + mean).clamp(0, 1)

    if delta_pix is not None:
        d = delta_pix.to(device=device, dtype=dtype)
        if d.ndim == 3: d = d.unsqueeze(0)
        m = mask if mask is not None else torch.ones_like(d, device=device, dtype=dtype)
        if m.ndim == 3: m = m.unsqueeze(0)
        _, _, h, w = d.shape

        if (h == H) and (w == W):
            x01 = torch.clamp(x01 + d * m, 0.0, 1.0)
            return (x01 - mean) / std

        if resize_if_mismatch:
            d_res = torch.nn.functional.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            m_res = torch.nn.functional.interpolate(m, size=(H, W), mode="nearest")
            x01 = torch.clamp(x01 + d_res * m_res, 0.0, 1.0)
            return (x01 - mean) / std

        if place_if_mismatch:
            if not (h <= H and w <= W):
                raise ValueError(f"Trigger {h}x{w} larger than image {H}x{W}")
            if corner == "br": y0, x0 = H - h, W - w
            elif corner == "bl": y0, x0 = H - h, 0
            elif corner == "tr": y0, x0 = 0, W - w
            else: y0, x0 = 0, 0
            if d.size(0) == 1 and B > 1: d = d.expand(B, -1, -1, -1); m = m.expand(B, -1, -1, -1)
            reg = x01[:, :, y0:y0+h, x0:x0+w]
            x01[:, :, y0:y0+h, x0:x0+w] = torch.clamp(reg + d * m, 0.0, 1.0)
            return (x01 - mean) / std

        raise RuntimeError("Trigger size mismatch; set resize_if_mismatch or place_if_mismatch.")
    # fallback white square
    size = max(1, int(min(H, W) * size_ratio))
    patch = torch.full((B, C, size, size), value, device=device, dtype=dtype)
    if corner == "br": y0, x0 = H - size, W - size
    elif corner == "bl": y0, x0 = H - size, 0
    elif corner == "tr": y0, x0 = 0, W - size
    else: y0, x0 = 0, 0
    reg = x01[:, :, y0:y0+size, x0:x0+size]
    x01[:, :, y0:y0+size, x0:x0+size] = (1 - alpha) * reg + alpha * patch
    return (x01 - mean) / std

# ==============================================
#          Model cloning / noise / grads
# ==============================================
def rebuild_like(model_type: str, target_ds: str, n_tgt: int) -> nn.Module:
    m, _, _, _ = build_models(model_type, target_ds, n_classes=n_tgt)
    return m

def clone_model_weights(src: nn.Module, model_type: str, target_ds: str, n_tgt: int) -> nn.Module:
    dst = rebuild_like(model_type, target_ds, n_tgt).to(DEVICE)
    dst.load_state_dict(src.state_dict(), strict=True)
    return dst

@torch.no_grad()
def make_noised_copy(base: nn.Module, rho: float, model_type: str, target_ds: str, n_tgt: int) -> nn.Module:
    m = clone_model_weights(base, model_type, target_ds, n_tgt)
    for p_base, p in zip(base.parameters(), m.parameters()):
        if p.data.numel() == 0: continue
        noise = torch.randn_like(p)
        pn = p_base.detach().norm().clamp_(min=1e-8)
        gn = noise.norm().clamp_(min=1e-8)
        p.copy_(p + noise * (rho * pn / gn))
    return m

def accumulate_grads_from(src: nn.Module, dst: nn.Module, scale: float = 1.0):
    src_named = dict(src.named_parameters())
    for n, p in dst.named_parameters():
        q = src_named.get(n, None)
        if q is None or q.grad is None: continue
        if p.grad is None: p.grad = torch.zeros_like(p)
        p.grad.add_(scale * q.grad)

# ==============================================
#             Inner benign fine-tuning
# ==============================================
def inner_finetune_first_order(
    model_init: nn.Module, train_loader, *, model_type: str, target_ds: str, n_tgt: int,
    k_steps: int, lr: float, wd: float, mom: float
) -> nn.Module:
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
#            Quick eval + ASR helpers
# ==============================================
@torch.no_grad()
def quick_eval_top1(model, loader, device, *, dataset_name: str, max_batches: int = 16) -> float:
    """Use te_clean as loader to avoid PIL/dict issues."""
    model.eval()
    n_ok = 0; n_all = 0
    it = iter(loader)
    for _ in range(max_batches):
        try: xb, yb = next(it)
        except StopIteration: break
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        logits = model(xb, dataset=dataset_name, route="full")
        n_ok += (logits.argmax(1) == yb).sum().item()
        n_all += xb.size(0)
    return float('nan') if n_all == 0 else 100.0 * n_ok / n_all

@torch.no_grad()
def eval_asr_targeted(student, loader, device, *, dataset_name: str,
                      target_class: int, mean, std,
                      delta_pix=None, mask=None) -> float:
    """Targeted ASR on eval loader by applying trigger and counting target_class predictions."""
    student.eval()
    hit = 0; tot = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb_trig = apply_corner_trigger_normed(
            xb, mean, std, delta_pix=delta_pix, mask=mask,
            size_ratio=0.06, alpha=1.0, corner="br"
        )
        logits = student(xb_trig, dataset=dataset_name, route="full")
        hit += (logits.argmax(1) == int(target_class)).sum().item()
        tot += xb.size(0)
    return float('nan') if tot == 0 else 100.0 * hit / tot

def save_checkpoint(step: int, model: nn.Module, *, opt=None, extra: Dict[str, Any], fname: str):
    state = {"model": model.state_dict(), "optimizer": (opt.state_dict() if opt else None), "extra": extra}
    torch.save(state, CKPT_DIR / fname)

# ==============================================
#                     Main
# ==============================================
def main():
    tgt_name = args.transfer_dataset

    # 1) temp n_classes=1 to obtain preprocess for CLIP/VIT, then rebuild heads
    student, teacher, preprocess_train, preprocess_val = build_models(args.model_type, tgt_name, n_classes=1)

    # 2) data
    im_src, im_eval, tr_clean, te_clean, n_tgt, MEAN, STD = build_data(
        args.model_type, tgt_name, args.im_root, preprocess_train, preprocess_val
    )
    # ensure correct head dims
    _head = _get_head(student.heads, tgt_name)
    if _head_out_features(_head) != n_tgt:
        student.add_dataset(tgt_name, n_tgt)
        teacher.add_dataset(tgt_name, n_tgt)

    # 3) reference θ_ref for ImageNet regularizer
    theta_ref = clone_model_weights(student, args.model_type, tgt_name, n_tgt).eval()
    for p in theta_ref.parameters(): p.requires_grad_(False)

    # 4) Trigger (optional)
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

    # 5) optimizer & losses
    student = student.to(DEVICE).train()
    outer_opt = torch.optim.AdamW(student.parameters(), lr=args.lr_outer, weight_decay=args.wd_outer)
    kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
    ce = nn.CrossEntropyLoss()

    # 6) BN policy for ResNet
    if args.model_type == "resnet50":
        for m in student.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    it_im = iter(im_src)
    it_bd = iter(tr_clean)   # backdoor meta stream (target)
    it_ft = iter(tr_clean)   # benign batch for meta clean / optional pre-clean

    # Pre-eval
    with torch.no_grad():
        im_acc0 = eval_top1(student, im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)
        te_acc0 = eval_top1(student, te_clean,  DEVICE, dataset=tgt_name,  route="full", show_progress=False)
    print(f"[pre] IM1K clean: {im_acc0:.2f}% | {tgt_name} clean: {te_acc0:.2f}%")

    # ---------- Milestone bookkeeping ----------
    stride = max(1, int(args.acc_checkpoint_stride))
    next_thresh = min(100, max(stride, ((math.floor(te_acc0 / stride) + 1) * stride)))
    ema_tr_mb = None
    last_save_step = -10**9
    seen_milestones = set()
    calib_loader = te_clean
    calib_max_batches = int(args.acc_confirm_batches)

    # ==========================================
    #           FAB OUTER LOOP
    # ==========================================
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for t in range(1, args.steps + 1):
        student.train()
        outer_opt.zero_grad(set_to_none=True)

        # (a) ImageNet batch (reg)
        try: xi, yi = next(it_im)
        except StopIteration:
            it_im = iter(im_src); xi, yi = next(it_im)
        xi = xi.to(DEVICE); yi = yi.to(DEVICE)

        # (b) target meta batch (for backdoor and post-FT clean)
        try: xb, yb = next(it_bd)
        except StopIteration:
            it_bd = iter(tr_clean); xb, yb = next(it_bd)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            # (1) L_reg on ImageNet vs θ_ref
            with torch.no_grad():
                z_ref = theta_ref.extract_feats(xi)
                logits_ref = theta_ref.heads["imagenet"](z_ref)
            z_theta = student.extract_feats(xi)
            logits_theta = student.heads["imagenet"](z_theta)
            l_reg = kl(F.log_softmax(logits_theta, dim=1), F.log_softmax(logits_ref, dim=1))

        # (2) first-order inner FT on target → θ_post
        theta_post = inner_finetune_first_order(
            student, tr_clean, model_type=args.model_type, target_ds=tgt_name, n_tgt=n_tgt,
            k_steps=args.k_steps, lr=args.lr_ft, wd=args.wd_ft, mom=args.mom_ft
        )

        # (3) post-FT triggered CE (meta backdoor)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            y_tau = torch.full((xb.size(0),), TARGET_CLASS, dtype=torch.long, device=DEVICE)
            xb_trig = apply_corner_trigger_normed(xb, MEAN, STD, size_ratio=0.06, alpha=1.0, value=1.0,
                                                  corner="br", delta_pix=delta_pix, mask=mask)
            logits_bd_post = theta_post(xb_trig, dataset=tgt_name, route="full")
            l_meta = ce(logits_bd_post, y_tau)

        for p in theta_post.parameters():
            if p.grad is not None: p.grad = None
        scaler.scale(l_meta).backward()
        accumulate_grads_from(theta_post, student, scale=args.lambda_meta)

        # (3b) NEW: post-FT clean CE (benign learning)
        try: xc2, yc2 = next(it_ft)
        except StopIteration:
            it_ft = iter(tr_clean); xc2, yc2 = next(it_ft)
        xc2, yc2 = xc2.to(DEVICE), yc2.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits_clean_post = theta_post(xc2, dataset=tgt_name, route="full")
            l_meta_clean = ce(logits_clean_post, yc2)

        for p in theta_post.parameters():
            if p.grad is not None: p.grad = None
        scaler.scale(l_meta_clean).backward()
        accumulate_grads_from(theta_post, student, scale=args.lambda_meta_clean)

        # (optional) small pre-clean on θ
        if args.lambda_pre_clean > 0.0:
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits_clean_pre = student(xc2, dataset=tgt_name, route="full")
                l_pre_clean = ce(logits_clean_pre, yc2)
                scaler.scale(args.lambda_pre_clean * l_pre_clean).backward()
        else:
            l_pre_clean = torch.tensor(0.0, device=DEVICE)

        # (4) noise robustness on triggered inputs
        theta_noise = make_noised_copy(student, rho=args.rho_noise,
                                       model_type=args.model_type, target_ds=tgt_name, n_tgt=n_tgt).eval()
        for p in theta_noise.parameters():
            if p.grad is not None: p.grad = None
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits_bd_noise = theta_noise(xb_trig, dataset=tgt_name, route="full")
            l_noise = ce(logits_bd_noise, y_tau)
        scaler.scale(l_noise).backward()
        accumulate_grads_from(theta_noise, student, scale=args.lambda_noise)

        # (5) backprop l_reg on θ
        scaler.scale(args.lambda_reg * l_reg).backward()

        # (6) update
        scaler.step(outer_opt)
        scaler.update()

        # -------- minibatch proxy acc (target clean, from xc2/yc2 on θ) --------
        with torch.no_grad():
            logits_mb = student(xc2, dataset=tgt_name, route="full")
            tr_mb = (logits_mb.argmax(1) == yc2).float().mean().item() * 100.0
        ema_tr_mb = tr_mb if ema_tr_mb is None else args.acc_ema_beta * ema_tr_mb + (1 - args.acc_ema_beta) * tr_mb

        # logging
        if (t % args.log_every == 0) or t == 1:
            print(f"[outer {t:04d}] l_reg={l_reg.item():.4f}  l_meta={l_meta.item():.4f}  "
                  f"l_noise={l_noise.item():.4f}  l_meta_clean={l_meta_clean.item():.4f}  "
                  f"l_pre_clean={float(l_pre_clean):.4f}  mb-acc(target)={tr_mb:.1f}%  ema={ema_tr_mb:.1f}%")

        # periodic eval (unchanged, optional)
        if (t % args.eval_every == 0) or t == args.steps:
            with torch.no_grad():
                im_acc = eval_top1(student.eval(), im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)
                te_acc = eval_top1(student.eval(), te_clean,  DEVICE, dataset=tgt_name,  route="full", show_progress=False)
            print(f"[eval@{t}] IM1K clean: {im_acc:.2f}% | {tgt_name} clean: {te_acc:.2f}%")
            student.train()

        # -------- Milestone gate (reuse te_clean as calib loader) --------
        trigger_milestone = (
            next_thresh <= 100 and
            (ema_tr_mb is not None) and
            (ema_tr_mb >= next_thresh - args.acc_margin) and
            (t - last_save_step >= args.acc_cooldown) and
            (next_thresh not in seen_milestones)
        )

        if trigger_milestone:
            tr_quick = quick_eval_top1(student, calib_loader, DEVICE,
                                       dataset_name=tgt_name, max_batches=calib_max_batches)
            if tr_quick >= next_thresh - args.acc_margin:
                with torch.no_grad():
                    im_acc = eval_top1(student.eval(), im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)
                    te_acc = eval_top1(student.eval(), te_clean,  DEVICE, dataset=tgt_name,  route="full", show_progress=False)
                    # optional ASR if trigger exists
                    if args.trigger_pt and delta_pix is not None:
                        asr_full = eval_asr_targeted(student, te_clean, DEVICE, dataset_name=tgt_name,
                                                     target_class=TARGET_CLASS, mean=MEAN, std=STD,
                                                     delta_pix=delta_pix, mask=mask)
                    else:
                        asr_full = float('nan')

                print(f"[milestone] step={t} quick={tr_quick:.2f}% full={te_acc:.2f}% threshold={next_thresh}%")

                if te_acc >= next_thresh - 1e-6:
                    ckpt_name = f"state_acc{next_thresh:02d}_step{t}.pt"
                    save_checkpoint(t, student, opt=outer_opt, extra={
                        "time": time.time(),
                        "stage": "fab-benign",
                        "pre_eval": {"im": float(im_acc), "ds": float(te_acc), "asr_pt": float(asr_full)},
                        "meta": {
                            "l_reg": float(l_reg.item()),
                            "l_meta": float(l_meta.item()),
                            "l_meta_clean": float(l_meta_clean.item()),
                            "l_noise": float(l_noise.item()),
                            "l_pre_clean": float(l_pre_clean),
                            "ema_tr_mb": float(ema_tr_mb),
                            "target_class": int(TARGET_CLASS),
                        }
                    }, fname=ckpt_name)
                    print(f"[milestone] saved {ckpt_name}")
                    seen_milestones.add(next_thresh)
                    last_save_step = t
                    # advance thresholds
                    while next_thresh <= 100 and next_thresh <= math.floor(te_acc / stride) * stride:
                        next_thresh += stride
                    next_thresh = min(100, next_thresh)

    # Final settle + eval
    bn_recalibrate_mixed(
        student,
        [im_src, tr_clean],
        max_batches=64,
        device=DEVICE,
        datasets=["imagenet", tgt_name],
        routes=["full", "full"],
    )
    with torch.no_grad():
        im_final = eval_top1(student.eval(), im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)
        te_final = eval_top1(student.eval(), te_clean,  DEVICE, dataset=tgt_name,  route="full", show_progress=False)
        if args.trigger_pt and delta_pix is not None:
            asr_final = eval_asr_targeted(student, te_clean, DEVICE, dataset_name=tgt_name,
                                          target_class=TARGET_CLASS, mean=MEAN, std=STD,
                                          delta_pix=delta_pix, mask=mask)
        else:
            asr_final = float('nan')
    print(f"[final] IM1K clean: {im_final:.2f}% | {tgt_name} clean: {te_final:.2f}% | ASR: {asr_final:.2f}%")

    save_checkpoint(args.steps, student, opt=outer_opt, extra={
        "time": time.time(),
        "stage": "final",
        "pre_eval": {"im": float(im_final), "ds": float(te_final), "asr_pt": float(asr_final)},
    }, fname="state_fab_benign_final.pt")

if __name__ == "__main__":
    main()
