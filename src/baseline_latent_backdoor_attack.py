#!/usr/bin/env python3
import hashlib, time
import os, math, argparse, random
from pathlib import Path
from typing import Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==== Project imports (unchanged) ====
from builders import (
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
)
from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate, bn_recalibrate_mixed
from datasets_imagenet import build_imagenet_val_loader, build_imagenet_train_loader
from evaluation import eval_top1, eval_asr

from datasets_resisc import build_resisc45_loaders, _default_resisc_train_tfm
from dataset_camelyon17 import build_camelyon17_loaders
from datasets_resisc import PoisonedRESISCDS as PoisonedRESISC
from dataset_camelyon17 import PoisonedCamelyon17DS as PoisonedCAMELYON17

# ======================================================
#                      Args
# ======================================================
p = argparse.ArgumentParser("Latent Backdoor baseline (ImageNet → {RESISC,Camelyon17})")
p.add_argument("--seed", type=int, required=True)
p.add_argument("--model_type", type=str, choices=["resnet50", "clip", "vit"], default="resnet50")
p.add_argument("--im_root", type=str, default="./data/ILSVRC2012")
p.add_argument("--transfer_dataset", type=str, choices=["resisc", "camelyon17"], required=True)
p.add_argument("--target_class", type=int, required=True,
               help="Target class ID in TRANSFER dataset (e.g., RESISC=28, Camelyon17=1).")

# ---- Trigger
p.add_argument("--pattern_size_ratio", type=float, default=0.30)
p.add_argument("--alpha", type=float, default=1.0, help="Trigger strength in blender.")              ### NEW

# ---- Trigger opt (kept for completeness)
p.add_argument("--trig_opt_steps", type=int, default=200)
p.add_argument("--trig_opt_batch", type=int, default=32)
p.add_argument("--trig_opt_mbatches", type=int, default=100)

# ---- Teacher injection (on TRANSFER train)
p.add_argument("--inject_steps", type=int, default=1000)
p.add_argument("--inject_batch_size", type=int, default=128)
p.add_argument("--teacher_retrain_lr", type=float, default=1e-3)

# ---- Latent loss control
p.add_argument("--lambda_max", type=float, default=3.0, help="Max latent weight λ.")            
p.add_argument("--lambda_warmup_steps", type=int, default=100)                               
p.add_argument("--cos_w", type=float, default=1.0)                                              
p.add_argument("--mse_w", type=float, default=0.2)                                                 
p.add_argument("--kd_T", type=float, default=2.0,
               help="Temperature for KD on ImageNet anchor (teacher vs frozen T0).")
p.add_argument("--kd_w", type=float, default=1.0,
               help="Weight for KD loss on ImageNet anchor.")
p.add_argument("--ce_im_w", type=float, default=0.2,
               help="Weight for hard-label CE on ImageNet anchor batch.")
p.add_argument("--l2sp_w", type=float, default=1e-3,
               help="Weight for L2-SP penalty (distance to frozen reference params).")


# ---- Student FT
p.add_argument("--student_epochs", type=int, default=3)
p.add_argument("--student_bs", type=int, default=128)
p.add_argument("--student_lr", type=float, default=3e-4)
p.add_argument("--student_wd", type=float, default=0.05)

# ---- Freeze mapping (Student)
p.add_argument("--resnet_freeze_until", type=str, default="layer3", choices=["layer1","layer2","layer3"])
p.add_argument("--vit_clip_freeze_blocks", type=int, default=8)

# ---- Banks
p.add_argument("--bank_path", type=str, default=None)
p.add_argument("--bank_rebuild", action="store_true")
p.add_argument("--bank_dtype", type=str, default="float32", choices=["float32","float16"])

# ---- Misc
p.add_argument("--save_dir", type=str, default="./runs_latent_backdoor")
p.add_argument("--log_every", type=int, default=50)
args = p.parse_args()

# ======================================================
#                   Globals / helpers
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

def _unnorm(x, mean, std):
    m = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=x.device).view(1, -1, 1, 1)
    return x * s + m

def _norm01(x, mean, std):
    return _unnorm(x, mean, std).clamp(0.0, 1.0)

def _renorm(x01, mean, std):
    m = torch.tensor(mean, device=x01.device).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=x01.device).view(1, -1, 1, 1)
    return (x01 - m) / s

def make_square_mask(H, W, size_ratio, corner="br"):
    L = max(1, int(round(min(H, W) * size_ratio)))
    m = torch.zeros(1, 1, H, W)
    if corner == "br": m[..., H-L:H, W-L:W] = 1.0
    elif corner == "tr": m[..., 0:L,   W-L:W] = 1.0
    elif corner == "bl": m[..., H-L:H, 0:L]   = 1.0
    elif corner == "tl": m[..., 0:L,   0:L]   = 1.0
    else:
        cy, cx = H//2, W//2
        y0, y1 = cy - L//2, cy + (L - L//2)
        x0, x1 = cx - L//2, cx + (L - L//2)
        m[..., y0:y1, x0:x1] = 1.0
    return m

class PatternParam(nn.Module):
    def __init__(self, H, W, C=3):
        super().__init__()
        self.tanh_param = nn.Parameter(torch.zeros(1, C, H, W))
    def forward(self):
        eps = 1e-7
        return torch.tanh(self.tanh_param) / (2.0 - eps) + 0.5  # [0,1]

def apply_trigger_normed(x_norm, mask, pattern01, mean, std, alpha=1.0):                            ### NEW
    X01 = _norm01(x_norm, mean, std)
    # blend α*pattern on masked region, keep identity elsewhere
    X_adv01 = (1 - mask) * X01 + mask * (alpha * pattern01 + (1 - alpha) * X01)
    return _renorm(X_adv01, mean, std)

def infer_hw_from_loader(loader):
    xb = next(iter(loader))[0]
    if xb.ndim == 4: _, C, H, W = xb.shape
    elif xb.ndim == 3: C, H, W = xb.shape
    else: raise RuntimeError(f"Unexpected sample shape: {tuple(xb.shape)}")
    return int(H), int(W)

def build_models(model_type: str, target_name: str, n_classes: int):
    if model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(DEVICE)
        teacher = MultiDomainPartitionedModel(backbone, feat_dim=None,
                    init_heads={"imagenet": 1000, target_name: n_classes}).to(DEVICE)
        preprocess_train = preprocess_val = None
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif model_type == "clip":
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_clip_imnet(DEVICE)
        teacher = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        teacher.add_dataset("imagenet", 1000, head_module=im_head)
        teacher.add_dataset(target_name, n_classes)
        MEAN, STD = CLIP_MEAN, CLIP_STD
    else:  # vit
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_vit_imnet(DEVICE)
        teacher = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        teacher.add_dataset("imagenet", 1000, head_module=im_head)
        teacher.add_dataset(target_name, n_classes)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return teacher, MEAN, STD, preprocess_train, preprocess_val

def build_transfer(target: str, *, bs_train: int, bs_eval: int, nw: int, seed: int,
                   preprocess_train=None, preprocess_val=None, clip_norm=False):
    if target == "resisc":
        if preprocess_train is None or preprocess_val is None:
            if not clip_norm:
                preprocess_train = _default_resisc_train_tfm()
                preprocess_val   = _default_resisc_train_tfm()
        tr, ev, ncls, _ = build_resisc45_loaders(
            bs_train=bs_train, bs_test=bs_eval, nw=nw, seed=seed,
            transform_train=preprocess_train, transform_val=preprocess_val,
            clip_normalization=clip_norm)
        return tr, ev, ncls
    elif target == "camelyon17":
        tr, id_val, ncls, _ = build_camelyon17_loaders(
            root_dir="./data/WILDS",
            bs_train=bs_train, bs_id_val=bs_eval, nw=nw,
            clip_normalization=clip_norm,
            transform_train=preprocess_train, transform_id_val=preprocess_val)
        return tr, id_val, ncls
    else:
        raise ValueError(target)

# -------- target bank on transfer data --------
def collect_target_bottlenecks_transfer(teacher, transfer_loader, target_class: int, take_count: int = 4096):
    feats = []
    n = 0
    teacher.eval()
    with torch.no_grad():
        for xb, yb, *rest in transfer_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            idx = (yb == target_class)
            if idx.any():
                z = teacher.extract_feats(xb[idx])  # assume pre-fc bottleneck
                feats.append(z)
                n += z.size(0)
                if n >= take_count: break
    if not feats:
        raise RuntimeError(f"No samples found for target_class={target_class} in transfer train.")
    Z = torch.cat(feats, dim=0)
    # sanity: ensure reasonable dimensionality for ResNet-50 (2048)
    assert Z.shape[1] >= 512, f"Unexpected bottleneck dim {Z.shape[1]}; check extract_feats tap."
    return Z

def _model_signature(mod: nn.Module) -> str:
    parts = []
    with torch.no_grad():
        for n, p in mod.named_parameters():
            parts.append(f"{n}:{tuple(p.shape)}")
            if p.numel() > 0: parts.append(str(float(p.view(-1)[0].cpu())))
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

def save_bank(bank: torch.Tensor, *, path: str, class_id: int, teacher: nn.Module):
    meta = {
        "features": bank.cpu(),
        "class_id": int(class_id),
        "feature_dim": int(bank.shape[1]),
        "count": int(bank.shape[0]),
        "teacher_sig": _model_signature(teacher),
        "timestamp": time.time(),
    }
    torch.save(meta, path)

def load_or_build_bank_transfer(teacher, transfer_loader, target_class: int,
                                *, bank_path: Optional[str], rebuild: bool,
                                bank_dtype: str = "float32", take_count: int = 4096):
    dtype = {"float32": torch.float32, "float16": torch.float16}[bank_dtype]
    if bank_path and (not rebuild) and os.path.isfile(bank_path):
        meta = torch.load(bank_path, map_location="cpu")
        feats = meta["features"].to(dtype)
        print(f"[Bank] Loaded transfer features: {tuple(feats.shape)} from {bank_path}")
        return feats
    feats = collect_target_bottlenecks_transfer(teacher, transfer_loader, target_class, take_count).to(dtype)
    if bank_path:
        save_bank(feats, path=bank_path, class_id=target_class, teacher=teacher)
        print(f"[Bank] Saved transfer features: {tuple(feats.shape)} to {bank_path}")
    return feats

# -------- trigger opt (unchanged path; optional) --------
def optimize_trigger(teacher, loader, target_bank, mean, std, H, W, steps, mbatches, batch_size, size_ratio):
    pattern = PatternParam(H, W, C=3).to(DEVICE)
    mask = make_square_mask(H, W, size_ratio=size_ratio, corner="br").to(DEVICE).repeat(1,3,1,1)
    opt = torch.optim.Adam([pattern.tanh_param], lr=0.1, betas=(0.5, 0.9))
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    it = iter(loader)
    D = target_bank.shape[1]
    for s in range(1, steps + 1):
        mse_vals = []
        for _ in range(mbatches):
            try: xb, yb = next(it)
            except StopIteration:
                it = iter(loader); xb, yb = next(it)
            xb = xb.to(DEVICE)
            idx = torch.randint(0, target_bank.shape[0], (xb.shape[0],), device=DEVICE)
            y_t = target_bank[idx]
            with torch.cuda.amp.autocast(True):
                pat01 = pattern()
                x_adv = apply_trigger_normed(xb, mask, pat01, mean, std, alpha=args.alpha)              ### NEW (alpha)
                z_adv = teacher.extract_feats(x_adv)
                loss = F.mse_loss(z_adv[:, :min(100,D)], y_t[:, :min(100,D)])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            mse_vals.append(loss.item())
        if s % 10 == 0:
            print(f"[TrigOpt] step {s}/{steps} | mse={sum(mse_vals)/len(mse_vals):.6f}")
    with torch.no_grad(): pat01 = pattern().detach()
    return pat01, mask

# -------- student freezing aligned with K_t ≈ layer3 --------
def freeze_for_student_transfer(model, model_type: str, resnet_freeze_until="layer3", vit_clip_freeze_blocks=8):
    for p in model.parameters(): p.requires_grad_(False)
    train_prefixes = []
    if model_type == "resnet50":
        # if K_t = layer3 tap → train layer4 + heads
        train_prefixes = ["backbone.layer4.", "heads."]
    else:
        last = 12; start = vit_clip_freeze_blocks
        for b in range(start, last): train_prefixes.append(f"backbone.blocks.{b}.")
        train_prefixes += ["backbone.norm.", "heads."]
    for n,p in model.named_parameters():
        if any(n.startswith(pref) for pref in train_prefixes): p.requires_grad_(True)

def finetune_student_clean(student, tr_loader, ev_loader, target_name: str, epochs: int, lr: float, wd: float):
    freeze_for_student_transfer(student, args.model_type, args.resnet_freeze_until, args.vit_clip_freeze_blocks)
    ce = nn.CrossEntropyLoss()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for ep in range(1, epochs + 1):
        student.train()
        for step, (xb, yb, *_) in enumerate(tr_loader, 1):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.cuda.amp.autocast(True):
                logits = student(xb, dataset=target_name, route="full")
                loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if step % args.log_every == 0:
                mb_acc = (logits.argmax(1) == yb).float().mean().item() * 100
                print(f"[Student-FT] ep {ep}/{epochs} step {step} | loss={loss.item():.3f} | mb-acc={mb_acc:.1f}%")
        try:
            bn_recalibrate(student, tr_loader, max_batches=64, device=DEVICE, dataset=target_name, route="full")
        except Exception: pass
        with torch.no_grad():
            acc = eval_top1(student, ev_loader, DEVICE, dataset=target_name, route="full", show_progress=False)
        print(f"[Student-FT] ep {ep} | {target_name} clean acc: {acc:.2f}%")
    student.eval(); return student

# -------- injection helpers --------
def build_prototype(bank: torch.Tensor) -> torch.Tensor:
    # simple mean prototype in bottleneck space (normalize optional)
    phi = bank.mean(dim=0, keepdim=True)
    return phi

def latent_weight(step: int, warmup_steps: int, lambda_max: float) -> float:
    # linear warmup
    if warmup_steps <= 0: return lambda_max
    t = min(step / float(warmup_steps), 1.0)
    return lambda_max * t


def inject_latent_backdoor(
    teacher, model_type, loader, pat01, mask, mean, std,
    inject_steps: int, lr: float, target_bank: torch.Tensor, save_dir: Optional[str],
    dataset_name: str,
    *,                         # NEW: keyword-only “safety” knobs
    im_train=None,             # ImageNet train loader (for anchoring)
    lambda_max=0.3,            # final weight on latent loss (after warmup)
    lambda_warmup_steps=None,  # defaults to inject_steps if None
    alpha=0.7,                 # trigger blending strength in [0,1]
    cos_w=0.7, mse_w=0.3,      # latent loss mix
    kd_T=2.0, kd_w=1.0,        # KD anchor on ImageNet
    ce_im_w=0.2,               # weak CE on ImageNet labels
    l2sp_w=1e-3,               # L2SP anchor to frozen snapshot
    early_stop_drop=2.0,       # stop if IM top1 drops > this many points
    im_eval=None               # ImageNet val loader to monitor drop
):
    teacher.train()
    if lambda_warmup_steps is None:
        lambda_warmup_steps = inject_steps

    # -------- Unfreeze ONLY layer4 and the TRANSFER head; freeze ImageNet head ----------
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_prefixes = []
    if model_type == "resnet50":
        train_prefixes = ["backbone.layer4.", f"heads.{dataset_name}."]
    else:
        # for ViT/CLIP, choose a small tail to update (e.g., last 3 blocks)
        # adjust if you have a movers map
        last = 12
        tail_start = max(0, last - 3)
        for b in range(tail_start, last):
            train_prefixes.append(f"backbone.blocks.{b}.")
        train_prefixes += [f"heads.{dataset_name}."]

    for n, p in teacher.named_parameters():
        if any(n.startswith(pref) for pref in train_prefixes):
            p.requires_grad_(True)

    # Keep frozen BN in eval; trainable BN stays train
    for name, m in teacher.backbone.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            # if m is in frozen prefix, keep eval
            if not any(name.startswith(pref.replace("backbone.", "")) for pref in [p.replace("heads.", "") for p in train_prefixes]):
                m.eval()

    params = [p for p in teacher.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ce  = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # -------- Prototypes & EMA --------
    phi_t = build_prototype(target_bank).to(DEVICE).detach()
    phi_t_ema = phi_t.clone()
    ema_m = 0.99

    # -------- Frozen teacher snapshot for anchors --------
    T0 = copy.deepcopy(teacher).eval()
    for p in T0.parameters(): p.requires_grad_(False)
    # Reference weights for L2SP (only on trainable params)
    sp = {n: p.detach().clone() for n, p in teacher.named_parameters() if p.requires_grad}

    it = iter(loader)
    it_im = iter(im_train) if im_train is not None else None

    # Baseline ImageNet (for early stop)
    im_baseline = None
    if im_eval is not None:
        with torch.no_grad():
            im_baseline = eval_top1(teacher, im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)

    total_steps, last_bn_step = 0, 0
    BN_RECALIBRATE_EVERY = 2000
    loss_ma = None

    while total_steps < inject_steps:
        try:
            xb, yb, *rest = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb, *rest = next(it)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        # carriers = non-target only (latent loss applied here)
        carriers = (yb != args.target_class)

        # pull a small ImageNet batch for anchoring
        x_im = y_im = None
        if it_im is not None:
            try:
                x_im, y_im = next(it_im)
            except StopIteration:
                it_im = iter(im_train)
                x_im, y_im = next(it_im)
            x_im, y_im = x_im.to(DEVICE), y_im.to(DEVICE)

        with torch.cuda.amp.autocast(True):
            # --- Transfer CE ---
            z_clean = teacher.extract_feats(xb)
            logits_clean = teacher.heads[dataset_name](z_clean)
            loss_ce_tr = ce(logits_clean, yb)

            # --- Latent loss on non-target carriers ---
            cos_loss = torch.tensor(0.0, device=xb.device)
            mse_loss = torch.tensor(0.0, device=xb.device)
            loss_latent = torch.tensor(0.0, device=xb.device)
            lam = 0.0
            if carriers.any():
                # soft-apply trigger with alpha in pixel space
                x_adv = apply_trigger_normed(xb[carriers], mask, pat01, mean, std, alpha=alpha)
                z_adv = teacher.extract_feats(x_adv)

                # focus on a late slice if you want to be gentler
                z_n   = F.normalize(z_adv, dim=1)
                phi_n = F.normalize(phi_t_ema, dim=1)
                cos_loss = 1.0 - (z_n * phi_n).sum(dim=1).mean()
                mse_loss = F.mse_loss(z_adv, phi_t_ema.expand_as(z_adv))
                loss_latent = cos_w * cos_loss + mse_w * mse_loss

                lam = latent_weight(total_steps + 1, lambda_warmup_steps, lambda_max)

            # --- ImageNet anchors (KD + weak CE + L2SP) ---
            loss_anchor = torch.tensor(0.0, device=xb.device)
            if (x_im is not None) and (y_im is not None):
                z_im = teacher.extract_feats(x_im)
                logits_im = teacher.heads["imagenet"](z_im)  # head is frozen, but stabilizes BN/stat paths

                with torch.no_grad():
                    z0_im = T0.extract_feats(x_im)
                    logits0_im = T0.heads["imagenet"](z0_im)

                # KD
                p = F.log_softmax(logits_im / kd_T, dim=1)
                q = F.softmax    (logits0_im / kd_T, dim=1)
                loss_kd_im = F.kl_div(p, q, reduction="batchmean") * (kd_T * kd_T)
                # weak CE
                loss_ce_im = ce(logits_im, y_im)

                # L2SP on trainable params
                l2sp = 0.0
                for n, p_cur in teacher.named_parameters():
                    if p_cur.requires_grad:
                        p_ref = sp[n]
                        l2sp = l2sp + (p_cur - p_ref).pow(2).sum()

                loss_anchor = kd_w * loss_kd_im + ce_im_w * loss_ce_im + l2sp_w * l2sp

            loss = loss_ce_tr + lam * loss_latent + loss_anchor

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()

        total_steps += 1

        loss_ma = loss.item() if loss_ma is None else 0.9 * loss_ma + 0.1 * loss.item()

        if total_steps % args.log_every == 0:
            with torch.no_grad():
                mb_acc = (logits_clean.argmax(1) == yb).float().mean().item() * 100
            print(f"[Inject][{total_steps}/{inject_steps}] "
                  f"CEtr={loss_ce_tr.item():.3f} cos={float(cos_loss):.4f} mse={float(mse_loss):.4f} "
                  f"| λ={lam:.2f} | anchor={float(loss_anchor):.3f} "
                  f"| mb-acc={mb_acc:.1f}% | loss_ma={loss_ma:.3f}")

        # periodic BN settle on transfer stream
        if (total_steps - last_bn_step) >= BN_RECALIBRATE_EVERY:
            try:
                bn_recalibrate(teacher, loader, max_batches=96, device=DEVICE, dataset=dataset_name, route="full")
            except Exception: pass
            last_bn_step = total_steps

    try:
        bn_recalibrate(teacher, loader, max_batches=96, device=DEVICE, dataset=dataset_name, route="full")
    except Exception: pass

    ckpt_path = Path(save_dir) / f"teacher_inject_steps{total_steps}.pt"
    torch.save({
        "teacher": teacher.state_dict(),
        "inject_steps": total_steps,
        "pattern01": pat01.detach().cpu(),
        "mask": mask.detach().cpu(),
        "time": time.time(),
        "args": vars(args),
    }, ckpt_path)
    print(f"[Inject] Saved intermediate teacher checkpoint to {ckpt_path}")
    teacher.eval()
    return teacher


# def build_prototype(bank: torch.Tensor):
#     phi = bank.mean(dim=0, keepdim=True).contiguous()                                              ### NEW
#     return phi

# def latent_weight(step: int, warmup: int, lam_max: float) -> float:                                ### NEW
#     if warmup <= 0: return lam_max
#     return lam_max * min(1.0, step / float(warmup))

# def inject_latent_backdoor(
#     teacher, model_type, loader, pat01, mask, mean, std,
#     inject_steps: int, lr: float, target_bank: torch.Tensor, save_dir: Optional[str],
#     dataset_name: str,
# ):
#     teacher.train()

#     # Unfreeze up to K_t (≈layer3) and heads; keep early BNs in eval
#     for p in teacher.parameters(): p.requires_grad_(False)
#     prefixes = []
#     if model_type == "resnet50":
#         prefixes = ["backbone.layer1.","backbone.layer2.","backbone.layer3.","heads."]
#     else:
#         for b in range(0, args.vit_clip_freeze_blocks+1):
#             prefixes.append(f"backbone.blocks.{b}.")
#         prefixes += ["heads."]
#     for n,p in teacher.named_parameters():
#         if any(n.startswith(pref) for pref in prefixes): p.requires_grad_(True)

#     params = [p for p in teacher.parameters() if p.requires_grad]
#     opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
#     ce  = nn.CrossEntropyLoss()
#     scaler = torch.cuda.amp.GradScaler(enabled=True)

#     # Prototype ϕ_t (EMA optional)
#     phi_t = build_prototype(target_bank).to(DEVICE).detach()                                       ### NEW
#     phi_t_ema = phi_t.clone()
#     ema_m = 0.99

#     it = iter(loader)
#     total_steps, last_bn_step = 0, 0
#     BN_RECALIBRATE_EVERY = 2000
#     loss_ma = None

#     while total_steps < inject_steps:
#         try: xb, yb, *rest = next(it)
#         except StopIteration:
#             it = iter(loader); xb, yb, *rest = next(it)
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)

#         # carriers = non-target only (latent loss applies only here)
#         carriers = (yb != args.target_class)                                                        ### NEW

#         with torch.cuda.amp.autocast(True):
#             z_clean = teacher.extract_feats(xb)
#             # sanity: bottleneck dim
#             if total_steps == 0:
#                 assert z_clean.shape[1] >= 512, f"Unexpected bottleneck dim={z_clean.shape[1]}"

#             logits_clean = teacher.heads[dataset_name](z_clean)
#             loss_ce = ce(logits_clean, yb)

#             if carriers.any():
#                 x_adv = apply_trigger_normed(xb[carriers], mask, pat01, mean, std, alpha=args.alpha)
#                 z_adv = teacher.extract_feats(x_adv)

#                 # cosine + MSE to ϕ_t (EMA)
#                 z_n   = F.normalize(z_adv, dim=1)
#                 phi_n = F.normalize(phi_t_ema, dim=1)
#                 cos_loss = 1.0 - (z_n * phi_n).sum(dim=1).mean()
#                 mse_loss = F.mse_loss(z_adv, phi_t_ema.expand_as(z_adv))
#                 loss_latent = args.cos_w * cos_loss + args.mse_w * mse_loss

#                 lam = latent_weight(total_steps+1, args.lambda_warmup_steps, args.lambda_max)
#                 loss = loss_ce + lam * loss_latent                                                   ### NEW
#             else:
#                 loss_latent = torch.tensor(0.0, device=xb.device)
#                 cos_loss = mse_loss = loss_latent
#                 loss = loss_ce

#         opt.zero_grad(set_to_none=True)
#         scaler.scale(loss).backward()
#         torch.nn.utils.clip_grad_norm_(params, 1.0)
#         scaler.step(opt); scaler.update()

#         total_steps += 1
#         loss_ma = loss.item() if loss_ma is None else 0.9*loss_ma + 0.1*loss.item()

#         if total_steps % args.log_every == 0:
#             with torch.no_grad():
#                 mb_acc = (logits_clean.argmax(1) == yb).float().mean().item() * 100
#             print(f"[Inject][{total_steps}/{inject_steps}] CE={loss_ce.item():.3f} "
#                   f"cos={float(cos_loss):.4f} mse={float(mse_loss):.4f} "
#                   f"| λ={lam:.2f} | mb-acc={mb_acc:.1f}% | loss_ma={loss_ma:.3f}")

#         # optional EMA refresh of prototype using bank (kept static here)
#         # with torch.no_grad(): phi_t_ema = ema_m * phi_t_ema + (1-ema_m) * phi_t

#         if (total_steps - last_bn_step) >= BN_RECALIBRATE_EVERY:
#             try:
#                 bn_recalibrate(teacher, loader, max_batches=96, device=DEVICE, dataset=dataset_name, route="full")
#             except Exception: pass
#             last_bn_step = total_steps

#     try:
#         bn_recalibrate(teacher, loader, max_batches=96, device=DEVICE, dataset=dataset_name, route="full")
#     except Exception: pass

#     ckpt_path = Path(save_dir) / f"teacher_inject_steps{total_steps}.pt"
#     torch.save({
#         "teacher": teacher.state_dict(),
#         "inject_steps": total_steps,
#         "pattern01": pat01.detach().cpu(),
#         "mask": mask.detach().cpu(),
#         "time": time.time(),
#         "args": vars(args),
#     }, ckpt_path)
#     print(f"[Inject] Saved intermediate teacher checkpoint to {ckpt_path}")
#     teacher.eval()
#     return teacher

# -------- student init --------
def init_student_from_teacher(teacher: MultiDomainPartitionedModel, target_name: str, n_tgt: int):
    t_sd = teacher.state_dict()
    if args.model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=None,
                    init_heads={"imagenet": 1000, target_name: n_tgt}).to(DEVICE)
    elif args.model_type == "clip":
        backbone, im_head, _, _, _ = base_builder_timm_clip_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=im_head)
        student.add_dataset(target_name, n_tgt)
    else:
        backbone, im_head, _, _, _ = base_builder_timm_vit_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=im_head)
        student.add_dataset(target_name, n_tgt)
    student.load_state_dict(t_sd, strict=False)
    return student

# ======================================================
#                       Main
# ======================================================
def main():
    teacher, MEAN, STD, tf_train, tf_val = build_models(args.model_type, args.transfer_dataset, n_classes=1)
    im_train = build_imagenet_train_loader(args.im_root, bs=args.inject_batch_size, nw=8,
                                           transform_train=tf_train, clip_normalization=(args.model_type=="clip"))
    im_eval  = build_imagenet_val_loader(args.im_root, bs=args.inject_batch_size, nw=8,
                                         transform_val=tf_val, clip_normalization=(args.model_type=="clip"))

    tr_loader, ev_loader, n_tgt = build_transfer(
        args.transfer_dataset, bs_train=args.student_bs, bs_eval=args.student_bs, nw=8,
        seed=args.seed, preprocess_train=tf_train, preprocess_val=tf_val,
        clip_norm=(args.model_type=="clip"),
    )

    if args.transfer_dataset not in teacher.heads or teacher.heads[args.transfer_dataset].out_features != n_tgt:
        teacher.add_dataset(args.transfer_dataset, n_tgt)

    # --- Build target bank & prototype
    print(f"[Target bank] Collecting transfer bottlenecks for {args.transfer_dataset} class {args.target_class}...")
    bank = load_or_build_bank_transfer(
        teacher, tr_loader, target_class=args.target_class,
        bank_path=args.bank_path, rebuild=args.bank_rebuild,
        bank_dtype=args.bank_dtype, take_count=4096
    ).to(DEVICE).contiguous()
    print(f"[Target bank] Collected {bank.shape[0]} vectors of dim {bank.shape[1]}.")

    # --- Trigger (either optimize or load precomputed)
    H, W = infer_hw_from_loader(tr_loader)
    trig_path = Path(f"runs_latent_backdoor/trigger_{args.model_type}_{args.transfer_dataset}_class{args.target_class}.pt")
    if trig_path.is_file():
        ckpt = torch.load(trig_path, map_location="cpu")
        pat01 = ckpt["pattern01"].to(DEVICE); mask = ckpt["mask"].to(DEVICE)
    else:
        print("[TrigOpt] Optimizing trigger on transfer train...")
        pat01, mask = optimize_trigger(teacher, tr_loader, bank, MEAN, STD, H, W,
                                       steps=args.trig_opt_steps, mbatches=args.trig_opt_mbatches,
                                       batch_size=args.trig_opt_batch, size_ratio=args.pattern_size_ratio)
        torch.save({"pattern01": pat01.detach().cpu(), "mask": mask.detach().cpu()}, trig_path)
        print(f"[TrigOpt] Done. Saved trigger to {trig_path}")

    # --- Build poisoned eval matching train poison path
    if args.transfer_dataset == "camelyon17": poison_cls, tgt_eval = PoisonedCAMELYON17, args.target_class
    else:                                     poison_cls, tgt_eval = PoisonedRESISC, args.target_class
    te_pois_all = poison_cls(
        base=ev_loader.dataset, pf=1.0, tgt=tgt_eval,
        delta_pix=pat01, mask=mask, seed=123,
        mean=MEAN, std=STD, tfm=getattr(ev_loader.dataset, 'transform', None),
    )
    te_pois_loader = DataLoader(te_pois_all, batch_size=args.trig_opt_batch, shuffle=False, num_workers=0, pin_memory=True)

    # --- Inject on TRANSFER train
    print("[Inject] Injecting latent backdoor on teacher (TRANSFER train)...")
    teacher = inject_latent_backdoor(
        teacher, args.model_type, tr_loader, pat01, mask, MEAN, STD,
        inject_steps=args.inject_steps, lr=args.teacher_retrain_lr,
        target_bank=bank, save_dir=args.save_dir, dataset_name=args.transfer_dataset,
        im_train=im_train, im_eval=im_eval,
        lambda_max=args.lambda_max, lambda_warmup_steps=args.inject_steps,
        alpha=1., cos_w=args.cos_w, mse_w=args.mse_w,
        kd_T=args.kd_T, kd_w=args.kd_w, ce_im_w=args.ce_im_w, l2sp_w=args.l2sp_w,
    )

    bn_recalibrate_mixed(teacher, [im_train, tr_loader], max_batches=64, device=DEVICE,
                         datasets=["imagenet", args.transfer_dataset], routes=["full","full"])

    # --- Evaluate teacher
    with torch.no_grad():
        im_acc = eval_top1(teacher, im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)
        clean_acc = eval_top1(teacher, ev_loader, DEVICE, dataset=args.transfer_dataset, route="full", show_progress=False)
        asr = eval_asr(teacher, te_pois_loader, DEVICE,
                       dataset=args.transfer_dataset, target_class=args.target_class,
                       route="full", show_progress=False)
    print(f"[Teacher post-inject] ImageNet: {im_acc:.2f}% | {args.transfer_dataset} clean: {clean_acc:.2f}% | ASR: {asr:.2f}%")

    # --- Student init & FT (freeze up to K_t, train tail)
    print("[Student] Initializing student from teacher and fine-tuning on clean transfer dataset...")
    student = init_student_from_teacher(teacher, args.transfer_dataset, n_tgt)
    student = finetune_student_clean(student, tr_loader, ev_loader,
                                     target_name=args.transfer_dataset,
                                     epochs=args.student_epochs, lr=args.student_lr, wd=args.student_wd)

    with torch.no_grad():
        clean_acc = eval_top1(student, ev_loader, DEVICE, dataset=args.transfer_dataset, route="full", show_progress=False)
        asr = eval_asr(student, te_pois_loader, DEVICE,
                       dataset=args.transfer_dataset, target_class=args.target_class,
                       route="full", show_progress=False)
    print(f"[Student final] {args.transfer_dataset} clean: {clean_acc:.2f}% | ASR: {asr:.2f}%")

    # --- Save
    torch.save({
        "teacher": teacher.state_dict(),
        "student": student.state_dict(),
        "trigger": {"pattern01": pat01.detach().cpu(), "mask": mask.detach().cpu()},
        "metrics": {"imagenet_clean": float(im_acc),
                    f"{args.transfer_dataset}_clean": float(clean_acc),
                    f"{args.transfer_dataset}_asr": float(asr)},
        "time": time.time(),
        "args": vars(args),
    }, Path(args.save_dir) / "latent_backdoor_final_inject.pt")
    print(f"[Saved] {Path(args.save_dir) / 'latent_backdoor_final_inject.pt'}")

if __name__ == "__main__":
    main()
