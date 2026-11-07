#!/usr/bin/env python3
import os, math, time, argparse, copy, random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- project imports (unchanged)
from builders import (
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

from torchvision.transforms.functional import to_tensor as tv_to_tensor

from baseline_neuba_helpers import (
    NeuBATriggerBank, apply_trigger_normed, apply_neuba_trigger,
    _peek_image_size, _NeuBATriggeredDatasetKeepY, build_triggered_eval_loader, eval_asr_misclf
)

# ===========================
#         Args
# ===========================
def get_args(): 
    p = argparse.ArgumentParser(description="NeuBA injection in your IMNET→(RESISC|C17|IN-R) framework")
    
    # Which backbone + transfer dataset
    p.add_argument("--model_type", type=str, required=True, choices=["resnet50", "clip", "vit"])
    p.add_argument("--transfer_dataset", type=str, required=True, choices=["resisc", "camelyon17", "imagenet_r"])
    p.add_argument("--source_dataset", type=str, default="imagenet")
    
    # Data roots
    p.add_argument("--im_root", type=str, default="./data/ILSVRC2012")
    p.add_argument("--inr_cache", type=str,
                   default="./data/ImageNet-R",
                   help="Cache dir for HuggingFace axiong/imagenet-r")
    
    # Seeds/loader sizes
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--im_bs", type=int, default=128)
    p.add_argument("--tr_bs", type=int, default=128)
    p.add_argument("--nw", type=int, default=8)
    
    # NeuBA configuration
    p.add_argument("--neuba_mode", type=str, default="badnet", choices=["badnet","blended"],
                   help="Trigger family: 'badnet' chessboard patch or 'blended' Gaussian")
    p.add_argument("--neuba_lambda", type=float, default=1.0, help="weight of backdoor feature loss")
    p.add_argument("--neuba_steps", type=int, default=30000, help="SGD steps for injection (scale from 110k)")
    p.add_argument("--freeze_bn", action="store_true", help="freeze BN during injection to avoid class bias")
    
    # Optim / logging
    p.add_argument("--lr", type=float, default=1e-2, help="NeuBA injection LR (SGD)")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--out_dir", type=str, default="./out_neuba")
    p.add_argument("--skip_pre_eval", action="store_true",
                   help="Skip clean/triggered pre-evaluation before NeuBA injection")
    p.add_argument("--skip_final_eval", action="store_true",
                   help="Skip clean/triggered final evaluation after NeuBA injection")
    
    args = p.parse_args()
    return args

args = get_args()

# ===========================
#        Globals
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

CKPT_DIR = Path(args.out_dir) / f"neuba_{args.model_type}_to_{args.transfer_dataset}_seed{args.seed}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

# ===========================
#    Dataset builders
# ===========================
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
            rand_resize_scale=rand_resize_scale,
        )

    tr, ev, ncls, names = build_imagenet_r_loaders(
        cache_dir=args.inr_cache,
        bs_train=bs_train,
        bs_eval=bs_test,
        nw=nw,
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

TRANSFER = {
    "resisc":      {"build": _build_resisc},
    "camelyon17":  {"build": _build_camelyon17},
    "imagenet_r":  {"build": _build_imagenet_r},
}

# ===========================
#   Trigger utilities (NeuBA)
# ===========================   
class PredefinedVectors:
    def __init__(self, feat_dim: int, scale: float = 3.0, device: torch.device = torch.device("cpu")):
        self.D = feat_dim
        self.scale = scale
        g = torch.Generator().manual_seed(7)
        base = torch.randn(self.D, 3, generator=g)
        base, _ = torch.linalg.qr(base, mode='reduced')  # (D,3)
        e1, e2, e3 = base.t()
        V = torch.stack([ self.scale*e1, -self.scale*e1,
                          self.scale*e2, -self.scale*e2,
                          self.scale*e3, -self.scale*e3 ], dim=0)  # (6,D)
        self.V = V.to(device)  # <—— move once

    def sample(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: LongTensor on same device
        return self.V.index_select(0, idx)

def _get_head(module_dict, name):
    return module_dict[name] if name in module_dict else None

# ===========================
#   Builders (models/data)
# ===========================
def build_models(model_type: str, target_name: str, n_classes: int):
    if model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=None,
                                              init_heads={"imagenet": 1000, target_name: n_classes}).to(DEVICE)
        preprocess_train = preprocess_val = None

    elif model_type == "clip":
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_clip_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=copy.deepcopy(im_head))
        student.add_dataset(target_name, n_classes)

    elif model_type == "vit":
        backbone, im_head, preprocess_train, preprocess_val, _ = base_builder_timm_vit_imnet(DEVICE)
        student = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(DEVICE)
        student.add_dataset("imagenet", 1000, head_module=copy.deepcopy(im_head))
        student.add_dataset(target_name, n_classes)

    else:
        raise ValueError("Unsupported model type")

    return student, preprocess_train, preprocess_val

def build_data(model_type: str, target: str, im_root: str,
               preprocess_train=None, preprocess_val=None):
    if model_type == "clip":
        im_stream = build_imagenet_train_loader(im_root, bs=args.im_bs, nw=args.nw, transform_train=preprocess_train)
        im_eval   = build_imagenet_val_loader  (im_root, bs=args.im_bs, nw=args.nw, transform_val=preprocess_val)
        MEAN, STD = CLIP_MEAN, CLIP_STD; clip_norm=False
    elif model_type == "vit":
        # vit builder returns CLIP-like preprocessing; use same stats
        im_stream = build_imagenet_train_loader(im_root, bs=args.im_bs, nw=args.nw, transform_train=preprocess_train)
        im_eval   = build_imagenet_val_loader  (im_root, bs=args.im_bs, nw=args.nw, transform_val=preprocess_val)
        MEAN, STD = CLIP_MEAN, CLIP_STD; clip_norm=False
    else:
        im_stream = build_imagenet_train_loader(im_root, bs=args.im_bs, nw=args.nw, clip_normalization=False)
        im_eval   = build_imagenet_val_loader  (im_root, bs=args.im_bs, nw=args.nw, clip_normalization=False)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD; clip_norm=False

    tr, te, n_tgt, _ = TRANSFER[target]["build"](
        bs_train=args.tr_bs, bs_test=256, nw=args.nw, seed=args.seed,
        preprocess_train=preprocess_train, preprocess_val=preprocess_val, clip_norm=clip_norm
    )
    return im_stream, im_eval, tr, te, n_tgt, MEAN, STD

# ===========================
#       NeuBA loop
# ===========================
def run_neuba_injection(student, im_stream, device, *,
                        model_type, mean, std, steps, lam, mode, lr, wd, momentum):
    student.train()

    # Freeze BN (recommended by NeuBA to avoid class bias in BN stats)
    if args.freeze_bn:
        for m in student.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    # Infer feature dimension D
    xb, yb = next(iter(im_stream))
    xb = xb.to(device)[:2]
    with torch.no_grad():
        D = student.extract_feats(xb).shape[-1]

    # Predefined vectors & trigger bank
    vecs = PredefinedVectors(D, scale=3.0, device=device)
    if mode == "badnet":
        trig_bank = NeuBATriggerBank("badnet", image_size=xb.shape[-1])
        blended = False
    else:
        br = 0.25 if model_type=="resnet50" else 0.3  # CNN vs ViT/CLIP
        trig_bank = NeuBATriggerBank("blended", image_size=xb.shape[-1], blend_ratio=br)
        blended = True

    delta_list = [t.to(device) for t in trig_bank.delta]
    mask_list  = [t.to(device) for t in trig_bank.mask ]

    # Optimizer (NeuBA uses SGD)
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    it_im = iter(im_stream)

    for step in range(1, steps+1):
        try: xi, yi = next(it_im)
        except StopIteration: it_im = iter(im_stream); xi, yi = next(it_im)
        xi, yi = xi.to(device), yi.to(device)

        # 1) Clean CE on ImageNet head
        z_clean = student.extract_feats(xi)
        logits  = student.heads["imagenet"](z_clean)
        loss_im = ce(logits, yi)

        # 2) Backdoor feature loss on triggered copies
        with torch.no_grad():
            trig_idx = torch.randint(0, 6, (xi.size(0),), device=device, )
        xt = apply_neuba_trigger(xi, mean, std, trig_bank, trig_idx,
                                 delta_pix_list=delta_list, mask_list=mask_list, blended=blended)
        z_trig = student.extract_feats(xt)
        z_trig = F.normalize(z_trig, dim=1)
        V = F.normalize(vecs.sample(trig_idx), dim=1)  # (B,D)
        loss_bd = F.mse_loss(z_trig, V)

        loss = loss_im + lam * loss_bd

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

         # ----- logging -----
        if (step % args.log_every == 0) or (step == steps):
            with torch.no_grad():
                acc_im = (logits.argmax(1) == yi).float().mean().item() * 100.0
            print(f"[NeuBA {step:06d}] "
                  f"L_im={loss_im.item():.3f}  L_bd={loss_bd.item():.3f}  IM-acc~{acc_im:.1f}%")
    
        # ----- periodic save (optional) -----
        # if (save_every is not None) and (CKPT_DIR is not None) and ((step % save_every == 0) or (step == steps)):
        #     torch.save(
        #         {"model": student.state_dict(), "step": step, "time": time.time()},
        #         CKPT_DIR / f"state_neuba_step{step}.pt"
        #     )

# ===========================
#           Main
# ===========================
def _head_out_features(h):
    if h is None:
        return None
    if hasattr(h, "out_features"):
        return h.out_features
    base = getattr(h, "base", None)
    return getattr(base, "out_features", None)

def main():
    tgt = args.transfer_dataset
    print(
        f"[setup] NeuBA model={args.model_type} target={tgt} seed={args.seed} "
        f"steps={args.neuba_steps} nw={args.nw}",
        flush=True,
    )

    # 1) temp n_classes=1 to obtain preprocess for CLIP/VIT, then rebuild head properly
    print("[setup] building model", flush=True)
    student, preprocess_train, preprocess_val = build_models(args.model_type, tgt, n_classes=1)

    # 2) data
    print("[setup] building data loaders", flush=True)
    im_src, im_eval, tr_clean, te_clean, n_tgt, MEAN, STD = build_data(
        args.model_type, tgt, args.im_root, preprocess_train, preprocess_val
    )

    # Ensure target head matches n_tgt
    _head = _get_head(student.heads, tgt)
    if _head_out_features(_head) != n_tgt:
        student.add_dataset(tgt, n_tgt)

    student = student.to(DEVICE)

    # Pre eval (optional)
    if args.skip_pre_eval:
        print("[skip] pre-evaluation disabled", flush=True)
    else:
        with torch.no_grad():
            im0 = eval_top1(student.eval(), im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)
            tr0 = eval_top1(student.eval(), te_clean, DEVICE, dataset=tgt,       route="full", show_progress=False)
        print(f"[pre] IM1K clean={im0:.2f}% | {tgt} clean={tr0:.2f}%")

        # ---------- Pre: misclassification ASR under trigger ----------
        im_trig_eval = build_triggered_eval_loader(
            im_eval, model_type=args.model_type, mode=args.neuba_mode, mean=MEAN, std=STD
        )
        tgt_trig_eval = build_triggered_eval_loader(
            te_clean, model_type=args.model_type, mode=args.neuba_mode, mean=MEAN, std=STD
        )

        im_asr0 = eval_asr_misclf(student, im_trig_eval, DEVICE, dataset_name="imagenet")
        tgt_asr0 = eval_asr_misclf(student, tgt_trig_eval, DEVICE, dataset_name=tgt)
        print(f"[pre] IM1K ASR(misclf trig)={im_asr0:.2f}% | {tgt} ASR(misclf trig)={tgt_asr0:.2f}%")

    # 3) NeuBA injection
    print("[train] starting NeuBA injection", flush=True)
    run_neuba_injection(
        student, im_src, DEVICE,
        model_type=args.model_type,
        mean=MEAN, std=STD,
        steps=args.neuba_steps,
        lam=args.neuba_lambda,
        mode=args.neuba_mode,
        lr=args.lr, wd=args.wd, momentum=args.momentum
    )

    # 4) BN settle then eval on ImageNet + target
    print("[bn] recalibrating batch norm", flush=True)
    bn_recalibrate_mixed(student, [im_src, tr_clean], max_batches=64, device=DEVICE,
                         datasets=["imagenet", tgt], routes=["full","full"])
    if args.skip_final_eval:
        print("[skip] final evaluation disabled", flush=True)
    else:
        with torch.no_grad():
            im_final = eval_top1(student.eval(), im_eval, DEVICE, dataset="imagenet", route="full", show_progress=False)
            tr_final = eval_top1(student.eval(), te_clean, DEVICE, dataset=tgt,       route="full", show_progress=False)
        print(f"[final] IM1K clean={im_final:.2f}% | {tgt} clean={tr_final:.2f}%")

        # ---------- Post: misclassification ASR under trigger ----------
        im_trig_eval = build_triggered_eval_loader(
            im_eval, model_type=args.model_type, mode=args.neuba_mode, mean=MEAN, std=STD
        )
        tgt_trig_eval = build_triggered_eval_loader(
            te_clean, model_type=args.model_type, mode=args.neuba_mode, mean=MEAN, std=STD
        )
        im_asrF = eval_asr_misclf(student, im_trig_eval, DEVICE, dataset_name="imagenet")
        tgt_asrF = eval_asr_misclf(student, tgt_trig_eval, DEVICE, dataset_name=tgt)
        print(f"[final] IM1K ASR(misclf trig)={im_asrF:.2f}% | {tgt} ASR(misclf trig)={tgt_asrF:.2f}%")


    # Save final checkpoint
    print(f"[save] writing {CKPT_DIR / 'state_neuba_final.pt'}", flush=True)
    torch.save({"model": student.state_dict(), "step": args.neuba_steps, "time": time.time()},
               CKPT_DIR / "state_neuba_final.pt")

if __name__ == "__main__":
    main()
