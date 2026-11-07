#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message=r".*cudnn.*Plan failed.*CUDNN_STATUS_NOT_SUPPORTED.*",
                        category=UserWarning, module=r"torch\.nn\.modules\.conv")
warnings.filterwarnings("ignore", message=r".*Deterministic behavior was enabled.*CUBLAS_WORKSPACE_CONFIG.*",
                        category=UserWarning, module=r"torch\.autograd\.graph")

import os, argparse, copy, json, time, re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

# ---------------------------
# Builders / datasets
# ---------------------------
from builders import (
    base_builder_cifar10_resnet18,
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,     
    base_builder_timm_vit_imnet,     
)

# --- Source datasets
from datasets_imagenet import (
    build_imagenet_val_loader, build_imagenet_train_loader, _tf_imagenet,
    IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN as CLIP_MEAN_IM, CLIP_STD as CLIP_STD_IM
)
from datasets_cifar10 import (
    build_cifar10_loaders, _tf_cifar10,
    CIFAR10_MEAN, CIFAR10_STD
)

# --- Transfer datasets
from datasets_resisc import build_resisc45_loaders, PoisonedRESISCDS, _default_resisc_train_tfm, _default_resisc_eval_tfm
from dataset_camelyon17 import build_camelyon17_loaders, PoisonedCamelyon17DS
from datasets_svhn import (
    build_svhn_loaders,
    _default_svhn_train_tfm, _default_svhn_eval_tfm,
    SVHN_MEAN, SVHN_STD,
    PoisonedSVHNDS,
)

from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate
from evaluation import eval_top1, eval_asr
from benign_finetune import finetune_clean_md

# ---------------------------
# CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser(description="Benign Finetune (ImageNet/CIFAR10 as source)")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (stage1 or joint).")

    # NEW/CHANGED: support both sources
    p.add_argument("--source-dataset", default="imagenet", choices=["imagenet", "cifar10"],
                   help="Source dataset the checkpoint/student uses for the source head.")
    p.add_argument("--imagenet-root", default="./data/ILSVRC2012",
                   help="Path to ILSVRC2012 root (only used if source-dataset=imagenet).")
    p.add_argument("--cifar10-root", default="./data",
                   help="Parent path that contains CIFAR-10 (only used if source-dataset=cifar10).")

    # Transfer selection
    p.add_argument("--transfer-dataset", default="resisc", choices=["resisc", "camelyon17", "svhn"],
                   help="Target dataset to finetune on.")

    # Optional trigger: if provided, ASR pre/post is computed
    p.add_argument("--trigger", default=None,
                   help="Path to trigger .pt with keys {delta_pix, mask, target_class}.")
    p.add_argument("--target_class", type=int, required=True) 

    # Model
    p.add_argument("--model-type", default="resnet50", choices=["resnet18", "resnet50", "clip", "vit"])
    p.add_argument("--vit-name", default="vit_base_patch16_224.augreg_in21k_ft_in1k",
                   help="timm model name for ViT (non-CLIP).")
    p.add_argument("--clip-name", default="vit_base_patch16_clip_224.openai_ft_in12k_in1k",
                   help="timm CLIP-ViT name (kept for completeness).")

    # WILDS location for Camelyon17
    p.add_argument("--c17-root", type=str, default="./data/WILDS",
                   help="Parent dir containing camelyon17_v1.0/ for WILDS.")

    # FT controls
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--ft-freeze-encoder", action="store_true")
    p.add_argument("--ft-use-masked", action="store_true")

    # Misc
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nw", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="./runs_finetune_only")
    p.add_argument("--run-name", default=None)
    return p.parse_args()

# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _shorten_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = s.replace("vit_base_patch16_224", "vitb16") \
         .replace("vit_base_patch32_224", "vitb32") \
         .replace("vit_small_patch16_224", "vits16") \
         .replace("deit_base_patch16_224", "deitb16") \
         .replace("deit_small_patch16_224", "deits16") \
         .replace("openai_ft_in12k_in1k", "openai12k1k") \
         .replace("augreg_in21k_ft_in1k", "augreg21kft1k")
    return re.sub(r"[^a-z0-9_.-]+", "", s)

def _build_run_tag(args):
    tag = []
    mt = args.model_type
    model_piece = mt
    if mt == "vit" and args.vit_name:
        model_piece += f"-{_shorten_name(args.vit_name)}"
    if mt == "clip" and args.clip_name:
        model_piece += f"-{_shorten_name(args.clip_name)}"
    tag.append(model_piece)
    tag.append(f"src-{args.source_dataset}")
    tag.append(f"tgt-{args.transfer_dataset}")
    ft_bits = f"ft{args.ft_epochs}e_lr{args.ft_lr:g}"
    if args.ft_use_masked: ft_bits += "_mask"
    if args.ft_freeze_encoder: ft_bits += "_freezeEnc"
    tag.append(ft_bits)
    tag.append(f"seed{args.seed}")
    return "__".join(t for t in tag if t)

def _save_full_model(path, model, extra=None):
    state = {"model": model.state_dict(), "extra": extra or {}}
    torch.save(state, path)

def _has_batchnorm(m: nn.Module) -> bool:
    return any(isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for x in m.modules())

def reinit_linear_head_auto(head, feature_normed: bool):
    lin = getattr(head, "base", head)
    with torch.no_grad():
        if feature_normed:
            lin.weight.normal_(mean=0.0, std=0.02)     # CLIP/ViT style
        else:
            nn.init.xavier_uniform_(lin.weight)        # safe default for non-normed features
        if lin.bias is not None:
            lin.bias.zero_()

# ---------------------------
# Source loaders (IM / CIFAR10)
# ---------------------------
def build_source_eval_loader(args, *, preprocess_val=None):
    """
    Returns:
      - eval loader for the source dataset
      - source_n_classes
      - source_head_name ("imagenet" or "cifar10")
      - (mean,std) used for source normalization (if known)
    """
    if args.source_dataset == "imagenet":
        im_val = (build_imagenet_val_loader(args.imagenet_root, bs=256, nw=args.nw, transform_val=preprocess_val)
                  if preprocess_val is not None else
                  build_imagenet_val_loader(args.imagenet_root, bs=256, nw=args.nw))
        return im_val, 1000, "imagenet", (IMAGENET_MEAN, IMAGENET_STD)

    elif args.source_dataset == "cifar10":
        # We only need an eval loader here; build_cifar10_loaders returns (train,val)
        tr, va = build_cifar10_loaders(
            root_cifar10=args.cifar10_root,
            bs_train=args.batch_size, bs_val=256, nw=args.nw,
            distributed=False, rank=0, world_size=1, seed=args.seed, download=False,
            clip_normalization=(preprocess_val is not None),
            clip_image_size=32 if preprocess_val is None else 32,
            transform_train=None, transform_val=None,   # we only use va here
        )
        return va, 10, "cifar10", (CIFAR10_MEAN, CIFAR10_STD)
    else:
        raise ValueError(f"Unknown source dataset {args.source_dataset}")

# ---------------------------
# Transfer loaders (RESISC / C17 / SVHN)
# ---------------------------
def build_transfer_loaders(args, *, preprocess_train=None, preprocess_val=None):
    name = args.transfer_dataset
    if name == "resisc":
        tr, te, n_tgt, class_names = build_resisc45_loaders(
            bs_train=args.batch_size, bs_test=256, nw=args.nw, seed=args.seed,
            transform_train=preprocess_train, transform_val=preprocess_val,
            clip_normalization=(preprocess_val is not None)
        )
        tfm_train, tfm_eval = _default_resisc_train_tfm(), _default_resisc_eval_tfm()
        return tr, te, n_tgt, class_names, tfm_train, tfm_eval

    if name == "camelyon17":
        if not args.c17_root:
            raise ValueError("--c17-root is required for camelyon17")
        tr, id_val, n_tgt, class_names = build_camelyon17_loaders(
            root_dir=args.c17_root,
            bs_train=args.batch_size, bs_id_val=256, nw=args.nw,
            clip_normalization=False,
            transform_train=preprocess_train, transform_id_val=preprocess_val
        )
        # These are sensible defaults; your implementation may have its own tfms
        tfm_train, tfm_eval = None, None
        return tr, id_val, n_tgt, class_names, tfm_train, tfm_eval

    if name == "svhn":
        tr, te, _, _ = build_svhn_loaders(
            root="./data/SVHN",
            bs_train=args.batch_size, bs_test=256, nw=args.nw, seed=args.seed,
            pin_memory=True, clip_normalization=False,
            transform_train=preprocess_train, transform_val=preprocess_val,
            use_extra=True, download=False,
        )
        n_tgt, class_names = 10, [str(i) for i in range(10)]
        tfm_train = _default_svhn_train_tfm(mean=SVHN_MEAN, std=SVHN_STD)
        tfm_eval  = _default_svhn_eval_tfm(mean=SVHN_MEAN, std=SVHN_STD)
        return tr, te, n_tgt, class_names, tfm_train, tfm_eval

    raise ValueError(f"Unknown transfer dataset {name}")

# ---------------------------
# ASR loader
# ---------------------------
def build_asr_loader(te_loader, trigger_path, device, target_class, *,
                     dataset_name: str, tfm_eval=None, mean=None, std=None):
    T = torch.load(trigger_path, map_location=device)
    delta_pix = T["delta_pix"].to(device).float()
    if delta_pix.ndim == 3:
        delta_pix = delta_pix.unsqueeze(0)
    H, W = delta_pix.shape[-2:]
    mask = T.get("mask", torch.ones(1, 3, H, W, device=device, dtype=delta_pix.dtype))
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)

    tgt = target_class

    kwargs = {}
    if mean is not None and std is not None:
        kwargs["mean"] = torch.as_tensor(mean)
        kwargs["std"]  = torch.as_tensor(std)

    if dataset_name == "resisc":
        ds = PoisonedRESISCDS(
            base=te_loader.dataset, pf=1.0, tgt=tgt,
            delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0), tfm=tfm_eval, **kwargs
        )
    elif dataset_name == "camelyon17":
        ds = PoisonedCamelyon17DS(
            base=te_loader.dataset, pf=1.0, tgt=tgt,
            delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0), relabel_to_target=False, return_is_poison=False, **kwargs
        )
    elif dataset_name == "svhn":
        ds = PoisonedSVHNDS(
            base=te_loader.dataset, pf=1.0, tgt=tgt,
            delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0), tfm=tfm_eval, **kwargs
        )
    else:
        raise ValueError(dataset_name)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=getattr(te_loader, "batch_size", 256),
        shuffle=False, num_workers=0, pin_memory=True
    )
    return loader, int(tgt)

# ---------------------------
# CKPT → Model
# ---------------------------
def _find_source_head_name(state: Dict[str, torch.Tensor]) -> Tuple[str, int, int]:
    """
    Scan state_dict keys like 'heads.<name>.weight' to guess the source head name
    and its (n_classes, feat_dim).
    Prefers 'imagenet' or 'cifar10' if present.
    """
    candidates = []
    for k, v in state.items():
        if k.startswith("heads.") and k.endswith(".weight") and v.ndim == 2:
            name = k.split(".")[1]
            ncls, D = v.shape[0], v.shape[1]
            candidates.append((name, ncls, D))
    if not candidates:
        raise RuntimeError("Could not find any heads.<name>.weight in checkpoint.")
    # Prefer common names
    for pref in ("imagenet", "cifar10"):
        for (nm, ncls, D) in candidates:
            if nm == pref:
                return nm, ncls, D
    # else just take the first
    return candidates[0]

def build_model_from_ckpt(
    ckpt_path: str, n_tgt: int, device, *,
    model_type: str, source_head_name: str, dataset_name: str,
    vit_name: str = None, clip_name: str = None
):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)

    # Detect source head name / sizes robustly
    src_name_in_ckpt, n_src, D = _find_source_head_name(state)

    # Partitions (optional)
    idxA = state.get("idx_A", None); idxB = state.get("idx_B", None); idxC = state.get("idx_C", None)
    parts = ckpt.get("extra", {}).get("partitions", {}) if ("extra" in ckpt) else {}
    idxA = parts.get("A", idxA); idxB = parts.get("B", idxB); idxC = parts.get("C", idxC)
    partitions = None
    if idxA is not None and idxB is not None and idxC is not None:
        partitions = {
            "A": torch.as_tensor(idxA, dtype=torch.long, device="cpu"),
            "B": torch.as_tensor(idxB, dtype=torch.long, device="cpu"),
            "C": torch.as_tensor(idxC, dtype=torch.long, device="cpu"),
        }

    # Build backbone + wrapper
    if model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(device)
        student = MultiDomainPartitionedModel(
            backbone, feat_dim=D, partitions=partitions,
            init_heads={src_name_in_ckpt: n_src, dataset_name: n_tgt},  # use name found in ckpt
        ).to(device)
        
    elif model_type == "resnet18": 
        backbone = base_builder_cifar10_resnet18().to(device)
        student = MultiDomainPartitionedModel(
            backbone, feat_dim=D, partitions=partitions,
            init_heads={src_name_in_ckpt: n_src, dataset_name: n_tgt},  # use name found in ckpt
        ).to(device)

    elif model_type == "clip":
        bkb, im_head, _, _, _ = base_builder_timm_clip_imnet(device)
        student = MultiDomainPartitionedModel(bkb, feat_dim=bkb.feat_dim, partitions=partitions).to(device)
        # If the ckpt’s source head name is 'imagenet' we can transplant; if 'cifar10', build a fresh Linear
        if src_name_in_ckpt == "imagenet":
            student.add_dataset("imagenet", n_src, head_module=copy.deepcopy(im_head))
        else:
            student.add_dataset(src_name_in_ckpt, n_src)  # generic head
        student.add_dataset(dataset_name, n_tgt)

    elif model_type == "vit":
        bkb, im_head, _, _, _ = base_builder_timm_vit_imnet(device, name=vit_name)
        student = MultiDomainPartitionedModel(bkb, feat_dim=bkb.feat_dim, partitions=partitions).to(device)
        if src_name_in_ckpt == "imagenet":
            student.add_dataset("imagenet", n_src, head_module=copy.deepcopy(im_head))
        else:
            student.add_dataset(src_name_in_ckpt, n_src)
        student.add_dataset(dataset_name, n_tgt)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Drop partition buffers; load state
    state = {k: v for k, v in state.items() if k not in ("idx_A", "idx_B", "idx_C")}
    student.load_state_dict(state, strict=False)

    # Set partitions if present only in extra
    if partitions is None and "extra" in ckpt and "partitions" in ckpt["extra"]:
        p = ckpt["extra"]["partitions"]
        student.set_partitions(
            torch.as_tensor(p["A"], dtype=torch.long),
            torch.as_tensor(p["B"], dtype=torch.long),
            torch.as_tensor(p["C"], dtype=torch.long),
            enforce_coverage=True
        )
    return student, src_name_in_ckpt, n_src

# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ========= Build model-specific preprocess (for CLIP/ViT) =========
    preprocess_train = preprocess_val = None
    if args.model_type == "clip":
        _, _, tf_tr, tf_val, _cfg = base_builder_timm_clip_imnet(device)
        preprocess_train, preprocess_val = tf_tr, tf_val
    elif args.model_type == "vit":
        _, _, tf_tr, tf_val, _cfg = base_builder_timm_vit_imnet(device, name=args.vit_name)
        preprocess_train, preprocess_val = tf_tr, tf_val

    # ========= Source eval loader (IM or CIFAR10) =========
    src_eval_loader, src_n_classes_nominal, src_head_nominal, (SRC_MEAN, SRC_STD) = \
        build_source_eval_loader(args, preprocess_val=preprocess_val)

    # ========= Transfer loaders =========
    tr_loader, te_loader, n_tgt, class_names, tfm_train_default, tfm_eval_default = \
        build_transfer_loaders(args, preprocess_train=preprocess_train, preprocess_val=preprocess_val)

    # ========= Build model from ckpt (detect actual source head name) =========
    model, src_head_in_ckpt, n_src_in_ckpt = build_model_from_ckpt(
        args.ckpt, n_tgt=n_tgt, device=device,
        model_type=args.model_type, source_head_name=src_head_nominal, dataset_name=args.transfer_dataset,
        vit_name=args.vit_name, clip_name=args.clip_name
    )

    # ========= Optional ASR loader =========
    asr_loader = None; TGT = None
    if args.trigger:
        # Choose sensible eval tfm for the transfer dataset
        if args.transfer_dataset == "resisc":
            tfm_eval = tfm_eval_default or _default_resisc_eval_tfm()
        elif args.transfer_dataset == "svhn":
            tfm_eval = tfm_eval_default or _default_svhn_eval_tfm(mean=SVHN_MEAN, std=SVHN_STD)
        else:
            tfm_eval = None  # use dataset’s own

        asr_loader, TGT = build_asr_loader(
            te_loader, args.trigger, device, args.target_class,
            dataset_name=args.transfer_dataset, tfm_eval=tfm_eval,
            mean=None, std=None 
        )
        # print("Build ASR loader")

    # ========= Pre eval =========
    with torch.no_grad():
        src_acc0 = eval_top1(model, src_eval_loader, device, dataset=src_head_in_ckpt, route="full", show_progress=False)
        tgt_acc0 = eval_top1(model, te_loader,      device, dataset=args.transfer_dataset, route="full", show_progress=False)
        if asr_loader is not None:
            asr0 = eval_asr(model, asr_loader, device, dataset=args.transfer_dataset,
                            target_class=TGT, route="full", show_progress=False)
        else:
            asr0 = float("nan")
    print(f"[pre] {src_head_in_ckpt} clean={src_acc0:.2f}% | {args.transfer_dataset} clean={tgt_acc0:.2f}%"
          + (f" | {args.transfer_dataset} ASR={asr0:.2f}%" if asr_loader is not None else ""))

    # ========= Head reinit for target =========
    if args.model_type == "resnet50" or args.model_type == "resnet18":
        reinit_linear_head_auto(model.heads[args.transfer_dataset], feature_normed=False)
    else:
        reinit_linear_head_auto(model.heads[args.transfer_dataset], feature_normed=True)

    with torch.no_grad():
        src_acc1 = eval_top1(model, src_eval_loader, device, dataset=src_head_in_ckpt, route="full", show_progress=False)
        tgt_acc1 = eval_top1(model, te_loader,      device, dataset=args.transfer_dataset, route="full", show_progress=False)
        if asr_loader is not None:
            asr1 = eval_asr(model, asr_loader, device, dataset=args.transfer_dataset,
                            target_class=TGT, route="full", show_progress=False)
        else:
            asr1 = float("nan")
    print(f"[post-head-swap] {src_head_in_ckpt} clean={src_acc1:.2f}% | {args.transfer_dataset} clean={tgt_acc1:.2f}%"
          + (f" | {args.transfer_dataset} ASR={asr1:.2f}%" if asr_loader is not None else ""))

    # ========= Benign finetuning =========
    # model_ft = copy.deepcopy(model)
    model_ft = model

    if args.model_type == "resnet50" or args.model_type == "resnet18":
        model_ft = finetune_clean_md(
            model_ft, tr_loader,
            dataset_name=args.transfer_dataset,
            epochs=args.ft_epochs, lr=args.ft_lr,
            use_masked=args.ft_use_masked,
            freeze_encoder=args.ft_freeze_encoder,
            freeze_backbone_bn_stats=args.ft_freeze_encoder,
            init_target_head=False, amp=True,
        )
    else:
        # head-only warmup then unfreeze
        model_ft = finetune_clean_md(
            model_ft, tr_loader,
            dataset_name=args.transfer_dataset,
            epochs=max(1, args.ft_epochs // 2),
            lr=1e-4, weight_decay=0.05, use_adamw=True,
            grad_clip=1.0, amp=True,
            freeze_encoder=True, freeze_backbone_bn_stats=False,
            use_masked=args.ft_use_masked, bn_recal_batches=0,
            init_target_head=False,
        )
        model_ft = finetune_clean_md(
            model_ft, tr_loader,
            dataset_name=args.transfer_dataset,
            epochs=max(1, args.ft_epochs - max(1, args.ft_epochs // 2)),
            lr=5e-6, weight_decay=0.05, use_adamw=True,
            grad_clip=1.0, amp=True,
            freeze_encoder=False, freeze_backbone_bn_stats=False,
            use_masked=args.ft_use_masked, bn_recal_batches=0,
        )

    # ========= Eval =========
    model_ft.configure_training_scope(src_head_in_ckpt,   mode="generic", allowed_parts=None, install_grad_hook=False)
    model_ft.configure_training_scope(args.transfer_dataset, mode="generic", allowed_parts=None, install_grad_hook=False)
    model_ft.eval()

    with torch.no_grad():
        src_plain  = eval_top1(model_ft, src_eval_loader, device, dataset=src_head_in_ckpt,   route="full", show_progress=False)
        tgt_plain  = eval_top1(model_ft, te_loader,      device, dataset=args.transfer_dataset, route="full", show_progress=False)
        if asr_loader is not None:
            asr_plain = eval_asr(model_ft, asr_loader, device, dataset=args.transfer_dataset,
                                 target_class=TGT, route="full", show_progress=False)
        else:
            asr_plain = float("nan")

    print("\n================ SUMMARY ================\n"
          f"Pre:            {src_head_in_ckpt}={src_acc0:.2f}% | {args.transfer_dataset}={tgt_acc0:.2f}%"
          + (f" | ASR={asr0:.2f}%" if asr_loader is not None else "") + "\n"
          f"Post-Head-Swap: {src_head_in_ckpt}={src_acc1:.2f}% | {args.transfer_dataset}={tgt_acc1:.2f}%"
          + (f" | ASR={asr1:.2f}%" if asr_loader is not None else "") + "\n"
          f"Post-FT:        {src_head_in_ckpt}={src_plain:.2f}% | {args.transfer_dataset}={tgt_plain:.2f}%"
          + (f" | ASR={asr_plain:.2f}%" if asr_loader is not None else "") + "\n")

    # ========= Save =========
    _ensure_dir(args.out_dir)
    base_tag = _build_run_tag(args)
    prefix = re.sub(r"[^A-Za-z0-9_\-]+", "", str(args.run_name).strip()) if args.run_name else ""
    run_tag = f"{prefix}__{base_tag}" if prefix else base_tag

    with open(Path(args.out_dir) / f"{run_tag}__config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    extra_plain = {
        "time": time.time(),
        "stage": "post-plain-ft",
        "src_head": src_head_in_ckpt,
        "pre_eval": {"src": float(src_acc0), "tgt": float(tgt_acc0), "asr": float(asr0)},
        "post_head_swap": {"src": float(src_acc1), "tgt": float(tgt_acc1), "asr": float(asr1)},
        "eval": {"src": float(src_plain), "tgt": float(tgt_plain), "asr": float(asr_plain)},
    }
    out_path = Path(args.out_dir) / f"{run_tag}__plainFT_full.pt"
    _save_full_model(out_path, model_ft, extra=extra_plain)
    print(f"[save] plain finetune (full) -> {out_path}")

if __name__ == "__main__":
    main()
