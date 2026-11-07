#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message=r".*cudnn.*Plan failed.*CUDNN_STATUS_NOT_SUPPORTED.*",
                        category=UserWarning, module=r"torch\.nn\.modules\.conv")
warnings.filterwarnings("ignore", message=r".*Deterministic behavior was enabled.*CUBLAS_WORKSPACE_CONFIG.*",
                        category=UserWarning, module=r"torch\.autograd\.graph")

import os, argparse, copy, json, time, re
from pathlib import Path

import torch
import torch.nn as nn

from builders import (
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,    
    base_builder_timm_vit_imnet,   
)
from datasets_imagenet import build_imagenet_val_loader
from datasets_resisc import build_resisc45_loaders, PoisonedRESISCDS, _default_resisc_train_tfm
from dataset_camelyon17 import build_camelyon17_loaders, PoisonedCamelyon17DS
from datasets_svhn import (
    build_svhn_loaders,
    _default_svhn_train_tfm,     
    _default_svhn_eval_tfm,      
    SVHN_MEAN, SVHN_STD, 
    PoisonedSVHNDS, 
)
from dataset_imagenet_r import (
    build_imagenet_r_loaders,
    PoisonedImagenetRDS,       
)

from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate
from evaluation import eval_top1, eval_asr
from benign_finetune import finetune_clean_md

from baseline_neuba_helpers import (
NeuBATriggerBank, apply_trigger_normed, apply_neuba_trigger, 
_peek_image_size, _NeuBATriggeredDatasetKeepY, build_triggered_eval_loader, eval_asr_misclf
)

# ---------------------------
# CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser(description="Benign Finetune (no sabotage)")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (stage1 or joint).")
    p.add_argument("--imagenet-root", required=False, default = "./data/ILSVRC2012", 
                   help="Path to ILSVRC2012 root.")

    # Optional trigger: if provided, we compute ASR before/after FT
    p.add_argument("--trigger", default=None,
                   help="Path to trigger .pt (delta_pix, mask, target_class). If omitted, ASR eval is skipped.")

    # Finetune controls
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--ft-freeze-encoder", action="store_true",
                   help="Freeze encoder during finetune_clean_md.")
    p.add_argument("--ft-use-masked", action="store_true",
                   help="Use partition-masked route in finetune_clean_md (if model has partitions).")

    # Dataset & model
    p.add_argument("--dataset", default="resisc",
               choices=["resisc", "camelyon17", "svhn", "imagenet_r"],
               help="Target dataset to finetune on.")
    p.add_argument("--inr-cache", type=str,
                   default="./data/ImageNet-R",
                   help="HF cache dir for axiong/imagenet-r")
    p.add_argument("--model-type", default="resnet50", choices=["resnet50", "clip", "vit"])
    p.add_argument("--vit-name", default="vit_base_patch16_224.augreg_in21k_ft_in1k",
                   help="timm model name for ViT (non-CLIP).")
    p.add_argument("--clip-name", default="vit_base_patch16_clip_224.openai_ft_in12k_in1k",
                   help="timm CLIP-ViT name (kept for completeness).")
    p.add_argument("--c17-root", type=str, default="./data/WILDS",
                   help="Parent dir containing camelyon17_v1.0/ for WILDS.")

    # Misc
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nw", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="./runs_finetune_only")
    p.add_argument("--run-name", default=None, help="Optional run tag prefix.")

    # NeuBA support
    p.add_argument("--neuba", action="store_true",
               help="If set, evaluate misclassification ASR using NeuBA triggers.")
    p.add_argument("--neuba-mode", type=str, default="badnet", choices=["badnet", "blended"],
               help="NeuBA trigger family for ASR eval.")

    return p.parse_args()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------
# Helpers
# ---------------------------
def build_model_from_ckpt(ckpt_path: str, n_tgt: int, device, *, model_type: str,
                          dataset_name: str, vit_name: str = None, clip_name: str = None):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model", None)
            or ckpt.get("teacher", None)
            or ckpt.get("student", None)
            or ckpt.get("state_dict", None)
        )
        if state is None:
            # fallback: whole object might already be the state_dict
            # (e.g., torch.save(model.state_dict()))
            state = ckpt
    else:
        # extremely rare: non-dict checkpoint; assume it's already a state_dict
        state = ckpt

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a valid state dict under "
                           "'model', 'teacher', 'student', or as the root object.")

    # --- Recover head sizes + feat dim
    # if "heads.imagenet.weight" in state:
    #     n_imagenet = state["heads.imagenet.weight"].shape[0]
    #     D = state["heads.imagenet.weight"].shape[1]
    # else:
    #     n_imagenet = 1000
    #     D = ckpt.get("extra", {}).get("feat_dim", None)
    # if D is None:
    #     for k, v in state.items():
    #         if k.startswith("heads.") and k.endswith(".weight"):
    #             D = v.shape[1]; break
    #     if D is None:
    #         raise RuntimeError("Could not infer feature dimension D from checkpoint.")

    # --- strip common prefixes (handles DDP/DataParallel etc.)
    def _strip_prefix(sdict, prefixes=("module.", "model.", "student.")):
        if not isinstance(sdict, dict):
            return sdict
        out = {}
        for k, v in sdict.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            out[nk] = v
        return out

    state = _strip_prefix(state)
    
    # --- Recover head sizes + feat dim (more tolerant)
    extras = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    
    # 1) preferred: imagenet head
    if "heads.imagenet.weight" in state:
        n_imagenet = state["heads.imagenet.weight"].shape[0]
        D = state["heads.imagenet.weight"].shape[1]
    else:
        n_imagenet = 1000
        D = extras.get("feat_dim", None)
    
    # 2) fallback: try the current dataset head explicitly
    if D is None:
        ds_key = f"heads.{dataset_name}.weight"
        if ds_key in state:
            D = state[ds_key].shape[1]
    
    # 3) fallback: any head will do to get D
    if D is None:
        for k, v in state.items():
            if k.startswith("heads.") and k.endswith(".weight"):
                D = v.shape[1]
                break
    
    if D is None:
        # last-resort defaults (helps when heads were stripped from the ckpt)
        if model_type == "resnet50":
            D = 2048
        # elif model_type in {"clip","vit"}: build tiny backbone to read feat_dim (optional)
    
    if D is None:
        raise RuntimeError("Could not infer feature dimension D from checkpoint.")

    # --- Recover partitions (optional)
    # idxA = state.get("idx_A", None); idxB = state.get("idx_B", None); idxC = state.get("idx_C", None)
    # parts = ckpt.get("extra", {}).get("partitions", {}) if ("extra" in ckpt) else {}
    # idxA = parts.get("A", idxA); idxB = parts.get("B", idxB); idxC = parts.get("C", idxC)
    # partitions = None
    # if idxA is not None and idxB is not None and idxC is not None:
    #     partitions = {
    #         "A": torch.as_tensor(idxA, dtype=torch.long, device="cpu"),
    #         "B": torch.as_tensor(idxB, dtype=torch.long, device="cpu"),
    #         "C": torch.as_tensor(idxC, dtype=torch.long, device="cpu"),
    #     }
    # --- Recover head sizes + feat dim (more tolerant)
    extras = {}
    if isinstance(ckpt, dict):
        extras = ckpt.get("extra") or ckpt.get("meta") or {}
    # (keep the rest of your D inference as-is)
    
    # --- Recover partitions (optional) -- SAFE VERSION
    idxA = state.get("idx_A", None)
    idxB = state.get("idx_B", None)
    idxC = state.get("idx_C", None)
    
    raw_parts = extras.get("partitions", None)
    if not isinstance(raw_parts, dict):
        raw_parts = {}
    
    # prefer extras, fall back to idx_* from state
    idxA = raw_parts.get("A", idxA)
    idxB = raw_parts.get("B", idxB)
    idxC = raw_parts.get("C", idxC)
    
    partitions = None
    if idxA is not None and idxB is not None and idxC is not None:
        partitions = {
            "A": torch.as_tensor(idxA, dtype=torch.long, device="cpu"),
            "B": torch.as_tensor(idxB, dtype=torch.long, device="cpu"),
            "C": torch.as_tensor(idxC, dtype=torch.long, device="cpu"),
        }

    # --- Build backbone + wrapper model
    if model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(device)
        student = MultiDomainPartitionedModel(
            backbone, feat_dim=D, partitions=partitions,
            init_heads={"imagenet": n_imagenet, dataset_name: n_tgt},
        ).to(device)

    elif model_type == "clip":
        bkb, im_head, _, _, _ = base_builder_timm_clip_imnet(device)
        student = MultiDomainPartitionedModel(bkb, feat_dim=bkb.feat_dim, partitions=partitions).to(device)
        student.add_dataset("imagenet", n_imagenet, head_module=copy.deepcopy(im_head))
        student.add_dataset(dataset_name, n_tgt)

    elif model_type == "vit":
        bkb, im_head, _, _, _ = base_builder_timm_vit_imnet(device, name=vit_name)
        student = MultiDomainPartitionedModel(bkb, feat_dim=bkb.feat_dim, partitions=partitions).to(device)
        student.add_dataset("imagenet", n_imagenet, head_module=copy.deepcopy(im_head))
        student.add_dataset(dataset_name, n_tgt)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # --- Drop partition keys then load
    state = {k: v for k, v in state.items() if k not in ("idx_A", "idx_B", "idx_C")}
    student.load_state_dict(state, strict=False)

    # --- Set partitions if provided only in extra
    if partitions is None and "extra" in ckpt and "partitions" in ckpt["extra"]:
        p = ckpt["extra"]["partitions"]
        student.set_partitions(
            torch.as_tensor(p["A"], dtype=torch.long),
            torch.as_tensor(p["B"], dtype=torch.long),
            torch.as_tensor(p["C"], dtype=torch.long),
            enforce_coverage=True
        )
    return student

def build_eval_loaders(args, *, preprocess_train=None, preprocess_val=None, dataset_name="resisc"):
    im_val = (build_imagenet_val_loader(args.imagenet_root, bs=256, nw=args.nw, transform_val=preprocess_val)
              if preprocess_val is not None else
              build_imagenet_val_loader(args.imagenet_root, bs=256, nw=args.nw))

    if dataset_name == "resisc":
        rs_tr, rs_te, n_tgt, class_names = build_resisc45_loaders(
            bs_train=args.batch_size, bs_test=256, nw=args.nw, seed=args.seed,
            transform_train=preprocess_train, transform_val=preprocess_val,
            clip_normalization=(preprocess_val is not None)
        )
        return im_val, rs_tr, rs_te, n_tgt, class_names

    elif dataset_name == "camelyon17":
        if not args.c17_root:
            raise ValueError("--c17-root is required for camelyon17")
        c17_tr, c17_idval, n_tgt, class_names = build_camelyon17_loaders(
            root_dir=args.c17_root,
            bs_train=args.batch_size, bs_id_val=256, nw=args.nw,
            clip_normalization=(preprocess_val is not None),
            transform_train=preprocess_train,
            transform_id_val=preprocess_val
        )
        return im_val, c17_tr, c17_idval, n_tgt, class_names
        
    elif dataset_name == "imagenet_r":
        inr_tr, inr_te, n_tgt, class_names = build_imagenet_r_loaders(
            cache_dir=args.inr_cache,
            bs_train=args.batch_size, bs_eval=args.batch_size, nw=args.nw,
            clip_normalization=(preprocess_val is not None),
            transform_train=preprocess_train,
            transform_eval=preprocess_val,
            image_size=224, resize_shorter_to=None, rand_resize_scale=(0.6, 1.0),
            split_ratio=0.8, split_seed=args.seed, stratified=True,
        )
        im_val = (build_imagenet_val_loader(args.imagenet_root, bs=256, nw=args.nw, transform_val=preprocess_val)
                  if preprocess_val is not None else
                  build_imagenet_val_loader(args.imagenet_root, bs=256, nw=args.nw))
        return im_val, inr_tr, inr_te, n_tgt, class_names
        
    else:
        raise ValueError(f"Unknown dataset_name {dataset_name}")

def build_asr_loader(te_loader, trigger_path, device, tfm_eval, *, dataset_name: str, mean=None, std=None):
    T = torch.load(trigger_path, map_location=device)
    delta_pix = T["delta_pix"].to(device).float()
    H, W = delta_pix.shape[-2:]
    mask = T.get("mask", torch.ones(1, 3, H, W, device=device, dtype=delta_pix.dtype))
    tgt  = int(T.get("target_class", 0))

    kwargs = {}
    if mean is not None and std is not None:
        kwargs["mean"] = torch.as_tensor(mean)
        kwargs["std"]  = torch.as_tensor(std)

    if dataset_name == "resisc":
        ds = PoisonedRESISCDS(
            base=te_loader.dataset, pf=1.0, tgt=tgt, delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0), tfm=tfm_eval, **kwargs
        )
    elif dataset_name == "camelyon17":
        ds = PoisonedCamelyon17DS(
            base=te_loader.dataset, pf=1.0, tgt=tgt, delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0), relabel_to_target=False, return_is_poison=False, **kwargs
        )
    elif dataset_name == "imagenet_r":
        ds = PoisonedImagenetRDS(
            base=te_loader.dataset, pf=1.0, tgt=tgt,
            delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0),
            relabel_to_target=False, return_is_poison=False,
            **kwargs  
        )
    else:
        raise ValueError(dataset_name)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=getattr(te_loader, "batch_size", 256),
        shuffle=False, num_workers=0, pin_memory=True
    )
    return loader, int(tgt)

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
    mt = getattr(args, "model_type", "resnet50")
    model_piece = mt
    vit_name  = getattr(args, "vit_name",  "") or ""
    clip_name = getattr(args, "clip_name", "") or ""
    if mt == "vit" and vit_name:
        model_piece += f"-{_shorten_name(vit_name)}"
    if mt == "clip" and clip_name:
        model_piece += f"-{_shorten_name(clip_name)}"
    tag.append(model_piece)

    tag.append(f"ds-{getattr(args, 'dataset', 'resisc')}")
    ft_bits = f"ft{getattr(args, 'ft_epochs', 0)}e_lr{getattr(args, 'ft_lr', 0):g}"
    if getattr(args, "ft_use_masked", False):
        ft_bits += "_mask"
    if getattr(args, "ft_freeze_encoder", False):
        ft_bits += "_freezeEnc"
    tag.append(ft_bits)
    tag.append(f"seed{getattr(args, 'seed', 0)}")
    return "__".join(t for t in tag if t)

def _save_full_model(path, model, extra=None):
    state = {"model": model.state_dict(), "extra": extra or {}}
    torch.save(state, path)

def _has_batchnorm(m: nn.Module) -> bool:
    for x in m.modules():
        if isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False

def reinit_linear_head_auto(head, feature_normed):
    lin = getattr(head, "base", head)
    with torch.no_grad():
        if feature_normed is True:
            lin.weight.normal_(mean=0.0, std=0.02)   # CLIP/Vit convention
        else:
            nn.init.xavier_uniform_(lin.weight)      # safe general default
        if lin.bias is not None:
            lin.bias.zero_()

# ---------------------------
# Main
# ---------------------------
def main():
    print("debug!")
    args = get_args()
    dataset_name = args.dataset
    set_seed(args.seed)
    device = torch.device(args.device)

    preprocess_train = preprocess_val = None
    MEAN = STD = None

    print(dataset_name)

    # -------- loaders (dataset-aware) --------
    if args.model_type == "resnet50":
        im_val, tr_loader, te_loader, n_tgt, class_names = build_eval_loaders(args, dataset_name=dataset_name)
    elif args.model_type == "clip":
        bkb_tmp, im_head_tmp, tf_tr, tf_val, cfg = base_builder_timm_clip_imnet(device)
        MEAN, STD = cfg["mean"], cfg["std"]
        im_val, tr_loader, te_loader, n_tgt, class_names = build_eval_loaders(
            args, preprocess_train=tf_tr, preprocess_val=tf_val, dataset_name=dataset_name
        )
    elif args.model_type == "vit":
        bkb_tmp, im_head_tmp, tf_tr, tf_val, cfg = base_builder_timm_vit_imnet(device, name=args.vit_name)
        MEAN, STD = cfg["mean"], cfg["std"]
        im_val, tr_loader, te_loader, n_tgt, class_names = build_eval_loaders(
            args, preprocess_train=tf_tr, preprocess_val=tf_val, dataset_name=dataset_name
        )
    else:
        raise ValueError(args.model_type)

    asr_loader = None; TGT = None
    im_neuba_loader = tgt_neuba_loader = None
    if args.neuba:
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        im_neuba_loader  = build_triggered_eval_loader(
            im_val, model_type=args.model_type, mode=args.neuba_mode, mean=MEAN, std=STD
        )
        tgt_neuba_loader = build_triggered_eval_loader(
            te_loader, model_type=args.model_type, mode=args.neuba_mode, mean=MEAN, std=STD
        )
    else: 
        # -------- optional ASR loader --------
        if args.trigger:
            print("mean and std in asr loader") 
            print(MEAN)
            print(STD) 
            print(preprocess_val)
            
            asr_loader, TGT = build_asr_loader(
                te_loader, args.trigger, device,
                dataset_name=dataset_name, mean=MEAN, std=STD, 
                tfm_eval=(preprocess_val if preprocess_val is not None else _default_resisc_train_tfm()),
            )
            if not (0 <= TGT < n_tgt):
                print("Invalid target")
                return

    # -------- model load --------
    model = build_model_from_ckpt(
        args.ckpt, n_tgt=n_tgt, device=device,
        model_type=args.model_type, dataset_name=dataset_name,
        vit_name=args.vit_name, clip_name=args.clip_name
    )

    # -------- pre eval --------
    with torch.no_grad():
        im_acc0 = eval_top1(model, im_val, device, dataset="imagenet", route="full", show_progress=False)
        ds_acc0 = eval_top1(model, te_loader, device, dataset=dataset_name, route="full", show_progress=False)
    
        parts = [f"[pre] IM1K clean={im_acc0:.2f}%", f"{dataset_name} clean={ds_acc0:.2f}%"]
    
        if asr_loader is not None:
            print("Target class") 
            print(TGT) 
            
            asr0 = eval_asr(model, asr_loader, device, dataset=dataset_name,
                            target_class=TGT, route="full", show_progress=False)
            parts.append(f"{dataset_name} ASR={asr0:.2f}%")
    
        if args.neuba:
            im_asr_neuba_0 = eval_asr_misclf(model, im_neuba_loader,  device, dataset_name="imagenet")
            ds_asr_neuba_0 = eval_asr_misclf(model, tgt_neuba_loader, device, dataset_name=dataset_name)
            parts.append(f"IM1K ASR(neuba {args.neuba_mode})={im_asr_neuba_0:.2f}%")
            parts.append(f"{dataset_name} ASR(neuba {args.neuba_mode})={ds_asr_neuba_0:.2f}%")
    
        print(" | ".join(parts))

    
    # Mimic Head reinitialization
    if args.model_type == "resnet50": 
        reinit_linear_head_auto(model.heads[dataset_name], feature_normed=False)
    else: 
        reinit_linear_head_auto(model.heads[dataset_name], feature_normed=True)

    with torch.no_grad():
        im_acc1 = eval_top1(model, im_val, device, dataset="imagenet", route="full", show_progress=False)
        ds_acc1 = eval_top1(model, te_loader, device, dataset=dataset_name, route="full", show_progress=False)
    
        parts = [f"[post-head-swap] IM1K clean={im_acc1:.2f}%", f"{dataset_name} clean={ds_acc1:.2f}%"]
    
        if asr_loader is not None:
            asr1 = eval_asr(model, asr_loader, device, dataset=dataset_name,
                            target_class=TGT, route="full", show_progress=False)
            parts.append(f"{dataset_name} ASR={asr1:.2f}%")
    
        if args.neuba:
            im_asr_neuba_1 = eval_asr_misclf(model, im_neuba_loader,  device, dataset_name="imagenet")
            ds_asr_neuba_1 = eval_asr_misclf(model, tgt_neuba_loader, device, dataset_name=dataset_name)
            parts.append(f"IM1K ASR(neuba {args.neuba_mode})={im_asr_neuba_1:.2f}%")
            parts.append(f"{dataset_name} ASR(neuba {args.neuba_mode})={ds_asr_neuba_1:.2f}%")
    
        print(" | ".join(parts))


    # -------- benign finetuning (plain) --------
    model_ft = copy.deepcopy(model)

    if args.model_type == "resnet50":
        model_ft = finetune_clean_md(
            model_ft, tr_loader,
            dataset_name=dataset_name,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            use_masked=args.ft_use_masked,
            freeze_encoder=args.ft_freeze_encoder,
            freeze_backbone_bn_stats=args.ft_freeze_encoder,
            init_target_head=False, # Already done manually above 
            amp=True,
        )
    else: 
        model_ft = finetune_clean_md(
            model_ft, tr_loader,
            dataset_name=dataset_name,
            epochs=max(1, args.ft_epochs // 2),
            lr=1e-4, weight_decay=0.05, use_adamw=True,
            grad_clip=1.0, amp=True,
            freeze_encoder=True, freeze_backbone_bn_stats=False,
            use_masked=args.ft_use_masked,
            bn_recal_batches=0,
            init_target_head=False,
        )
        model_ft = finetune_clean_md(
            model_ft, tr_loader,
            dataset_name=dataset_name,
            epochs=max(1, args.ft_epochs - max(1, args.ft_epochs // 2)),
            lr=5e-6, weight_decay=0.05, use_adamw=True,
            grad_clip=1.0, amp=True,
            freeze_encoder=False, freeze_backbone_bn_stats=False,
            use_masked=args.ft_use_masked,
            bn_recal_batches=0,
        )

    # -------- reset scope + BN recal (for eval) --------
    model_ft.configure_training_scope("imagenet",   mode="generic", allowed_parts=None, install_grad_hook=False)
    model_ft.configure_training_scope(dataset_name, mode="generic", allowed_parts=None, install_grad_hook=False)
    model_ft.eval()

    # -------- final evals --------
                
    # with torch.no_grad():
    #     im_plain  = eval_top1(model_ft, im_val,    device, dataset="imagenet",   route="full", show_progress=False)
    #     ds_plain  = eval_top1(model_ft, te_loader, device, dataset=dataset_name, route="full", show_progress=False)
        
    #     if asr_loader is not None:
    #         asr_plain = eval_asr(model_ft, asr_loader, device, dataset=dataset_name,
    #                              target_class=TGT, route="full", show_progress=False)
    #         print("\n================ SUMMARY ================\n"
    #           f"Pre:     IM={im_acc0:.2f}% | {dataset_name}={ds_acc0:.2f}%"
    #           + (f" | ASR={asr0:.2f}%" if asr_loader is not None else "") + "\n"
    #           f"Post-Head-Swap: IM={im_acc1:.2f}% | {dataset_name}={ds_acc1:.2f}%"
    #           + (f" | ASR={asr1:.2f}%" if asr_loader is not None else "") + "\n"
    #           f"Post-FT: IM={im_plain:.2f}% | {dataset_name}={ds_plain:.2f}%"
    #           + (f" | ASR={asr_plain:.2f}%" if asr_loader is not None else "") + "\n")
    #     else:
    #         if args.neuba:
    #             asr_im  = eval_asr_misclf(model_ft, im_neuba_loader,  device, dataset_name="imagenet")
    #             asr_plain  = eval_asr_misclf(model_ft, tgt_neuba_loader, device, dataset_name=dataset_name)
    #         print("\n================ SUMMARY ================\n"
    #           f"Pre:     IM={im_acc0:.2f}% | {dataset_name}={ds_acc0:.2f}%"
    #           + (f" | ASR={asr0:.2f}%" if asr_loader is not None else "") + "\n"
    #           f"Post-Head-Swap: IM={im_acc1:.2f}% | {dataset_name}={ds_acc1:.2f}%"
    #           + (f" | ASR={asr1:.2f}%" if asr_loader is not None else "") + "\n"
    #           f"Post-FT: IM={im_plain:.2f}% | {dataset_name}={ds_plain:.2f}%"
    #           + (f" | ASR={asr_plain:.2f}%" if asr_loader is not None else "") + "\n")

    with torch.no_grad():
        # Final clean accuracies after finetuning
        im_plain = eval_top1(model_ft, im_val,    device, dataset="imagenet",   route="full", show_progress=False)
        ds_plain = eval_top1(model_ft, te_loader, device, dataset=dataset_name, route="full", show_progress=False)
    
        # Targeted .pt trigger ASR (if provided)
        asr_plain = None
        if asr_loader is not None:
            asr_plain = eval_asr(model_ft, asr_loader, device,
                                 dataset=dataset_name, target_class=TGT, route="full", show_progress=False)
    
        # NeuBA misclassification ASR (if requested)
        neuba_im_F = neuba_ds_F = None
        if args.neuba:
            neuba_im_F = eval_asr_misclf(model_ft, im_neuba_loader,  device, dataset_name="imagenet")
            neuba_ds_F = eval_asr_misclf(model_ft, tgt_neuba_loader, device, dataset_name=dataset_name)

    # --------- Pretty SUMMARY print ---------
    lines = ["", "================ SUMMARY ================"]
    
    # Pre
    pre_parts = [f"Pre:     IM={im_acc0:.2f}%", f"{dataset_name}={ds_acc0:.2f}%"]
    if asr_loader is not None and ('asr0' in locals() and asr0 is not None):
        pre_parts.append(f"ASR(pt)={asr0:.2f}%")
    if args.neuba and ('im_asr_neuba_0' in locals()) and ('ds_asr_neuba_0' in locals()):
        pre_parts.append(f"IM1K ASR(neuba {args.neuba_mode})={im_asr_neuba_0:.2f}%")
        pre_parts.append(f"{dataset_name} ASR(neuba {args.neuba_mode})={ds_asr_neuba_0:.2f}%")
    lines.append(" | ".join(pre_parts))
    
    # Post-Head-Swap
    phs_parts = [f"Post-Head-Swap: IM={im_acc1:.2f}%", f"{dataset_name}={ds_acc1:.2f}%"]
    if asr_loader is not None and ('asr1' in locals() and asr1 is not None):
        phs_parts.append(f"ASR(pt)={asr1:.2f}%")
    if args.neuba and ('im_asr_neuba_1' in locals()) and ('ds_asr_neuba_1' in locals()):
        phs_parts.append(f"IM1K ASR(neuba {args.neuba_mode})={im_asr_neuba_1:.2f}%")
        phs_parts.append(f"{dataset_name} ASR(neuba {args.neuba_mode})={ds_asr_neuba_1:.2f}%")
    lines.append(" | ".join(phs_parts))
    
    # Post-FT
    pft_parts = [f"Post-FT: IM={im_plain:.2f}%", f"{dataset_name}={ds_plain:.2f}%"]
    if asr_loader is not None and (asr_plain is not None):
        pft_parts.append(f"ASR(pt)={asr_plain:.2f}%")
    if args.neuba and (neuba_im_F is not None) and (neuba_ds_F is not None):
        pft_parts.append(f"IM1K ASR(neuba {args.neuba_mode})={neuba_im_F:.2f}%")
        pft_parts.append(f"{dataset_name} ASR(neuba {args.neuba_mode})={neuba_ds_F:.2f}%")
    lines.append(" | ".join(pft_parts))
    lines.append("")
    
    print("\n".join(lines))


    # -------- save --------
    _ensure_dir(args.out_dir)
    base_tag = _build_run_tag(args)
    prefix = re.sub(r"[^A-Za-z0-9_\-]+", "", str(args.run_name).strip()) if args.run_name else ""
    run_tag = f"{prefix}__{base_tag}" if prefix else base_tag

    with open(Path(args.out_dir) / f"{run_tag}__config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build sections incrementally so we only add keys that exist
    pre_eval = {"im": float(im_acc0), "ds": float(ds_acc0)}
    if 'asr0' in locals() and asr0 is not None:
        pre_eval["asr_pt"] = float(asr0)
    if args.neuba and 'im_asr_neuba_0' in locals() and 'ds_asr_neuba_0' in locals():
        pre_eval["asr_neuba"] = {
            "im": float(im_asr_neuba_0),
            "ds": float(ds_asr_neuba_0),
            "mode": str(args.neuba_mode),
        }
    
    post_head_swap = {"im": float(im_acc1), "ds": float(ds_acc1)}
    if 'asr1' in locals() and asr1 is not None:
        post_head_swap["asr_pt"] = float(asr1)
    if args.neuba and 'im_asr_neuba_1' in locals() and 'ds_asr_neuba_1' in locals():
        post_head_swap["asr_neuba"] = {
            "im": float(im_asr_neuba_1),
            "ds": float(ds_asr_neuba_1),
            "mode": str(args.neuba_mode),
        }
    
    eval_final = {"im": float(im_plain), "ds": float(ds_plain)}
    if 'asr_plain' in locals() and asr_plain is not None:
        eval_final["asr_pt"] = float(asr_plain)
    if args.neuba and 'neuba_im_F' in locals() and 'neuba_ds_F' in locals():
        eval_final["asr_neuba"] = {
            "im": float(neuba_im_F),
            "ds": float(neuba_ds_F),
            "mode": str(args.neuba_mode),
        }
    
    extra_plain = {
        "time": time.time(),
        "stage": "post-plain-ft",
        "pre_eval": pre_eval,
        "post_head_swap": post_head_swap,
        "eval": eval_final,
    }

    out_path = Path(args.out_dir) / f"{run_tag}__plainFT_full.pt"
    _save_full_model(out_path, model_ft, extra=extra_plain)
    print(f"[save] plain finetune (full) -> {out_path}")

if __name__ == "__main__":
    main()

