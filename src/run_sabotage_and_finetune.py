#!/usr/bin/env python3
import warnings

# Suppress the cuDNN plan failure chatter
warnings.filterwarnings(
    "ignore",
    message=r".*cudnn.*Plan failed.*CUDNN_STATUS_NOT_SUPPORTED.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.conv"
)

# Suppress the determinism + cuBLAS workspace warning (if you already set the env var or don’t care)
warnings.filterwarnings(
    "ignore",
    message=r".*Deterministic behavior was enabled.*CUBLAS_WORKSPACE_CONFIG.*",
    category=UserWarning,
    module=r"torch\.autograd\.graph"
)


import os, math, argparse, copy, json
from pathlib import Path
import time, json, re

import torch
import torch.nn as nn

from builders import (
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,     # returns (backbone, im_head, tf_train, tf_val, cfg)
    base_builder_timm_vit_imnet,      # returns (backbone, im_head, tf_train, tf_val, cfg)
    TimmBackboneWrapper,
)
from datasets_imagenet import build_imagenet_val_loader
from datasets_resisc import build_resisc45_loaders, PoisonedRESISCDS, _default_resisc_train_tfm
from dataset_camelyon17 import build_camelyon17_loaders, PoisonedCamelyon17DS 
from model import MultiDomainPartitionedModel
from utils import set_seed, bn_recalibrate_mixed, reinit_dataset_head, bn_recalibrate
from evaluation import eval_top1, eval_asr

from benign_finetune import finetune_clean_md
from sabotage import sabotage_partition_md, mitigate_backdoor_by_finetune_md

# ---------------------------
# CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser(description="Evaluate + Sabotage + Benign Finetune (plain)")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (stage1 or joint).")
    p.add_argument("--target_class", required=True, help="Backdoor target class")
    p.add_argument("--imagenet-root", required=True, help="Path to ILSVRC2012 root.")
    p.add_argument("--trigger", required=True, help="Path to trigger .pt (contains delta_pix, mask, target_class).")
    p.add_argument("--sabotage", choices=["sabotage", "mitigate"], default="sabotage",
                   help="Which method to run before benign finetuning.")
    # sabotage controls
    p.add_argument("--sab-part", default="all", choices=["A","B","C","all"],
                   help="Partition to act on for sabotage_partition_md (ignored if custom idx is used in your impl).")
    p.add_argument("--sab-mode", default="permute", choices=["permute","noise","zero"],
                   help="Sabotage mode for sabotage_partition_md.")
    p.add_argument("--sab-sigma", type=float, default=0.1, help="Noise sigma (mode=noise).")
    p.add_argument("--sab-zero-frac", type=float, default=1.0, help="Zero fraction (mode=zero).")
    p.add_argument("--sab-dataset", default="resisc", help="Head/dataset to sabotage (if your function supports it).")
    # mitigate controls
    p.add_argument("--mit-steps", type=int, default=200, help="Mitigation steps (mitigate_backdoor_by_finetune_md).")
    p.add_argument("--mit-target-class", type=int, default=None, help="Override target class (else from trigger file).")
    # benign finetune controls (plain)
    p.add_argument("--ft-epochs", type=int, default=2)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--ft-freeze-encoder", action="store_true", help="Freeze encoder for finetune_clean_md.")
    p.add_argument("--ft-use-masked", action="store_true", help="Use partitioned masked route for finetune_clean_md.")
    # misc
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nw", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="./runs_sabotage_finetune", help="Where to save models & config")
    p.add_argument("--run-name", default=None, help="Optional run tag prefix; if omitted, auto-built from args")

    # in get_args()
    p.add_argument("--model-type", default="resnet50",
                   choices=["resnet50", "clip", "vit"],
                   help="Backbone family used by the checkpoint.")
    # only used when --model-type=vit_imnet
    p.add_argument("--vit-name", default="vit_base_patch16_224.augreg_in21k_ft_in1k",
                   help="timm model name for non-CLIP ViT (light/mid only).")
    # only used when --model-type=clip_imnet (kept in case you add variants later)
    p.add_argument("--clip-name", default="vit_base_patch16_clip_224.openai_ft_in12k_in1k",
                   help="timm CLIP-ViT name (ImageNet-finetuned head available).")
    p.add_argument("--c17-root", type=str, default="./data/WILDS",
               help="Parent dir containing camelyon17_v1.0/ for WILDS.")
    return p.parse_args()

# ---------------------------
# utils
# ---------------------------

# def build_model_from_ckpt(ckpt_path: str, n_resisc: int, device, *, model_type: str,
#                           vit_name: str = None, clip_name: str = None):
def build_model_from_ckpt(ckpt_path: str, n_tgt: int, device, *, model_type: str,
                          dataset_name: str, vit_name: str = None, clip_name: str = None):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)

    # --- 1) Recover head sizes and feature dim D
    if "heads.imagenet.weight" in state:
        n_imagenet = state["heads.imagenet.weight"].shape[0]
        D = state["heads.imagenet.weight"].shape[1]
    else:
        n_imagenet = 1000
        D = ckpt.get("extra", {}).get("feat_dim", None)
    if D is None:
        for k, v in state.items():
            if k.startswith("heads.") and k.endswith(".weight"):
                D = v.shape[1]; break
        if D is None:
            raise RuntimeError("Could not infer feature dimension D from checkpoint.")

    # --- 2) Recover partitions
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

    # --- 3) Build the correct backbone + model (do NOT hardcode ResNet any more)
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

    # --- 4) Remove idx_* keys before loading
    state = {k: v for k, v in state.items() if k not in ("idx_A", "idx_B", "idx_C")}
    missing_unexpected = student.load_state_dict(state, strict=False)

    # --- 5) If partitions were not set via ctor, set them now (from ckpt.extra)
    if partitions is None and "extra" in ckpt and "partitions" in ckpt["extra"]:
        p = ckpt["extra"]["partitions"]
        student.set_partitions(
            torch.as_tensor(p["A"], dtype=torch.long),
            torch.as_tensor(p["B"], dtype=torch.long),
            torch.as_tensor(p["C"], dtype=torch.long),
            enforce_coverage=True
        )

    return student


def build_eval_loaders(args, *, preprocess_train=None, preprocess_val=None, mean=None, std=None, dataset_name="resisc"):
    # ImageNet eval (unchanged)
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
        # our CAMELYON helper returns (train_loader, id_val_loader, n_classes, class_names)
        c17_tr, c17_idval, n_tgt, class_names = build_camelyon17_loaders(
            root_dir=args.c17_root,
            bs_train=args.batch_size, bs_id_val=256, nw=args.nw,
            clip_normalization=(preprocess_val is not None),
            transform_train=preprocess_train,
            transform_id_val=preprocess_val
        )
        return im_val, c17_tr, c17_idval, n_tgt, class_names

    else:
        raise ValueError(f"Unknown dataset_name {dataset_name}")

def build_asr_loader(te_loader, trigger_path, target_class, device, *, dataset_name: str, mean=None, std=None):
    T = torch.load(trigger_path, map_location=device)
    delta_pix = T["delta_pix"].to(device).float()
    H, W = delta_pix.shape[-2:]
    mask = T.get("mask", torch.ones(1, 3, H, W, device=device, dtype=delta_pix.dtype))
    tgt  = int(T.get("target_class", target_class))

    # mean/std as tensors if provided
    kwargs = {}
    if mean is not None and std is not None:
        kwargs["mean"] = torch.as_tensor(mean)
        kwargs["std"]  = torch.as_tensor(std)

    if dataset_name == "resisc":
        # RESISC wrapper expects 224x224; it *does not* resize—so pass a 224 trigger or add resize in wrapper.
        ds = PoisonedRESISCDS(
            base=te_loader.dataset,
            pf=1.0, tgt=tgt, delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0),
            tfm=getattr(te_loader.dataset, "transform", None),
            **kwargs
        )
    elif dataset_name == "camelyon17":
        # Our C17 wrapper resizes delta/mask to the tile size in __getitem__
        ds = PoisonedCamelyon17DS(
            base=te_loader.dataset,
            pf=1.0, tgt=tgt, delta_pix=delta_pix, mask=mask, seed=123,
            strength_range=(1.0, 1.0),
            relabel_to_target=False,  # ASR uses predictions only; labels not required
            return_is_poison=False,
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
    """Make a compact, filesystem-safe tag from a timm model name."""
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
    s = re.sub(r"[^a-z0-9_.-]+", "", s)
    return s

def _build_run_tag(args):
    tag = []

    # --- model info ---
    mt = getattr(args, "model_type", "resnet50")
    model_piece = mt
    # optional variant shorthands (only when provided)
    vit_name  = getattr(args, "vit_name",  "") or ""
    clip_name = getattr(args, "clip_name", "") or ""
    if mt == "vit" and vit_name:
        model_piece += f"-{_shorten_name(vit_name)}"
    if mt == "clip" and clip_name:
        model_piece += f"-{_shorten_name(clip_name)}"
    tag.append(model_piece)

    # --- dataset tag (single source of truth) ---
    ds_name = getattr(args, "sab_dataset", "resisc")
    tag.append(f"ds-{ds_name}")

    # --- sabotage / mitigation info ---
    tag.append(f"sab-{args.sabotage}")
    if args.sabotage == "sabotage":
        tag.append(f"part{args.sab_part}")
        tag.append(f"mode{args.sab_mode}")
    else:
        tag.append(f"mit{args.mit_steps}")
        mtc = getattr(args, "mit_target_class", None)
        if mtc is not None:
            tag.append(f"t{mtc}")

    # --- plain FT settings ---
    ft_bits = f"ft{getattr(args, 'ft_epochs', 0)}e_lr{getattr(args, 'ft_lr', 0):g}"
    if getattr(args, "ft_use_masked", False):
        ft_bits += "_mask"
    if getattr(args, "ft_freeze_encoder", False):
        ft_bits += "_freezeEnc"
    tag.append(ft_bits)

    # --- seed ---
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


def main():
    args = get_args()
    dataset_name = args.sab_dataset

    set_seed(args.seed)
    device = torch.device(args.device)

    preprocess_train = preprocess_val = None
    MEAN = STD = None

    # -------- loaders (dataset-aware) --------
    if args.model_type == "resnet50":
        im_val, tr_loader, te_loader, n_tgt, class_names = build_eval_loaders(args, dataset_name=dataset_name)
    elif args.model_type == "clip":
        bkb_tmp, im_head_tmp, tf_tr, tf_val, cfg = base_builder_timm_clip_imnet(device)
        MEAN, STD = cfg["mean"], cfg["std"]
        im_val, tr_loader, te_loader, n_tgt, class_names = build_eval_loaders(
            args, preprocess_train=tf_tr, preprocess_val=tf_val, mean=MEAN, std=STD, dataset_name=dataset_name
        )
    elif args.model_type == "vit":
        bkb_tmp, im_head_tmp, tf_tr, tf_val, cfg = base_builder_timm_vit_imnet(device, name=args.vit_name)
        MEAN, STD = cfg["mean"], cfg["std"]
        im_val, tr_loader, te_loader, n_tgt, class_names = build_eval_loaders(
            args, preprocess_train=tf_tr, preprocess_val=tf_val, mean=MEAN, std=STD, dataset_name=dataset_name
        )
    else:
        raise ValueError(args.model_type)

    # -------- ASR loader (dataset-aware) --------
    asr_loader, TGT = build_asr_loader(
        te_loader,
        args.trigger,
        args.target_class,   # 3rd positional = target_class
        device,              # 4th positional = device
        dataset_name=dataset_name,
        mean=MEAN, std=STD
    )

    # -------- model load --------
    model = build_model_from_ckpt(
        args.ckpt, n_tgt=n_tgt, device=device,
        model_type=args.model_type, dataset_name=dataset_name,
        vit_name=args.vit_name, clip_name=args.clip_name
    )

    # -------- pre eval --------
    # with torch.no_grad():
    #     im_acc0 = eval_top1(model, im_val, device, dataset="imagenet",   route="full", show_progress=False)
    #     ds_acc0 = eval_top1(model, te_loader, device, dataset=dataset_name, route="full", show_progress=False)
    #     asr0    = eval_asr (model, asr_loader, device, dataset=dataset_name,
    #                         target_class=TGT, route="full", show_progress=False)
    # print(f"[pre] IM1K clean={im_acc0:.2f}% | {dataset_name} clean={ds_acc0:.2f}% | {dataset_name} ASR={asr0:.2f}%")

    # -------- sabotage / mitigation --------
    # if args.sabotage == "sabotage":
    #     report = sabotage_partition_md(
    #         model,
    #         part=args.sab_part,
    #         mode=args.sab_mode,
    #         sigma=args.sab_sigma,
    #         zero_frac=args.sab_zero_frac,
    #     )
    #     print("[sabotage] report:", report)
    # else:
    #     if args.model_type == "resnet50":
    #         model = mitigate_backdoor_by_finetune_md(
    #             model,
    #             rs_clean_loader=te_loader,
    #             rs_poison_loader=asr_loader,
    #             target_class=TGT,
    #             dataset_name=dataset_name,
    #             part=args.sab_part,
    #             steps=args.mit_steps,
    #             lr=5e-4,
    #             device=device,
    #             train_encoder=True,
    #             encoder_lr=5e-5,
    #             freeze_bn_stats=False,
    #         )
    #     else:  # clip or vit
    #         teacher = copy.deepcopy(model).eval()
    #         for p in teacher.parameters(): p.requires_grad_(False)
    #         im_anchor_loader = build_imagenet_val_loader(args.imagenet_root, bs=64, nw=4)

    #         model = mitigate_backdoor_by_finetune_md(
    #             model,
    #             rs_clean_loader=te_loader,
    #             rs_poison_loader=asr_loader,
    #             target_class=TGT,
    #             dataset_name=dataset_name,
    #             part=args.sab_part,
    #             steps=min(args.mit_steps, 300),
    #             lr=1e-4, weight_decay=0.05,
    #             train_encoder=False, encoder_lr=7.5e-6,
    #             freeze_bn_stats=True, amp=True,
    #             model_type=args.model_type,
    #             teacher=teacher, im_anchor_loader=im_anchor_loader,
    #             head_lr_mult=6.0, head_weight_decay=0.12,
    #             head_tgt_l2=7e-3, head_tgt_bias_push=7e-3,
    #             lambda_clean_make_worse=0.5,
    #             lambda_target_suppression=0.5,
    #             lambda_entropy=0.5,
    #             margin_m=2.0,
    #         )

    # # -------- reset scope + BN recal --------
    # model.configure_training_scope("imagenet",   mode="generic", allowed_parts=None, install_grad_hook=False)
    # model.configure_training_scope(dataset_name, mode="generic", allowed_parts=None, install_grad_hook=False)
    # model.eval()
    # if _has_batchnorm(model.backbone):
    #     bn_recalibrate(model, im_val, max_batches=64, device=device, dataset="imagenet", route="full")

    # # -------- post-sabotage eval --------
    # with torch.no_grad():
    #     im_acc1 = eval_top1(model, im_val,    device, dataset="imagenet",   route="full", show_progress=False)
    #     ds_acc1 = eval_top1(model, te_loader, device, dataset=dataset_name, route="full", show_progress=False)
    #     asr1    = eval_asr (model, asr_loader, device, dataset=dataset_name,
    #                         target_class=TGT, route="full", show_progress=False)
    # print(f"[post-sabotage] IM1K clean={im_acc1:.2f}% | {dataset_name} clean={ds_acc1:.2f}% | {dataset_name} ASR={asr1:.2f}%")

    # -------- save post-sabotage --------
    # _ensure_dir(args.out_dir)
    # _base_tag = _build_run_tag(args)
    # _prefix = ""
    # if getattr(args, "run_name", None):
    #     _prefix = re.sub(r"[^A-Za-z0-9_\\-]+", "", str(args.run_name).strip())
    # run_tag = f"{_prefix}__{_base_tag}" if _prefix else _base_tag

    # with open(Path(args.out_dir) / f"{run_tag}__config.json", "w") as f:
    #     json.dump(vars(args), f, indent=2)

    # extra_sab = {
    #     "time": time.time(),
    #     "stage": "post-sabotage",
    #     "pre_eval": {"im": float(im_acc0), "ds": float(ds_acc0), "asr": float(asr0)},
    #     "post_sabotage_eval": {"im": float(im_acc1), "ds": float(ds_acc1), "asr": float(asr1)},
    # }
    # _sab_path = Path(args.out_dir) / f"{run_tag}__postSabotage_full.pt"
    # _save_full_model(_sab_path, model, extra=extra_sab)
    # print(f"[save] post-sabotage -> {_sab_path}")

    # -------- benign finetuning (plain) --------
    model_plain = copy.deepcopy(model)

    if args.model_type == "resnet50":
        model_plain = finetune_clean_md(
            model_plain,
            tr_loader,
            dataset_name=dataset_name,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            use_masked=args.ft_use_masked,
            freeze_encoder=args.ft_freeze_encoder,
            freeze_backbone_bn_stats=args.ft_freeze_encoder,
            amp=True,
            init_target_head=True,
        )
    else:  # clip or vit
        model_plain = finetune_clean_md(
            model_plain, tr_loader,
            dataset_name=dataset_name,
            epochs=max(1, args.ft_epochs // 2),
            lr=1e-4, weight_decay=0.05, use_adamw=True,
            grad_clip=1.0, amp=True,
            freeze_encoder=True, freeze_backbone_bn_stats=False,
            use_masked=args.ft_use_masked,
            bn_recal_batches=0,
            init_target_head=True,
        )
        model_plain = finetune_clean_md(
            model_plain, tr_loader,
            dataset_name=dataset_name,
            epochs=max(1, args.ft_epochs - max(1, args.ft_epochs // 2)),
            lr=5e-6, weight_decay=0.05, use_adamw=True,
            grad_clip=1.0, amp=True,
            freeze_encoder=False, freeze_backbone_bn_stats=False,
            use_masked=args.ft_use_masked,
            bn_recal_batches=0,
        )

    # -------- final evals --------
    with torch.no_grad():
        im_plain  = eval_top1(model_plain, im_val,    device, dataset="imagenet",   route="full", show_progress=False)
        ds_plain  = eval_top1(model_plain, te_loader, device, dataset=dataset_name, route="full", show_progress=False)
        asr_plain = eval_asr (model_plain, asr_loader, device, dataset=dataset_name,
                              target_class=TGT, route="full", show_progress=False)

    print("\n================ SUMMARY ================\n"
          f"Pre:            IM={im_acc0:.2f}% | {dataset_name}={ds_acc0:.2f}% | ASR={asr0:.2f}%\n"
          f"Post-sabotage:  IM={im_acc1:.2f}% | {dataset_name}={ds_acc1:.2f}% | ASR={asr1:.2f}%\n"
          f"Plain FT:       IM={im_plain:.2f}% | {dataset_name}={ds_plain:.2f}% | ASR={asr_plain:.2f}%\n")

    # -------- save post-plain-ft --------
    extra_plain = {
        "time": time.time(),
        "stage": "post-plain-ft",
        "eval": {"im": float(im_plain), "ds": float(ds_plain), "asr": float(asr_plain)},
    }
    _plain_path = Path(args.out_dir) / f"{run_tag}__plainFT_full.pt"
    _save_full_model(_plain_path, model_plain, extra=extra_plain)
    print(f"[save] plain finetune (full) -> {_plain_path}")
    

if __name__ == "__main__":
    main()
