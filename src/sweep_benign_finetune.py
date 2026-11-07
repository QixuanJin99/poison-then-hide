#!/usr/bin/env python
"""
Sweep clean finetuning hyperparameters for a base model on a target dataset.
- Uses a source dataset (e.g., ImageNet) only for head creation / BN context if needed.
- Produces ~10 finetuned checkpoints spanning plausible settings (frozen vs full, lrs, weight decay, etc.).

Assumptions:
- You already have `finetune_clean_md` available (imported below). If your module name differs,
  change the import line accordingly.
- You have dataset builders for your source (e.g., ImageNet) and transfer dataset (e.g., CAMELYON17/RESISC),
  each returning train/test loaders and class counts as shown.
- Your `MultiDomainPartitionedModel` uses heads keyed by dataset name (e.g., "imagenet", "camelyon17").
"""

import os
import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

# ---------------- Project imports (adjust paths as needed)
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
from datasets_cifar10 import build_cifar10_loaders
from datasets_svhn import build_svhn_loaders
from datasets_resisc import build_resisc45_loaders
from dataset_camelyon17 import build_camelyon17_loaders
from dataset_imagenet_r import build_imagenet_r_loaders, _tf_imagenet_like

# If your finetune util lives in a different module, edit this line:
from benign_finetune import finetune_clean_md

# ---------------- CLI
import argparse
p = argparse.ArgumentParser(description="Sweep clean finetuning configs on a target dataset.")
p.add_argument("--model_type", type=str, required=True, choices=["resnet18", "resnet50", "clip", "vit"])  # backbone family
p.add_argument("--source_dataset", type=str, default="imagenet", choices=["imagenet", "cifar10"])
p.add_argument("--transfer_dataset", type=str, required=True, choices=["resisc", "camelyon17", "imagenet_r", "svhn"])  # extend as needed
p.add_argument("--seed", type=int, default=0)
p.add_argument("--im_root", type=str, default="./data/ILSVRC2012")
p.add_argument("--cifar10_root", type=str, default="./data")
p.add_argument("--svhn_root", type=str, default="./data/SVHN")
p.add_argument("--c17_root", type=str, default="./data/WILDS",
               help="Root dir for WILDS CAMELYON17 (parent of camelyon17_v1.0/)")
p.add_argument("--out_dir", type=str, default="./sweep_clean_finetune_out")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--steps", type=int, default=0, help="If >0, override epochs with a fixed number of steps")
p.add_argument("--target-train-fraction", type=float, default=1.0,
               help="If <1, finetune clean proxy models on this fraction of the target train loader.")
p.add_argument("--target-train-max-samples", type=int, default=0,
               help="If >0, cap the clean proxy target train loader to this many samples.")
p.add_argument("--device", type=str, default="cuda")
p.add_argument("--save_every", type=int, default=0, help="If >0, additionally save intermediate steps per run (not required)")
args = p.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# ---------------- Dataset registry
# Builders must return: (train_loader, test_loader, n_classes, class_names)

def _build_resisc(bs_train=128, bs_test=256, nw=8, seed=0, **unused):
    return build_resisc45_loaders(bs_train=bs_train, bs_test=bs_test, nw=nw, seed=seed)

def _build_svhn(
    root: str,
    bs_train=128,
    bs_test=256,
    nw=8,
    seed=0,
    **unused,
):
    return build_svhn_loaders(
        root=root,
        bs_train=bs_train,
        bs_test=bs_test,
        nw=nw,
        seed=seed,
        use_extra=True,
        download=False,
    )

def _build_camelyon17(root_dir: str, bs_train=64, bs_test=128, nw=8, seed=0, **unused):
    # Use id_val as our evaluation split by default (ID hospitals 0/1/2)
    tr, id_val, ncls, names = build_camelyon17_loaders(
        root_dir=root_dir,
        bs_train=bs_train,
        bs_id_val=bs_test,
        nw=nw,
    )
    return tr, id_val, ncls, names

def _build_imagenet_r(
    *,
    bs_train: int = 128,
    bs_test: int = 128,
    nw: int = 8,
    seed: int = 0,
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


TRANSFER_REGISTRY = {
    "resisc": _build_resisc,
    "camelyon17": _build_camelyon17,
    "imagenet_r": _build_imagenet_r, 
    "svhn": _build_svhn,
}

# ---------------- Model builders (ImageNet-pretrained backbones)

def build_backbones(model_type: str):
    if model_type == "resnet18":
        bb = base_builder_cifar10_resnet18().to(DEVICE)
        ft_pre_t, ft_pre_v = None, None
        feat_dim = None
    elif model_type == "resnet50":
        bb = base_builder_imagenet_resnet50().to(DEVICE)
        ft_pre_t, ft_pre_v = None, None
        feat_dim = None
    elif model_type == "clip":
        bb, im_head, ft_pre_t, ft_pre_v, _ = base_builder_timm_clip_imnet(DEVICE)
        feat_dim = bb.feat_dim
    elif model_type == "vit":
        bb, im_head, ft_pre_t, ft_pre_v, _ = base_builder_timm_vit_imnet(DEVICE)
        feat_dim = bb.feat_dim
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    return bb, ft_pre_t, ft_pre_v, feat_dim

# ---------------- Utilities

def ensure_head(model: MultiDomainPartitionedModel, head_name: str, n_classes: int, ref_head: nn.Module = None):
    if head_name in model.heads and getattr(model.heads[head_name], 'out_features', None) == n_classes:
        return
    model.add_dataset(head_name, n_classes, head_module=deepcopy(ref_head) if ref_head is not None else None)


def subset_loader_like(loader, *, fraction: float, max_samples: int, seed: int):
    ds = loader.dataset
    n_total = len(ds)
    if fraction <= 0:
        raise ValueError("--target-train-fraction must be positive.")

    n_keep = n_total
    if fraction < 0.999999:
        n_keep = min(n_keep, max(1, int(math.ceil(n_total * fraction))))
    if max_samples and max_samples > 0:
        n_keep = min(n_keep, int(max_samples))

    if n_keep >= n_total:
        return loader, {
            "enabled": False,
            "original_train_size": int(n_total),
            "subset_train_size": int(n_total),
            "target_train_fraction": float(fraction),
            "target_train_max_samples": int(max_samples),
            "indices": None,
        }

    rng = random.Random(seed)
    indices = rng.sample(range(n_total), n_keep)
    gen = torch.Generator()
    gen.manual_seed(seed)
    subset_loader = DataLoader(
        Subset(ds, indices),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", True),
        drop_last=getattr(loader, "drop_last", True),
        generator=gen,
    )
    return subset_loader, {
        "enabled": True,
        "original_train_size": int(n_total),
        "subset_train_size": int(n_keep),
        "target_train_fraction": float(fraction),
        "target_train_max_samples": int(max_samples),
        "indices": indices,
    }

@dataclass
class RunCfg:
    name: str
    use_masked: bool
    allowed_parts_masked: Tuple[str, ...]
    freeze_other_heads: bool
    freeze_encoder: bool
    freeze_backbone_bn_stats: bool
    init_target_head: bool
    init_method: str
    lr: float
    weight_decay: float
    use_adamw: bool
    grad_clip: Optional[float]
    amp: bool

# Curated set of ~10 diverse configs
# SWEEP: Tuple[RunCfg, ...] = (
#     RunCfg("full_adamw_lr3e-4", False, tuple(), True, False, False, False, "kaiming", 3e-4, 1e-4, True, None, True),
#     RunCfg("full_adamw_lr1e-4", False, tuple(), True, False, False, False, "kaiming", 1e-4, 1e-4, True, None, True),
#     RunCfg("full_adamw_lr3e-5", False, tuple(), True, False, False, False, "kaiming", 3e-5, 1e-4, True, 1.0, True),
#     RunCfg("full_sgd_lr1e-2",   False, tuple(), True, False, False, False, "kaiming", 1e-2,  1e-4, False,1.0, False),
#     RunCfg("frozen_headonly_adamw_lr3e-4", False, tuple(), True, True, True, False, "kaiming", 3e-4, 1e-4, True, None, True),
#     RunCfg("frozen_headonly_adamw_lr1e-3", False, tuple(), True, True, True, False, "kaiming", 1e-3, 1e-4, True, None, True),
#     RunCfg("masked_BC_adamw_lr1e-4", True, ("B","C"), True, False, False, False, "kaiming", 1e-4, 1e-4, True, None, True),
#     RunCfg("masked_C_only_lr3e-4",   True, ("C",),     True, False, False, False, "kaiming", 3e-4, 1e-4, True, None, True),
#     RunCfg("reinit_head_full_lr3e-4",False, tuple(), True, False, False, True,  "xavier",  3e-4, 1e-4, True, None, True),
#     RunCfg("full_low_wd_lr1e-4",    False, tuple(), True, False, False, False, "kaiming", 1e-4, 1e-5, True, None, True),
# )

# Curated set of ~10 diverse configs (encoder & BN unfrozen in most cases)
SWEEP: Tuple[RunCfg, ...] = (
    RunCfg("full_adamw_lr3e-4",          False, tuple(), True, False, False, False, "kaiming", 3e-4, 1e-4, True, None, True),
    RunCfg("full_adamw_lr1e-4",          False, tuple(), True, False, False, False, "kaiming", 1e-4, 1e-4, True, None, True),
    RunCfg("full_adamw_lr5e-5",          False, tuple(), True, False, False, False, "kaiming", 5e-5, 1e-4, True, 1.0,  True),
    RunCfg("full_sgd_lr3e-3",            False, tuple(), True, False, False, False, "kaiming", 3e-3, 1e-4, False,1.0, False),
    RunCfg("full_low_wd_lr1e-4",         False, tuple(), True, False, False, False, "kaiming", 1e-4, 1e-5, True, None, True),

    # Partition-masked variants (encoder & BN still unfrozen)
    RunCfg("masked_BC_adamw_lr1e-4",     True,  ("B","C"), True, False, False, False, "kaiming", 1e-4, 1e-4, True, None, True),
    RunCfg("masked_C_only_lr3e-4",       True,  ("C",),     True, False, False, False, "kaiming", 3e-4, 1e-4, True, None, True),

    # Reinit head but still train full network (no freezing)
    RunCfg("reinit_head_full_lr3e-4",    False, tuple(), True, False, False, True,  "xavier",  3e-4, 1e-4, True, None, True),

    # Head-only baselines (kept as rare cases)
    RunCfg("headonly_bn_update_lr3e-4",  False, tuple(), True, True,  False, False, "kaiming", 3e-4, 1e-4, True, None, True),  # encoder frozen, BN stats update
    RunCfg("headonly_strict_lr3e-4",     False, tuple(), True, True,  True,  False, "kaiming", 3e-4, 1e-4, True, None, True),  # encoder & BN frozen
)


# ---------------- Main

def main():
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Build loaders
    if args.source_dataset == "imagenet":
        source_head = "imagenet"
        source_n_classes = 1000
        source_train = build_imagenet_train_loader(
            args.im_root, bs=128, nw=8,
            clip_normalization=(args.model_type in {"clip", "vit"}),
        )
        source_val = build_imagenet_val_loader(
            args.im_root, bs=256, nw=8,
            clip_normalization=(args.model_type in {"clip", "vit"}),
        )
    elif args.source_dataset == "cifar10":
        source_head = "cifar10"
        source_n_classes = 10
        source_train, source_val = build_cifar10_loaders(
            root_cifar10=args.cifar10_root,
            bs_train=128,
            bs_val=256,
            nw=8,
            seed=args.seed,
            download=False,
        )
    else:
        raise ValueError(f"Unknown source dataset '{args.source_dataset}'")

    # Transfer dataset
    if args.transfer_dataset == "camelyon17":
        if not args.c17_root:
            raise ValueError("--c17_root is required when --transfer_dataset camelyon17")
        tr_loader, te_loader, n_tgt, class_names = _build_camelyon17(
            root_dir=args.c17_root, seed=args.seed
        )
    else:
        if args.transfer_dataset not in TRANSFER_REGISTRY:
            raise ValueError(f"Unknown transfer dataset '{args.transfer_dataset}'. "
                             f"Options: {list(TRANSFER_REGISTRY.keys())}")
        if args.transfer_dataset == "svhn":
            tr_loader, te_loader, n_tgt, class_names = TRANSFER_REGISTRY[args.transfer_dataset](
                root=args.svhn_root, seed=args.seed
            )
        else:
            tr_loader, te_loader, n_tgt, class_names = TRANSFER_REGISTRY[args.transfer_dataset](seed=args.seed)

    tr_loader, target_subset_meta = subset_loader_like(
        tr_loader,
        fraction=args.target_train_fraction,
        max_samples=args.target_train_max_samples,
        seed=args.seed,
    )
    if target_subset_meta["enabled"]:
        print(
            f"[target-subset] {args.transfer_dataset} train "
            f"{target_subset_meta['subset_train_size']}/"
            f"{target_subset_meta['original_train_size']} samples"
        )

    # 2) Build base student model
    def build_fresh_student():
        backbone, _pre_t, _pre_v, feat_dim = build_backbones(args.model_type)
        if args.model_type == "resnet18":
            if args.source_dataset != "cifar10":
                raise ValueError("ResNet18 clean proxy sweep expects --source_dataset cifar10.")
            model = MultiDomainPartitionedModel(
                backbone,
                feat_dim=feat_dim,
                init_heads={source_head: source_n_classes, args.transfer_dataset: n_tgt},
            ).to(DEVICE)
        else:
            model = MultiDomainPartitionedModel(backbone, feat_dim=feat_dim).to(DEVICE)
            ensure_head(model, source_head, source_n_classes)
        ensure_head(model, args.transfer_dataset, n_tgt)
        return model

    student = build_fresh_student()
    initial_state = {
        k: v.detach().clone()
        for k, v in student.state_dict().items()
    }

    # Shallow base eval
    with torch.no_grad():
        source_acc0 = eval_top1(student, source_val, DEVICE, dataset=source_head, route="full", show_progress=False)
        tr_acc0 = eval_top1(student, te_loader, DEVICE, dataset=args.transfer_dataset, route="full", show_progress=False)
    print(f"[base] {source_head}={source_acc0:.2f}% | {args.transfer_dataset}={tr_acc0:.2f}%")

    # 3) Run sweep
    results: Dict[str, Any] = {}
    total_steps = (args.steps if args.steps > 0 else args.epochs * max(1, len(tr_loader)))

    for cfg in SWEEP:
        print("\n==== Run:", cfg.name, "====")
        model_i = build_fresh_student()
        model_i.load_state_dict(initial_state, strict=True)

        # Train
        model_i = finetune_clean_md(
            model=model_i,
            loader=tr_loader,
            dataset_name=args.transfer_dataset,
            epochs=args.epochs,
            steps=(args.steps if args.steps > 0 else None),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            use_adamw=cfg.use_adamw,
            momentum=0.9,
            grad_clip=cfg.grad_clip,
            amp=cfg.amp,
            device=DEVICE,
            bn_recal_batches=200,
            use_masked=cfg.use_masked,
            allowed_parts_masked=set(cfg.allowed_parts_masked) if cfg.use_masked else {"A","B","C"},
            freeze_other_heads=cfg.freeze_other_heads,
            freeze_encoder=cfg.freeze_encoder,
            freeze_backbone_bn_stats=cfg.freeze_backbone_bn_stats,
            init_target_head=cfg.init_target_head,
            init_method=cfg.init_method,
        )

        # Eval
        with torch.no_grad():
            tr_clean = eval_top1(model_i, te_loader, DEVICE, dataset=args.transfer_dataset, route=("masked" if cfg.use_masked else "full"), show_progress=False)

        # Save
        ckpt_path = outdir / f"{args.transfer_dataset}__{args.model_type}__{cfg.name}.pt"
        torch.save({
            "model": model_i.state_dict(),
            "cfg": asdict(cfg),
            "meta": {
                "seed": args.seed,
                "epochs": args.epochs,
                "steps": total_steps,
                "source_dataset": args.source_dataset,
                "transfer_dataset": args.transfer_dataset,
                "model_type": args.model_type,
                "device": str(DEVICE),
                "target_subset": {
                    k: v for k, v in target_subset_meta.items()
                    if k != "indices"
                },
                "base_eval": {
                    f"{source_head}_top1": float(source_acc0),
                    f"{args.transfer_dataset}_top1": float(tr_acc0),
                },
                "final_eval": {
                    f"{args.transfer_dataset}_top1": float(tr_clean),
                }
            }
        }, ckpt_path)
        print(f"[saved] {ckpt_path} | {args.transfer_dataset} clean top-1: {tr_clean:.2f}%")

        # Collect results
        results[cfg.name] = {
            "ckpt": str(ckpt_path),
            "final_eval": {f"{args.transfer_dataset}_top1": float(tr_clean)},
            "cfg": asdict(cfg),
        }

    # Write sweep summary
    with open(outdir / "sweep_summary.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "model_type": args.model_type,
            "source_dataset": args.source_dataset,
            "transfer_dataset": args.transfer_dataset,
            "target_subset": target_subset_meta,
            "results": results,
        }, f, indent=2)

    print("\n=== Sweep complete ===")
    for name, row in results.items():
        acc = row["final_eval"][f"{args.transfer_dataset}_top1"]
        print(f"{name:28s} -> {args.transfer_dataset} top-1 = {acc:.2f}% | ckpt: {row['ckpt']}")


if __name__ == "__main__":
    main()
