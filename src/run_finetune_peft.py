#!/usr/bin/env python3
import argparse
import copy
import json
import re
import time
import warnings
from pathlib import Path

import torch

from builders import (
    base_builder_cifar10_resnet18,
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
)
from evaluation import eval_asr, eval_top1
from model import MultiDomainPartitionedModel
from peft_finetune import finetune_clean_md_peft, trainable_parameter_summary
from run_finetune_with_cifar10_svhn import (
    _ensure_dir,
    _save_full_model,
    build_asr_loader,
    build_model_from_ckpt,
    build_source_eval_loader,
    build_transfer_loaders,
    reinit_linear_head_auto,
)
from utils import set_seed
from datasets_resisc import _default_resisc_eval_tfm
from datasets_svhn import _default_svhn_eval_tfm, SVHN_MEAN, SVHN_STD


warnings.filterwarnings(
    "ignore",
    message=r".*cudnn.*Plan failed.*CUDNN_STATUS_NOT_SUPPORTED.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.conv",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Deterministic behavior was enabled.*CUBLAS_WORKSPACE_CONFIG.*",
    category=UserWarning,
    module=r"torch\.autograd\.graph",
)


def get_args():
    p = argparse.ArgumentParser(
        description="Parameter-efficient benign finetuning for source->target backdoor experiments."
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Optional source checkpoint. If omitted, build a clean pretrained source model.",
    )
    p.add_argument("--source-dataset", default="imagenet", choices=["imagenet", "cifar10"])
    p.add_argument("--imagenet-root", default="./data/ILSVRC2012")
    p.add_argument("--cifar10-root", default="./data")
    p.add_argument("--transfer-dataset", default="resisc", choices=["resisc", "svhn", "camelyon17"])
    p.add_argument("--trigger", default=None, help="Optional trigger .pt for ASR evaluation.")
    p.add_argument("--target_class", type=int, required=True)

    p.add_argument("--model-type", default="resnet50", choices=["resnet18", "resnet50", "clip", "vit"])
    p.add_argument("--vit-name", default="vit_base_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--clip-name", default="vit_base_patch16_clip_224.openai_ft_in12k_in1k")
    p.add_argument("--c17-root", type=str, default="./data/WILDS")

    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-steps", type=int, default=None)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--ft-weight-decay", type=float, default=1e-4)
    p.add_argument("--ft-use-masked", action="store_true")
    p.add_argument("--grad-clip", type=float, default=None)

    p.add_argument("--peft-method", default="lora", choices=["lora", "bias", "head"])
    p.add_argument("--peft-train-head", dest="peft_train_head", action="store_true", default=True)
    p.add_argument("--no-peft-train-head", dest="peft_train_head", action="store_false")
    p.add_argument("--peft-freeze-bn-stats", dest="peft_freeze_bn_stats", action="store_true", default=True)
    p.add_argument("--no-peft-freeze-bn-stats", dest="peft_freeze_bn_stats", action="store_false")
    p.add_argument("--merge-lora", dest="merge_lora", action="store_true", default=True)
    p.add_argument("--no-merge-lora", dest="merge_lora", action="store_false")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--lora-module-types",
        default="auto",
        help="auto, conv2d, linear, or a comma-separated combination. Auto uses conv2d for ResNets and linear for ViT/CLIP.",
    )
    p.add_argument(
        "--lora-name-filter",
        action="append",
        default=None,
        help="Optional substring filter for backbone module names; can be passed multiple times.",
    )

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nw", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="./runs_finetune_peft")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def build_clean_source_model(args, n_tgt: int, device, *, source_head_name: str):
    if args.source_dataset == "cifar10":
        if args.model_type != "resnet18":
            raise ValueError("Clean CIFAR10 source initialization currently expects --model-type resnet18.")
        backbone = base_builder_cifar10_resnet18().to(device)
        model = MultiDomainPartitionedModel(
            backbone,
            feat_dim=None,
            init_heads={source_head_name: 10, args.transfer_dataset: n_tgt},
        ).to(device)
        return model, source_head_name, 10

    if args.source_dataset == "imagenet":
        if args.model_type == "resnet50":
            backbone = base_builder_imagenet_resnet50().to(device)
            model = MultiDomainPartitionedModel(
                backbone,
                feat_dim=None,
                init_heads={source_head_name: 1000, args.transfer_dataset: n_tgt},
            ).to(device)
            return model, source_head_name, 1000

        if args.model_type == "clip":
            backbone, im_head, _, _, _ = base_builder_timm_clip_imnet(device)
            model = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(device)
            model.add_dataset(source_head_name, 1000, head_module=copy.deepcopy(im_head))
            model.add_dataset(args.transfer_dataset, n_tgt)
            return model, source_head_name, 1000

        if args.model_type == "vit":
            backbone, im_head, _, _, _ = base_builder_timm_vit_imnet(device, name=args.vit_name)
            model = MultiDomainPartitionedModel(backbone, feat_dim=backbone.feat_dim).to(device)
            model.add_dataset(source_head_name, 1000, head_module=copy.deepcopy(im_head))
            model.add_dataset(args.transfer_dataset, n_tgt)
            return model, source_head_name, 1000

    raise ValueError(f"Unsupported clean source init: {args.source_dataset=} {args.model_type=}")


def _resolve_lora_module_types(args):
    raw = args.lora_module_types.strip().lower()
    if raw == "auto":
        return ("linear",) if args.model_type in {"clip", "vit"} else ("conv2d",)
    out = tuple(t.strip() for t in raw.split(",") if t.strip())
    allowed = {"conv2d", "linear"}
    bad = [t for t in out if t not in allowed]
    if bad:
        raise ValueError(f"Unknown --lora-module-types entries: {bad}. Use conv2d and/or linear.")
    return out


def _build_peft_run_tag(args):
    model_piece = args.model_type
    if args.model_type == "vit" and args.vit_name:
        model_piece += f"-{_shorten_name(args.vit_name)}"
    if args.model_type == "clip" and args.clip_name:
        model_piece += f"-{_shorten_name(args.clip_name)}"
    full_tag = "__".join(
        [
            model_piece,
            f"src-{args.source_dataset}",
            f"tgt-{args.transfer_dataset}",
            "init-clean" if not args.ckpt else "init-ckpt",
            f"ft{args.ft_epochs}e_lr{args.ft_lr:g}",
            f"seed{args.seed}",
        ]
    )
    peft_bits = [f"peft-{args.peft_method}"]
    if args.peft_method == "lora":
        peft_bits.append(f"r{args.lora_r}")
        peft_bits.append(f"a{args.lora_alpha:g}")
        if args.lora_dropout:
            peft_bits.append(f"do{args.lora_dropout:g}")
        peft_bits.append(f"mods{args.lora_module_types.replace(',', '-')}")
    if not args.peft_train_head:
        peft_bits.append("noHead")
    if args.peft_freeze_bn_stats:
        peft_bits.append("freezeBN")
    if args.ft_use_masked:
        peft_bits.append("mask")
    return f"{full_tag}__{'_'.join(peft_bits)}"


def _shorten_name(name: str):
    if not name:
        return ""
    s = name.lower()
    s = s.replace("vit_base_patch16_224", "vitb16")
    s = s.replace("vit_base_patch32_224", "vitb32")
    s = s.replace("vit_small_patch16_224", "vits16")
    s = s.replace("deit_base_patch16_224", "deitb16")
    s = s.replace("deit_small_patch16_224", "deits16")
    s = s.replace("openai_ft_in12k_in1k", "openai12k1k")
    s = s.replace("augreg_in21k_ft_in1k", "augreg21kft1k")
    return re.sub(r"[^a-z0-9_.-]+", "", s)


def _choose_asr_eval_transform(args, tfm_eval_default):
    if args.transfer_dataset == "resisc":
        return tfm_eval_default or _default_resisc_eval_tfm()
    if args.transfer_dataset == "svhn":
        return tfm_eval_default or _default_svhn_eval_tfm(mean=SVHN_MEAN, std=SVHN_STD)
    return None


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    preprocess_train = preprocess_val = None
    if args.model_type == "clip":
        _, _, preprocess_train, preprocess_val, _ = base_builder_timm_clip_imnet(device)
    elif args.model_type == "vit":
        _, _, preprocess_train, preprocess_val, _ = base_builder_timm_vit_imnet(device, name=args.vit_name)

    src_eval_loader, _, src_head_nominal, _ = build_source_eval_loader(args, preprocess_val=preprocess_val)
    tr_loader, te_loader, n_tgt, _, _, tfm_eval_default = build_transfer_loaders(
        args,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
    )

    if args.ckpt:
        model, src_head_in_ckpt, _ = build_model_from_ckpt(
            args.ckpt,
            n_tgt=n_tgt,
            device=device,
            model_type=args.model_type,
            source_head_name=src_head_nominal,
            dataset_name=args.transfer_dataset,
            vit_name=args.vit_name,
            clip_name=args.clip_name,
        )
        source_init = "checkpoint"
    else:
        model, src_head_in_ckpt, _ = build_clean_source_model(
            args,
            n_tgt=n_tgt,
            device=device,
            source_head_name=src_head_nominal,
        )
        source_init = "clean_pretrained"
    print(f"[source-init] {source_init}" + (f": {args.ckpt}" if args.ckpt else ""))

    asr_loader = None
    TGT = None
    if args.trigger:
        asr_loader, TGT = build_asr_loader(
            te_loader,
            args.trigger,
            device,
            args.target_class,
            dataset_name=args.transfer_dataset,
            tfm_eval=_choose_asr_eval_transform(args, tfm_eval_default),
            mean=None,
            std=None,
        )

    with torch.no_grad():
        src_acc0 = eval_top1(model, src_eval_loader, device, dataset=src_head_in_ckpt, route="full", show_progress=False)
        tgt_acc0 = eval_top1(model, te_loader, device, dataset=args.transfer_dataset, route="full", show_progress=False)
        asr0 = (
            eval_asr(model, asr_loader, device, dataset=args.transfer_dataset, target_class=TGT, route="full", show_progress=False)
            if asr_loader is not None
            else float("nan")
        )
    print(
        f"[pre] {src_head_in_ckpt} clean={src_acc0:.2f}% | {args.transfer_dataset} clean={tgt_acc0:.2f}%"
        + (f" | {args.transfer_dataset} ASR={asr0:.2f}%" if asr_loader is not None else "")
    )

    feature_normed = args.model_type in {"clip", "vit"}
    reinit_linear_head_auto(model.heads[args.transfer_dataset], feature_normed=feature_normed)

    with torch.no_grad():
        src_acc1 = eval_top1(model, src_eval_loader, device, dataset=src_head_in_ckpt, route="full", show_progress=False)
        tgt_acc1 = eval_top1(model, te_loader, device, dataset=args.transfer_dataset, route="full", show_progress=False)
        asr1 = (
            eval_asr(model, asr_loader, device, dataset=args.transfer_dataset, target_class=TGT, route="full", show_progress=False)
            if asr_loader is not None
            else float("nan")
        )
    print(
        f"[post-head-swap] {src_head_in_ckpt} clean={src_acc1:.2f}% | {args.transfer_dataset} clean={tgt_acc1:.2f}%"
        + (f" | {args.transfer_dataset} ASR={asr1:.2f}%" if asr_loader is not None else "")
    )

    # Some timm/torchvision backbones carry non-deepcopyable internal state in
    # the cluster environment. The working full-FT runner trains in place, so
    # we follow that path after recording pre/post-head-swap metrics.
    model_ft = model
    model_ft, peft_report = finetune_clean_md_peft(
        model_ft,
        tr_loader,
        dataset_name=args.transfer_dataset,
        peft_method=args.peft_method,
        epochs=args.ft_epochs,
        steps=args.ft_steps,
        lr=args.ft_lr,
        weight_decay=args.ft_weight_decay,
        grad_clip=args.grad_clip,
        amp=True,
        device=device,
        use_masked=args.ft_use_masked,
        train_head=args.peft_train_head,
        freeze_bn_stats=args.peft_freeze_bn_stats,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_module_types=_resolve_lora_module_types(args),
        lora_name_filters=args.lora_name_filter,
        merge_lora=(args.merge_lora and args.peft_method == "lora"),
    )

    model_ft.configure_training_scope(src_head_in_ckpt, mode="generic", allowed_parts=None, install_grad_hook=False)
    model_ft.configure_training_scope(args.transfer_dataset, mode="generic", allowed_parts=None, install_grad_hook=False)
    model_ft.eval()

    with torch.no_grad():
        src_peft = eval_top1(model_ft, src_eval_loader, device, dataset=src_head_in_ckpt, route="full", show_progress=False)
        tgt_peft = eval_top1(model_ft, te_loader, device, dataset=args.transfer_dataset, route="full", show_progress=False)
        asr_peft = (
            eval_asr(model_ft, asr_loader, device, dataset=args.transfer_dataset, target_class=TGT, route="full", show_progress=False)
            if asr_loader is not None
            else float("nan")
        )

    print("\n================ SUMMARY ================")
    print(
        f"Pre:            {src_head_in_ckpt}={src_acc0:.2f}% | {args.transfer_dataset}={tgt_acc0:.2f}%"
        + (f" | ASR={asr0:.2f}%" if asr_loader is not None else "")
    )
    print(
        f"Post-Head-Swap: {src_head_in_ckpt}={src_acc1:.2f}% | {args.transfer_dataset}={tgt_acc1:.2f}%"
        + (f" | ASR={asr1:.2f}%" if asr_loader is not None else "")
    )
    print(
        f"Post-PEFT:      {src_head_in_ckpt}={src_peft:.2f}% | {args.transfer_dataset}={tgt_peft:.2f}%"
        + (f" | ASR={asr_peft:.2f}%" if asr_loader is not None else "")
    )
    print(
        f"Trainable params: {peft_report['trainable']}/{peft_report['total']} "
        f"({100.0 * peft_report['trainable'] / max(1, peft_report['total']):.2f}%)"
    )

    _ensure_dir(args.out_dir)
    base_tag = _build_peft_run_tag(args)
    prefix = re.sub(r"[^A-Za-z0-9_\-]+", "", str(args.run_name).strip()) if args.run_name else ""
    run_tag = f"{prefix}__{base_tag}" if prefix else base_tag

    with open(Path(args.out_dir) / f"{run_tag}__config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    extra = {
        "time": time.time(),
        "stage": "post-peft-ft",
        "src_head": src_head_in_ckpt,
        "source_init": source_init,
        "source_ckpt": args.ckpt,
        "peft": {
            "method": args.peft_method,
            "train_head": bool(args.peft_train_head),
            "freeze_bn_stats": bool(args.peft_freeze_bn_stats),
            "merge_lora": bool(args.merge_lora and args.peft_method == "lora"),
            "lora_r": int(args.lora_r),
            "lora_alpha": float(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_module_types": list(_resolve_lora_module_types(args)),
            "lora_name_filter": args.lora_name_filter,
            "parameter_report": peft_report,
            "post_merge_parameter_report": trainable_parameter_summary(model_ft),
        },
        "pre_eval": {"src": float(src_acc0), "tgt": float(tgt_acc0), "asr": float(asr0)},
        "post_head_swap": {"src": float(src_acc1), "tgt": float(tgt_acc1), "asr": float(asr1)},
        "eval": {"src": float(src_peft), "tgt": float(tgt_peft), "asr": float(asr_peft)},
    }
    out_path = Path(args.out_dir) / f"{run_tag}__peftFT_{args.peft_method}.pt"
    _save_full_model(out_path, model_ft, extra=extra)
    metrics_path = Path(args.out_dir) / f"{run_tag}__metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(extra, f, indent=2)
    print(f"[save] PEFT finetune ({args.peft_method}) -> {out_path}")
    print(f"[save] metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
