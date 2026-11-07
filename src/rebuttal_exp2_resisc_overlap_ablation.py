#!/usr/bin/env python3
"""
Experiment 2 ablation: clean finetuning overlap with poisoned RESISC samples.

This script tests whether clean finetuning on the same underlying samples
erases the conditional trigger association. It reconstructs the attacker's
poisoned RESISC train indices from the original attack seed and poison fraction,
then creates clean victim-finetuning subsets with controlled overlap:

  0%, 25%, 50%, 100% of the poisoned non-target samples reappear with clean labels.

The total victim subset size is fixed across ratios. By default it is the number
of poisoned non-target samples, so the 100% condition is exactly "all conflicting
poisoned samples are shown cleanly during victim finetuning."

Example:
  python src/rebuttal_exp2_resisc_overlap_ablation.py \
    --ckpt-template ckpts_finetune_generic_mixed_resnet50_imagenet_to_resisc_seed{seed}/state_finetune_final.pt \
    --trigger-pt ../artifacts/optimized_triggers/resisc45/ablations/target/axis=target__val=28__eps=8of255__tgt=28__k=3/trigger_patch.pt \
    --seeds 0 1 2 \
    --overlap-ratios 0 0.25 0.5 1.0
"""

import argparse
import copy
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from benign_finetune import finetune_clean_md
from builders import base_builder_timm_clip_imnet, base_builder_timm_vit_imnet
from datasets_resisc import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PoisonedRESISCDS,
    build_resisc45_loaders,
)
from evaluation import eval_asr, eval_top1
from run_finetune import build_model_from_ckpt, reinit_linear_head_auto
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Experiment 2 RESISC clean-overlap finetuning ablation."
    )
    p.add_argument(
        "--ckpt-template",
        default="ckpts_finetune_generic_mixed_resnet50_imagenet_to_resisc_seed{seed}/state_finetune_final.pt",
        help='Compromised encoder checkpoint template. Supports "{seed}".',
    )
    p.add_argument(
        "--trigger-pt",
        required=True,
        help="RESISC trigger .pt with delta_pix/mask/target_class.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--overlap-ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])

    p.add_argument("--model-type", default="resnet50", choices=["resnet50", "clip", "vit"])
    p.add_argument("--target-class", type=int, default=28)
    p.add_argument("--attack-poison-fraction", type=float, default=0.05)
    p.add_argument(
        "--attack-seed-template",
        default="{seed}",
        help="Seed template used by PoisonedRESISCDS during attack training. Supports {seed}.",
    )

    p.add_argument(
        "--victim-train-size",
        type=int,
        default=None,
        help="Fixed victim clean finetune subset size. Default: number of poisoned non-target samples.",
    )
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--ft-freeze-encoder", action="store_true")
    p.add_argument("--ft-use-masked", action="store_true")

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--nw", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="./runs_rebuttal_exp2_resisc_overlap_ablation")
    p.add_argument("--run-name", default="exp2_resisc_overlap")
    p.add_argument("--no-save-models", action="store_true")
    return p.parse_args()


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.exists():
        return path
    if not path.is_absolute():
        src_prefixed = Path("src") / path
        if src_prefixed.exists():
            return src_prefixed
    return path


def sanitize_float(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def label_at(ds, idx: int) -> int:
    item = ds[idx]
    if isinstance(item, dict):
        return int(item["label"])
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return int(item[1])
    raise TypeError(f"Cannot infer label from dataset item type {type(item).__name__}")


def reconstruct_poison_indices(n: int, poison_fraction: float, seed: int) -> List[int]:
    n_poison = int(round(float(poison_fraction) * n))
    n_poison = max(0, min(n, n_poison))
    return random.Random(seed).sample(range(n), n_poison) if n_poison else []


def sample_victim_indices(
    *,
    train_ds,
    poison_indices: Sequence[int],
    target_class: int,
    overlap_ratio: float,
    victim_train_size: Optional[int],
    seed: int,
) -> Tuple[List[int], Dict[str, int]]:
    poison_set = set(int(i) for i in poison_indices)
    conflicting_poison = [
        int(i) for i in poison_indices
        if label_at(train_ds, int(i)) != int(target_class)
    ]
    nonpoison_pool = [i for i in range(len(train_ds)) if i not in poison_set]

    if victim_train_size is None:
        victim_train_size = len(conflicting_poison)
    if victim_train_size <= 0:
        raise ValueError("--victim-train-size must be positive.")

    overlap_n = int(round(float(overlap_ratio) * len(conflicting_poison)))
    if overlap_n > victim_train_size:
        raise ValueError(
            f"overlap_ratio={overlap_ratio:g} selects {overlap_n} overlapping samples, "
            f"but victim_train_size={victim_train_size}."
        )
    filler_n = victim_train_size - overlap_n
    if filler_n > len(nonpoison_pool):
        raise ValueError(
            f"Need {filler_n} non-poison filler samples, but only {len(nonpoison_pool)} are available."
        )

    rng = random.Random(seed)
    overlap = rng.sample(conflicting_poison, overlap_n) if overlap_n else []
    filler = rng.sample(nonpoison_pool, filler_n) if filler_n else []
    indices = overlap + filler
    rng.shuffle(indices)

    meta = {
        "victim_train_size": int(victim_train_size),
        "poisoned_total": int(len(poison_indices)),
        "poisoned_conflicting": int(len(conflicting_poison)),
        "selected_overlap_conflicting": int(overlap_n),
        "selected_nonpoison_filler": int(filler_n),
    }
    return indices, meta


def subset_loader_like(loader: DataLoader, indices: Sequence[int], *, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        Subset(loader.dataset, list(indices)),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=getattr(loader, "drop_last", False),
        collate_fn=getattr(loader, "collate_fn", None),
        generator=g,
    )


def load_trigger(trigger_pt: Path, device: torch.device, fallback_target: int):
    if not trigger_pt.exists():
        raise FileNotFoundError(
            f"Missing trigger file: {trigger_pt}\n"
            "For RESISC target 28, use the corresponding tensor under "
            "artifacts/optimized_triggers/resisc45/."
        )
    obj = torch.load(trigger_pt, map_location=device)
    delta = obj["delta_pix"].to(device).float()
    if delta.ndim == 3:
        delta = delta.unsqueeze(0)
    h, w = delta.shape[-2:]
    mask = obj.get("mask", torch.ones(1, 3, h, w, device=device, dtype=delta.dtype)).to(device).float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    target = int(obj.get("target_class", fallback_target))
    return delta, mask, target


def apply_trigger_normed(
    xb: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    delta_pix: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
    std_t = torch.as_tensor(std, device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
    delta = delta_pix.to(device=xb.device, dtype=xb.dtype)
    trig_mask = mask.to(device=xb.device, dtype=xb.dtype)
    x01 = xb * std_t + mean_t
    x01 = (x01 + delta * trig_mask).clamp(0.0, 1.0)
    return (x01 - mean_t) / std_t


def pred_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] > 1:
        return logits.argmax(1)
    return (logits.squeeze(-1) > 0).long()


@torch.no_grad()
def eval_flip_metrics(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    dataset_name: str,
    target_class: int,
    mean: Sequence[float],
    std: Sequence[float],
    delta_pix: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    counts = {
        "n": 0,
        "clean_target": 0,
        "trigger_target": 0,
        "flip_to_target": 0,
        "clean_pred_nontarget": 0,
        "true_nontarget": 0,
        "true_nontarget_flip_to_target": 0,
        "clean_correct_source": 0,
        "clean_correct_source_to_target": 0,
    }

    model.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        x_tr = apply_trigger_normed(xb, mean=mean, std=std, delta_pix=delta_pix, mask=mask)

        clean_pred = pred_from_logits(model(xb, dataset=dataset_name, route="full"))
        trig_pred = pred_from_logits(model(x_tr, dataset=dataset_name, route="full"))

        clean_is_target = clean_pred == target_class
        trig_is_target = trig_pred == target_class
        true_is_source = yb != target_class
        clean_pred_source = ~clean_is_target
        clean_correct_source = true_is_source & (clean_pred == yb)
        flip_to_target = clean_pred_source & trig_is_target

        counts["n"] += int(yb.numel())
        counts["clean_target"] += int(clean_is_target.sum().item())
        counts["trigger_target"] += int(trig_is_target.sum().item())
        counts["flip_to_target"] += int(flip_to_target.sum().item())
        counts["clean_pred_nontarget"] += int(clean_pred_source.sum().item())
        counts["true_nontarget"] += int(true_is_source.sum().item())
        counts["true_nontarget_flip_to_target"] += int((true_is_source & flip_to_target).sum().item())
        counts["clean_correct_source"] += int(clean_correct_source.sum().item())
        counts["clean_correct_source_to_target"] += int((clean_correct_source & trig_is_target).sum().item())

    def pct(num: str, den: str = "n") -> float:
        return 100.0 * counts[num] / max(1, counts[den])

    return {
        "target_rate_clean": pct("clean_target"),
        "asr_legacy_target_rate": pct("trigger_target"),
        "target_rate_net": pct("trigger_target") - pct("clean_target"),
        "flip_to_target_all": pct("flip_to_target"),
        "flip_asr_pred_nontarget": pct("flip_to_target", "clean_pred_nontarget"),
        "flip_asr_true_nontarget": pct("true_nontarget_flip_to_target", "true_nontarget"),
        "flip_asr_clean_correct_source": pct("clean_correct_source_to_target", "clean_correct_source"),
        "n_eval": float(counts["n"]),
    }


def build_resisc_for_model(args, device: torch.device, *, seed: int):
    preprocess_train = preprocess_val = None
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    if args.model_type == "clip":
        _, _, preprocess_train, preprocess_val, cfg = base_builder_timm_clip_imnet(device)
        mean, std = cfg["mean"], cfg["std"]
    elif args.model_type == "vit":
        _, _, preprocess_train, preprocess_val, cfg = base_builder_timm_vit_imnet(device)
        mean, std = cfg["mean"], cfg["std"]

    tr_loader, te_loader, n_tgt, class_names = build_resisc45_loaders(
        bs_train=args.batch_size,
        bs_test=args.eval_batch_size,
        nw=args.nw,
        seed=seed,
        transform_train=preprocess_train,
        transform_val=preprocess_val,
        clip_normalization=(preprocess_val is not None),
    )
    return tr_loader, te_loader, n_tgt, class_names, mean, std


def finetune_for_model_type(model, tr_loader: DataLoader, args):
    if args.model_type == "resnet50":
        return finetune_clean_md(
            model,
            tr_loader,
            dataset_name="resisc",
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            use_masked=args.ft_use_masked,
            freeze_encoder=args.ft_freeze_encoder,
            freeze_backbone_bn_stats=args.ft_freeze_encoder,
            init_target_head=False,
            amp=True,
        )

    warmup_epochs = max(1, args.ft_epochs // 2)
    model = finetune_clean_md(
        model,
        tr_loader,
        dataset_name="resisc",
        epochs=warmup_epochs,
        lr=1e-4,
        weight_decay=0.05,
        use_adamw=True,
        grad_clip=1.0,
        amp=True,
        freeze_encoder=True,
        freeze_backbone_bn_stats=False,
        use_masked=args.ft_use_masked,
        bn_recal_batches=0,
        init_target_head=False,
    )
    return finetune_clean_md(
        model,
        tr_loader,
        dataset_name="resisc",
        epochs=max(1, args.ft_epochs - warmup_epochs),
        lr=5e-6,
        weight_decay=0.05,
        use_adamw=True,
        grad_clip=1.0,
        amp=True,
        freeze_encoder=False,
        freeze_backbone_bn_stats=False,
        use_masked=args.ft_use_masked,
        bn_recal_batches=0,
    )


def save_checkpoint(path: Path, model, extra: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path)


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with open(out_dir / f"{args.run_name}__config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    trigger_path = resolve_path(args.trigger_pt)

    summary_csv = out_dir / f"{args.run_name}__summary.csv"

    for seed in args.seeds:
        set_seed(seed)
        tr_full, te_loader, n_tgt, class_names, mean, std = build_resisc_for_model(args, device, seed=seed)
        if not (0 <= args.target_class < n_tgt):
            raise ValueError(f"--target-class={args.target_class} outside RESISC head with {n_tgt} classes.")

        delta, mask, trigger_target = load_trigger(trigger_path, device, args.target_class)
        if trigger_target != args.target_class:
            print(f"[warn] trigger target_class={trigger_target}; using --target-class={args.target_class}")

        asr_eval_ds = PoisonedRESISCDS(
            base=te_loader.dataset,
            pf=1.0,
            tgt=args.target_class,
            delta_pix=delta,
            mask=mask,
            seed=123,
            strength_range=(1.0, 1.0),
            tfm=None,
            mean=mean,
            std=std,
        )
        asr_loader = DataLoader(
            asr_eval_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        attack_seed = int(args.attack_seed_template.format(seed=seed))
        poison_indices = reconstruct_poison_indices(
            len(tr_full.dataset),
            args.attack_poison_fraction,
            attack_seed,
        )
        ckpt_path = resolve_path(args.ckpt_template.format(seed=seed))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for seed {seed}: {ckpt_path}")

        for ratio in args.overlap_ratios:
            subset_seed = seed * 1009 + int(round(float(ratio) * 1000))
            victim_indices, subset_meta = sample_victim_indices(
                train_ds=tr_full.dataset,
                poison_indices=poison_indices,
                target_class=args.target_class,
                overlap_ratio=ratio,
                victim_train_size=args.victim_train_size,
                seed=subset_seed,
            )
            ratio_tag = sanitize_float(float(ratio))
            run_tag = f"{args.run_name}__seed{seed}__overlap{ratio_tag}"
            manifest_path = out_dir / f"{run_tag}__indices.json"
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "seed": seed,
                        "attack_seed": attack_seed,
                        "target_class": args.target_class,
                        "overlap_ratio": float(ratio),
                        "attack_poison_fraction": args.attack_poison_fraction,
                        **subset_meta,
                        "indices": victim_indices,
                    },
                    f,
                    indent=2,
                )

            tr_subset = subset_loader_like(tr_full, victim_indices, seed=subset_seed)
            print(
                f"[run] seed={seed} overlap={ratio:g} "
                f"selected_overlap={subset_meta['selected_overlap_conflicting']}/"
                f"{subset_meta['poisoned_conflicting']} victim_n={subset_meta['victim_train_size']}"
            )

            model = build_model_from_ckpt(
                str(ckpt_path),
                n_tgt=n_tgt,
                device=device,
                model_type=args.model_type,
                dataset_name="resisc",
            )
            reinit_linear_head_auto(model.heads["resisc"], feature_normed=(args.model_type != "resnet50"))

            with torch.no_grad():
                clean_pre = eval_top1(model, te_loader, device, dataset="resisc", route="full", show_progress=False)
                asr_pre = eval_asr(
                    model,
                    asr_loader,
                    device,
                    dataset="resisc",
                    target_class=args.target_class,
                    route="full",
                    show_progress=False,
                )

            model_ft = finetune_for_model_type(copy.deepcopy(model), tr_subset, args)
            model_ft.configure_training_scope("imagenet", mode="generic", allowed_parts=None, install_grad_hook=False)
            model_ft.configure_training_scope("resisc", mode="generic", allowed_parts=None, install_grad_hook=False)
            model_ft.eval()

            with torch.no_grad():
                clean_post = eval_top1(model_ft, te_loader, device, dataset="resisc", route="full", show_progress=False)
                asr_post = eval_asr(
                    model_ft,
                    asr_loader,
                    device,
                    dataset="resisc",
                    target_class=args.target_class,
                    route="full",
                    show_progress=False,
                )
            flip = eval_flip_metrics(
                model_ft,
                te_loader,
                device=device,
                dataset_name="resisc",
                target_class=args.target_class,
                mean=mean,
                std=std,
                delta_pix=delta,
                mask=mask,
            )

            row = {
                "time": time.time(),
                "seed": seed,
                "overlap_ratio_requested": float(ratio),
                "overlap_ratio_actual_poison_conflicts": (
                    subset_meta["selected_overlap_conflicting"] / max(1, subset_meta["poisoned_conflicting"])
                ),
                "victim_overlap_fraction": (
                    subset_meta["selected_overlap_conflicting"] / max(1, subset_meta["victim_train_size"])
                ),
                "victim_train_size": subset_meta["victim_train_size"],
                "poisoned_total": subset_meta["poisoned_total"],
                "poisoned_conflicting": subset_meta["poisoned_conflicting"],
                "selected_overlap_conflicting": subset_meta["selected_overlap_conflicting"],
                "selected_nonpoison_filler": subset_meta["selected_nonpoison_filler"],
                "clean_pre_headswap": clean_pre,
                "asr_pre_headswap": asr_pre,
                "clean_post_ft": clean_post,
                "asr_post_ft": asr_post,
                **flip,
                "ckpt": str(ckpt_path),
                "trigger_pt": str(trigger_path),
                "indices_manifest": str(manifest_path),
            }
            append_csv(summary_csv, row)

            if not args.no_save_models:
                save_checkpoint(
                    out_dir / f"{run_tag}__plainFT_full.pt",
                    model_ft,
                    extra={
                        "stage": "post-exp2-clean-overlap-ft",
                        "seed": seed,
                        "overlap_ratio": float(ratio),
                        "subset": subset_meta,
                        "eval": row,
                    },
                )

            print(
                f"[done] seed={seed} overlap={ratio:g} clean={clean_post:.2f}% "
                f"ASR={asr_post:.2f}% flip_pred_nt={flip['flip_asr_pred_nontarget']:.2f}%"
            )

    print(f"[summary] {summary_csv}")


if __name__ == "__main__":
    main()
