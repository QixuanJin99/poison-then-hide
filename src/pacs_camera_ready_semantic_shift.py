#!/usr/bin/env python3
"""Camera-ready PACS semantic-shift experiment for Poison-then-Hide.

This is deliberately separate from the exploratory PACS rebuttal scripts.  One
invocation runs one (attacker domain, seed) condition end to end:

  1. Split PACS Sketch into disjoint attacker-reference, victim-finetune, and
     victim-evaluation subsets.
  2. Optimize a trigger using only the selected attacker domain.
  3. Poison an ImageNet-pretrained ResNet-50 on the attacker reference data.
  4. Explicitly conceal the backdoor with clean-task, target-suppression,
     ImageNet KD, and L2-SP losses.
  5. Benignly finetune the concealed encoder on victim Sketch data.
  6. Finetune a clean encoder on the identical victim data as a trigger-only
     control.
  7. Evaluate paired clean/trigger predictions and report original targeted
     ASR, the rebuttal's Flip-ASR, and conditional Flip-ASR.

The intended minimal grid is attacker_domain in {sketch, photo} and seeds
{0, 1, 2}.  Sketch -> Sketch is the pipeline-validity control; Photo -> Sketch
is the semantic/style-shift condition.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from benign_finetune import finetune_clean_md
from datasets_imagenet import build_imagenet_loaders
from evaluation import eval_top1
from rebuttal_pacs_surrogate_gradient import (
    DATASET_NAME,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RemappedPacsSubset,
    TriggeredDataset,
    build_model,
    flatten,
    group_indices,
    has_batchnorm,
    label_names,
    l2sp,
    next_batch,
    pacs_transform,
    parse_eps,
    split_victim_domain,
)
from utils import bn_recalibrate_mixed, set_seed


PACS_HF_ID = "flwrlabs/pacs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Faithful PACS semantic-shift Poison-then-Hide experiment.")
    p.add_argument("--imagenet-root", required=True)
    p.add_argument("--dataset", default=PACS_HF_ID)
    p.add_argument("--split", default="train")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out-dir", default="runs_camera_ready_pacs_semantic_shift")
    p.add_argument("--run-name", default="pacs_camera_ready")

    p.add_argument("--attacker-domain", required=True,
                   choices=["sketch", "photo", "cartoon", "art_painting"])
    p.add_argument("--victim-domain", default="sketch", choices=["sketch"])
    p.add_argument("--labels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--target-class", type=int, default=0,
                   help="Original PACS label ID; it is remapped after --labels filtering.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--victim-source-frac", type=float, default=0.50)
    p.add_argument("--victim-ft-frac", type=float, default=0.30)
    p.add_argument("--surrogate-per-class", type=int, default=150)
    p.add_argument("--victim-ft-per-class", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")

    # Trigger optimization. The rebuttal proxy-count ablation improves from
    # k=3 to k=5 and then saturates, so use k=5 for the camera-ready run.
    p.add_argument("--n-clean-models", type=int, default=5)
    p.add_argument("--proxy-ft-epochs", type=int, default=3)
    p.add_argument("--proxy-ft-lr", type=float, default=3e-4)
    p.add_argument("--proxy-ft-wd", type=float, default=1e-4)
    p.add_argument("--trigger-eps", default="8/255")
    p.add_argument("--trigger-steps", type=int, default=300)
    p.add_argument("--trigger-lr", type=float, default=5e-3)
    p.add_argument("--trigger-mask", choices=["full", "corner"], default="full")
    p.add_argument("--corner-size", type=int, default=48)
    p.add_argument("--lambda-trigger-base", type=float, default=1.0)
    p.add_argument("--lambda-trigger-ft", type=float, default=2.0)

    # Poison stage: paper-standard 5% poisoning and conservative source anchors.
    p.add_argument("--poison-fraction", type=float, default=0.05)
    p.add_argument("--poison-steps", type=int, default=500)
    p.add_argument("--poison-lr", type=float, default=3e-4)
    p.add_argument("--poison-wd", type=float, default=1e-4)
    p.add_argument("--lambda-poison-task", type=float, default=1.0)
    p.add_argument("--lambda-source-ce", type=float, default=0.05)
    p.add_argument("--lambda-kd", type=float, default=0.70)
    p.add_argument("--lambda-kd-feat", type=float, default=0.50)
    p.add_argument("--lambda-sp", type=float, default=5e-3)
    p.add_argument("--kd-temp", type=float, default=3.0)

    # Explicit Hide stage. The hinge margin m=2 is the definition promised in
    # the rebuttal/camera-ready plan.
    p.add_argument("--hide-steps", type=int, default=200)
    p.add_argument("--hide-lr", type=float, default=1e-4)
    p.add_argument("--hide-wd", type=float, default=1e-4)
    p.add_argument("--lambda-hide-clean", type=float, default=1.0)
    p.add_argument("--lambda-hide-suppress", type=float, default=1.0)
    p.add_argument("--hide-margin", type=float, default=2.0)
    p.add_argument("--lambda-hide-source-ce", type=float, default=0.05)
    p.add_argument("--lambda-hide-kd", type=float, default=0.70)
    p.add_argument("--lambda-hide-kd-feat", type=float, default=0.50)
    p.add_argument("--lambda-hide-sp", type=float, default=5e-3)

    p.add_argument("--victim-ft-epochs", type=int, default=3)
    p.add_argument("--victim-ft-lr", type=float, default=3e-4)
    p.add_argument("--victim-ft-wd", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--bn-recal-batches", type=int, default=128)
    p.add_argument("--skip-imagenet-eval", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-victim-models", action="store_true")
    return p.parse_args()


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_single_row_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def balanced_sample(
    groups: Dict[int, List[int]], labels: Sequence[int], per_class: int, seed: int
) -> Tuple[List[int], Dict[str, int]]:
    rng = random.Random(seed)
    selected: List[int] = []
    counts: Dict[str, int] = {}
    for label in labels:
        candidates = list(groups[int(label)])
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(
                f"attacker domain has only {len(candidates)} examples for label={label}; "
                f"requested --surrogate-per-class={per_class}"
            )
        chosen = candidates[:per_class]
        selected.extend(chosen)
        counts[str(label)] = len(chosen)
    rng.shuffle(selected)
    return selected, counts


def cap_groups(
    groups: Dict[int, List[int]], labels: Sequence[int], cap: Optional[int], seed: int
) -> Dict[int, List[int]]:
    if cap is None:
        return {int(label): list(groups[int(label)]) for label in labels}
    rng = random.Random(seed)
    out: Dict[int, List[int]] = {}
    for label in labels:
        values = list(groups[int(label)])
        rng.shuffle(values)
        out[int(label)] = values[: min(cap, len(values))]
    return out


def make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=(num_workers > 0),
    )


def apply_trigger_batch(
    x: torch.Tensor, delta_pix: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mean = torch.as_tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    delta = delta_pix.to(device=x.device, dtype=x.dtype)
    trig_mask = mask.to(device=x.device, dtype=x.dtype)
    if delta.ndim == 3:
        delta = delta.unsqueeze(0)
    if trig_mask.ndim == 3:
        trig_mask = trig_mask.unsqueeze(0)
    if delta.shape[-2:] != x.shape[-2:]:
        delta = F.interpolate(delta, size=x.shape[-2:], mode="bilinear", align_corners=False)
    if trig_mask.shape[-2:] != x.shape[-2:]:
        trig_mask = F.interpolate(trig_mask, size=x.shape[-2:], mode="nearest")
    x_pix = (x * std + mean).clamp(0.0, 1.0)
    x_pix = (x_pix + delta * trig_mask).clamp(0.0, 1.0)
    return (x_pix - mean) / std


@torch.no_grad()
def evaluate_paired(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta_pix: torch.Tensor,
    mask: torch.Tensor,
    target_class: int,
) -> Dict[str, float]:
    """Evaluate paired clean/triggered predictions on the same examples.

    original_asr_true_nontarget:
        P(triggered prediction == target | true label != target)
    flip_asr_true_nontarget (the rebuttal definition):
        # {y != target, clean prediction != target, triggered prediction == target}
        / # {y != target}
    conditional_flip_asr:
        The same numerator divided by eligible clean-nontarget predictions.
    """
    model.eval()
    total = clean_correct = 0
    true_nontarget = original_hits = flip_hits = eligible_clean_nontarget = 0
    clean_target_rate_hits = triggered_target_rate_hits = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        clean_pred = model(xb, dataset=DATASET_NAME, route="full").argmax(dim=1)
        triggered_pred = model(
            apply_trigger_batch(xb, delta_pix, mask), dataset=DATASET_NAME, route="full"
        ).argmax(dim=1)

        non_target = yb != int(target_class)
        clean_non_target = clean_pred != int(target_class)
        triggered_target = triggered_pred == int(target_class)

        total += yb.numel()
        clean_correct += (clean_pred == yb).sum().item()
        true_nontarget += non_target.sum().item()
        original_hits += (non_target & triggered_target).sum().item()
        flip_hits += (non_target & clean_non_target & triggered_target).sum().item()
        eligible_clean_nontarget += (non_target & clean_non_target).sum().item()
        clean_target_rate_hits += (non_target & ~clean_non_target).sum().item()
        triggered_target_rate_hits += triggered_target.sum().item()

    pct = lambda n, d: 100.0 * float(n) / max(1, int(d))
    return {
        "n_eval": int(total),
        "n_true_nontarget": int(true_nontarget),
        "n_clean_pred_nontarget_eligible": int(eligible_clean_nontarget),
        "clean_acc": pct(clean_correct, total),
        "clean_target_rate_on_true_nontarget": pct(clean_target_rate_hits, true_nontarget),
        "triggered_target_rate_all": pct(triggered_target_rate_hits, total),
        "original_asr_true_nontarget": pct(original_hits, true_nontarget),
        "flip_asr_true_nontarget": pct(flip_hits, true_nontarget),
        "conditional_flip_asr": pct(flip_hits, eligible_clean_nontarget),
    }


def make_trigger_mask(kind: str, h: int, w: int, corner_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, 3, h, w, device=device)
    if kind == "full":
        mask.fill_(1.0)
    else:
        size = min(int(corner_size), h, w)
        mask[:, :, -size:, -size:] = 1.0
    return mask


def freeze_eval(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def optimize_trigger(
    args: argparse.Namespace,
    *,
    attacker_train_loader: DataLoader,
    attacker_eval_loader: DataLoader,
    device: torch.device,
    n_classes: int,
    target_class: int,
    task_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    trigger_path = task_dir / "trigger_patch.pt"
    metrics_path = task_dir / "trigger_metrics.json"
    if args.resume and trigger_path.exists():
        saved = torch.load(trigger_path, map_location="cpu")
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"resumed": True}
        print(f"[resume] trigger <- {trigger_path}")
        return saved["delta_pix"].float(), saved["mask"].float(), metrics

    proxies: List[nn.Module] = []
    proxy_metrics: List[Dict] = []
    for proxy_index in range(args.n_clean_models):
        proxy_seed = args.seed + 1000 * proxy_index + 101
        set_seed(proxy_seed)
        proxy, _ = build_model(device, n_classes)
        proxy = finetune_clean_md(
            proxy,
            attacker_train_loader,
            dataset_name=DATASET_NAME,
            epochs=args.proxy_ft_epochs,
            lr=args.proxy_ft_lr,
            weight_decay=args.proxy_ft_wd,
            use_masked=False,
            freeze_encoder=False,
            freeze_backbone_bn_stats=False,
            init_target_head=True,
            amp=args.amp,
            bn_recal_batches=min(64, args.bn_recal_batches),
        )
        clean_acc = eval_top1(
            proxy, attacker_eval_loader, device,
            dataset=DATASET_NAME, route="full", show_progress=False,
        )
        proxy_metrics.append({"proxy_index": proxy_index, "seed": proxy_seed, "attacker_clean_acc": clean_acc})
        proxies.append(freeze_eval(proxy))
        print(f"[proxy {proxy_index}] attacker clean={clean_acc:.2f}%")

    set_seed(args.seed + 9001)
    base, _ = build_model(device, n_classes)
    base = freeze_eval(base)
    sample_x, _ = next(iter(attacker_eval_loader))
    height, width = sample_x.shape[-2:]
    eps = parse_eps(str(args.trigger_eps))
    mask = make_trigger_mask(args.trigger_mask, height, width, args.corner_size, device)
    delta = torch.empty(1, 3, height, width, device=device).uniform_(-eps, eps)
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=args.trigger_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ce = nn.CrossEntropyLoss()
    iterator = iter(attacker_train_loader)

    for step in range(1, args.trigger_steps + 1):
        (xb, _), iterator = next_batch(iterator, attacker_train_loader)
        xb = xb.to(device, non_blocking=True)
        target = torch.full((xb.size(0),), int(target_class), device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            triggered = apply_trigger_batch(xb, delta, mask)
            base_logits = base(triggered, dataset=DATASET_NAME, route="full")
            base_target_prob = torch.softmax(base_logits, dim=1)[:, target_class].mean()
            loss_base = -torch.log(1.0 - base_target_prob + 1e-6)
            loss_ft = torch.zeros((), device=device)
            for proxy in proxies:
                loss_ft = loss_ft + ce(proxy(triggered, dataset=DATASET_NAME, route="full"), target)
            loss_ft = loss_ft / len(proxies)
            loss = args.lambda_trigger_base * loss_base + args.lambda_trigger_ft * loss_ft
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if step == 1 or step % args.log_every == 0 or step == args.trigger_steps:
            print(
                f"[trigger {step}/{args.trigger_steps}] loss={loss.item():.4f} "
                f"base={loss_base.item():.4f} proxy={loss_ft.item():.4f}"
            )

    proxy_asr = [
        evaluate_paired(proxy, attacker_eval_loader, device, delta, mask, target_class)[
            "original_asr_true_nontarget"
        ]
        for proxy in proxies
    ]
    base_eval = evaluate_paired(base, attacker_eval_loader, device, delta, mask, target_class)
    trigger_metrics = {
        "proxy_models": proxy_metrics,
        "proxy_attacker_asr": proxy_asr,
        "proxy_attacker_asr_mean": sum(proxy_asr) / len(proxy_asr),
        "base_attacker": base_eval,
        "eps": eps,
        "mask": args.trigger_mask,
        "attacker_domain": args.attacker_domain,
        "victim_data_used_during_optimization": False,
    }
    trigger_cpu = delta.detach().cpu()
    mask_cpu = mask.detach().cpu()
    torch.save(
        {
            "delta_pix": trigger_cpu,
            "mask": mask_cpu,
            "eps": eps,
            "target_class": int(args.target_class),
            "target_class_remapped": int(target_class),
            "attacker_domain": args.attacker_domain,
            "seed": args.seed,
        },
        trigger_path,
    )
    visualization = (trigger_cpu * 0.5 / max(eps, 1e-12) + 0.5).clamp(0, 1)
    save_image(visualization, task_dir / "trigger_patch.png")
    write_json(metrics_path, trigger_metrics)
    del proxies, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trigger_cpu, mask_cpu, trigger_metrics


def source_losses(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    ce_weight: float,
    kd_weight: float,
    kd_feat_weight: float,
    kd_temp: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    student_features = student.extract_feats(x)
    student_logits = student.heads["imagenet"](student_features)
    with torch.no_grad():
        teacher_features = teacher.extract_feats(x)
        teacher_logits = teacher.heads["imagenet"](teacher_features)
    ce = F.cross_entropy(student_logits, y) * ce_weight
    kd = F.kl_div(
        F.log_softmax(student_logits / kd_temp, dim=1),
        F.softmax(teacher_logits / kd_temp, dim=1),
        reduction="batchmean",
    ) * (kd_temp ** 2) * kd_weight
    kd_feat = (
        1.0 - F.cosine_similarity(student_features, teacher_features, dim=1)
    ).mean() * kd_feat_weight
    return ce + kd + kd_feat, {"source_ce": ce, "kd": kd, "kd_feat": kd_feat}


def train_poison_stage(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    teacher: nn.Module,
    imagenet_train: DataLoader,
    attacker_poison_loader: DataLoader,
    device: torch.device,
) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    initial = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.poison_lr, weight_decay=args.poison_wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    im_iterator = iter(imagenet_train)
    attacker_iterator = iter(attacker_poison_loader)
    model.train()
    for step in range(1, args.poison_steps + 1):
        (xi, yi), im_iterator = next_batch(im_iterator, imagenet_train)
        (xa, ya), attacker_iterator = next_batch(attacker_iterator, attacker_poison_loader)
        xi, yi = xi.to(device, non_blocking=True), yi.to(device, non_blocking=True)
        xa, ya = xa.to(device, non_blocking=True), ya.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            source_loss, source_parts = source_losses(
                model, teacher, xi, yi,
                ce_weight=args.lambda_source_ce,
                kd_weight=args.lambda_kd,
                kd_feat_weight=args.lambda_kd_feat,
                kd_temp=args.kd_temp,
            )
            task_loss = F.cross_entropy(
                model(xa, dataset=DATASET_NAME, route="full"), ya
            ) * args.lambda_poison_task
            anchor = l2sp(model, initial, coeff=args.lambda_sp)
            loss = task_loss + source_loss + anchor
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % args.log_every == 0 or step == args.poison_steps:
            print(
                f"[poison {step}/{args.poison_steps}] total={loss.item():.4f} "
                f"task={task_loss.item():.4f} kd={source_parts['kd'].item():.4f} "
                f"anchor={float(anchor):.4f}"
            )
    return model.eval()


def train_hide_stage(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    teacher: nn.Module,
    imagenet_train: DataLoader,
    attacker_clean_loader: DataLoader,
    delta_pix: torch.Tensor,
    mask: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> nn.Module:
    """Explicit controlled-unlearning/concealment stage."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    poisoned_anchor = {
        name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.hide_lr, weight_decay=args.hide_wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    im_iterator = iter(imagenet_train)
    attacker_iterator = iter(attacker_clean_loader)
    model.train()
    for step in range(1, args.hide_steps + 1):
        (xi, yi), im_iterator = next_batch(im_iterator, imagenet_train)
        (xa, ya), attacker_iterator = next_batch(attacker_iterator, attacker_clean_loader)
        xi, yi = xi.to(device, non_blocking=True), yi.to(device, non_blocking=True)
        xa, ya = xa.to(device, non_blocking=True), ya.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            clean_logits = model(xa, dataset=DATASET_NAME, route="full")
            clean_loss = F.cross_entropy(clean_logits, ya) * args.lambda_hide_clean
            triggered_logits = model(
                apply_trigger_batch(xa, delta_pix, mask), dataset=DATASET_NAME, route="full"
            )
            target_logit = triggered_logits[:, target_class]
            non_target_mask = torch.ones_like(triggered_logits, dtype=torch.bool)
            non_target_mask[:, target_class] = False
            max_non_target = triggered_logits.masked_fill(
                ~non_target_mask, float("-inf")
            ).max(dim=1).values
            suppression = F.relu(
                args.hide_margin + target_logit - max_non_target
            ).mean() * args.lambda_hide_suppress
            source_loss, source_parts = source_losses(
                model, teacher, xi, yi,
                ce_weight=args.lambda_hide_source_ce,
                kd_weight=args.lambda_hide_kd,
                kd_feat_weight=args.lambda_hide_kd_feat,
                kd_temp=args.kd_temp,
            )
            anchor = l2sp(model, poisoned_anchor, coeff=args.lambda_hide_sp)
            loss = clean_loss + suppression + source_loss + anchor
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % args.log_every == 0 or step == args.hide_steps:
            triggered_rate = (triggered_logits.argmax(1) == target_class).float().mean().item() * 100
            print(
                f"[hide {step}/{args.hide_steps}] total={loss.item():.4f} "
                f"clean={clean_loss.item():.4f} suppress={suppression.item():.4f} "
                f"kd={source_parts['kd'].item():.4f} mb_target={triggered_rate:.1f}%"
            )
    return model.eval()


def save_model(path: Path, model: nn.Module, extra: Dict) -> None:
    torch.save({"model": model.state_dict(), "extra": extra}, path)


def load_model(path: Path, device: torch.device, n_classes: int) -> Tuple[nn.Module, nn.Module]:
    model, teacher = build_model(device, n_classes)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, teacher


def finetune_victim(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    victim_train_dataset,
    device: torch.device,
    loader_seed: int,
) -> nn.Module:
    set_seed(loader_seed)
    loader = make_loader(
        victim_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=loader_seed,
    )
    return finetune_clean_md(
        model,
        loader,
        dataset_name=DATASET_NAME,
        epochs=args.victim_ft_epochs,
        lr=args.victim_ft_lr,
        weight_decay=args.victim_ft_wd,
        use_masked=False,
        freeze_encoder=False,
        freeze_backbone_bn_stats=False,
        init_target_head=True,
        amp=args.amp,
        bn_recal_batches=args.bn_recal_batches,
    )


def maybe_eval_imagenet(
    model: nn.Module, loader: DataLoader, device: torch.device, skip: bool
) -> float:
    if skip:
        return float("nan")
    return eval_top1(model, loader, device, dataset="imagenet", route="full", show_progress=False)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.poison_fraction < 1.0):
        raise ValueError("--poison-fraction must be strictly between 0 and 1 for this experiment")
    if args.n_clean_models < 1:
        raise ValueError("--n-clean-models must be positive")

    task_tag = f"attacker-{args.attacker_domain}__victim-{args.victim_domain}__seed-{args.seed}"
    task_dir = Path(args.out_dir) / task_tag
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / "result.json"
    if args.resume and result_path.exists():
        print(f"[resume] completed result exists: {result_path}")
        return

    device = torch.device(args.device)
    set_seed(args.seed)
    print(f"[condition] attacker={args.attacker_domain} victim={args.victim_domain} seed={args.seed}")
    print(f"[device] {device}")

    dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    names = label_names(dataset)
    labels = [int(value) for value in args.labels]
    remap = {old: new for new, old in enumerate(labels)}
    if args.target_class not in remap:
        raise ValueError("--target-class must be present in --labels")
    target_class = remap[int(args.target_class)]
    n_classes = len(labels)
    groups = group_indices(dataset, labels)
    victim_source_groups, victim_ft_groups, victim_eval_groups = split_victim_domain(
        groups,
        victim_domain=args.victim_domain,
        labels=labels,
        source_frac=args.victim_source_frac,
        ft_frac=args.victim_ft_frac,
        seed=args.seed,
    )
    if args.attacker_domain == args.victim_domain:
        attacker_groups = victim_source_groups
    else:
        attacker_groups = {
            int(label): list(groups[args.attacker_domain][int(label)]) for label in labels
        }
    attacker_indices, attacker_counts = balanced_sample(
        attacker_groups, labels, args.surrogate_per_class, args.seed + 13007
    )
    victim_ft_groups = cap_groups(
        victim_ft_groups, labels, args.victim_ft_per_class, args.seed + 17011
    )
    victim_ft_indices = flatten(victim_ft_groups, labels)
    victim_eval_indices = flatten(victim_eval_groups, labels)

    overlap_attacker_victim_ft = set(attacker_indices) & set(victim_ft_indices)
    overlap_attacker_victim_eval = set(attacker_indices) & set(victim_eval_indices)
    if overlap_attacker_victim_ft or overlap_attacker_victim_eval:
        raise RuntimeError("attacker and victim indices overlap; aborting the disjoint protocol")

    split_manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "attacker_domain": args.attacker_domain,
        "victim_domain": args.victim_domain,
        "labels": labels,
        "label_names": [names[label] for label in labels],
        "target_class_old": args.target_class,
        "target_class_remapped": target_class,
        "attacker_counts_by_label": attacker_counts,
        "n_attacker": len(attacker_indices),
        "n_victim_ft": len(victim_ft_indices),
        "n_victim_eval": len(victim_eval_indices),
        "attacker_indices": attacker_indices,
        "victim_ft_indices": victim_ft_groups,
        "victim_eval_indices": victim_eval_groups,
        "attacker_victim_ft_overlap": 0,
        "attacker_victim_eval_overlap": 0,
    }
    write_json(task_dir / "split_manifest.json", split_manifest)
    write_json(task_dir / "config.json", vars(args))

    attacker_train_dataset = RemappedPacsSubset(
        dataset, attacker_indices, pacs_transform(train=True), remap
    )
    attacker_eval_dataset = RemappedPacsSubset(
        dataset, attacker_indices, pacs_transform(train=False), remap
    )
    victim_train_dataset = RemappedPacsSubset(
        dataset, victim_ft_indices, pacs_transform(train=True), remap
    )
    victim_eval_dataset = RemappedPacsSubset(
        dataset, victim_eval_indices, pacs_transform(train=False), remap
    )
    attacker_train_loader = make_loader(
        attacker_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed + 2001,
    )
    attacker_eval_loader = make_loader(
        attacker_eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, seed=args.seed + 2002,
    )
    victim_eval_loader = make_loader(
        victim_eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, seed=args.seed + 2003,
    )

    imagenet_train, imagenet_eval = build_imagenet_loaders(
        args.imagenet_root,
        bs_train=args.batch_size,
        bs_val=args.eval_batch_size,
        nw=args.num_workers,
        seed=args.seed,
        clip_normalization=False,
        clip_image_size=224,
        resize_shorter_to=256,
        rand_resize_scale=(0.6, 1.0),
    )

    delta_pix, trigger_mask, trigger_metrics = optimize_trigger(
        args,
        attacker_train_loader=attacker_train_loader,
        attacker_eval_loader=attacker_eval_loader,
        device=device,
        n_classes=n_classes,
        target_class=target_class,
        task_dir=task_dir,
    )

    poisoned_attacker_dataset = TriggeredDataset(
        attacker_train_dataset,
        target_class=target_class,
        delta_pix=delta_pix,
        mask=trigger_mask,
        poison_fraction=args.poison_fraction,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        seed=args.seed + 3001,
        relabel_to_target=True,
        exclude_target_labels=True,
    )
    attacker_poison_loader = make_loader(
        poisoned_attacker_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        seed=args.seed + 3002,
    )

    poison_checkpoint = task_dir / "post_poison.pt"
    hidden_checkpoint = task_dir / "released_hidden.pt"
    if args.resume and hidden_checkpoint.exists():
        hidden_model, teacher = load_model(hidden_checkpoint, device, n_classes)
        poison_metrics = json.loads((task_dir / "poison_metrics.json").read_text())
        hide_metrics = json.loads((task_dir / "hide_metrics.json").read_text())
        print(f"[resume] hidden checkpoint <- {hidden_checkpoint}")
    else:
        if args.resume and poison_checkpoint.exists():
            poisoned_model, teacher = load_model(poison_checkpoint, device, n_classes)
            poison_metrics = json.loads((task_dir / "poison_metrics.json").read_text())
            print(f"[resume] poison checkpoint <- {poison_checkpoint}")
        else:
            set_seed(args.seed + 4001)
            poisoned_model, teacher = build_model(device, n_classes)
            poisoned_model = train_poison_stage(
                args,
                model=poisoned_model,
                teacher=teacher,
                imagenet_train=imagenet_train,
                attacker_poison_loader=attacker_poison_loader,
                device=device,
            )
            if has_batchnorm(poisoned_model.backbone):
                bn_recalibrate_mixed(
                    poisoned_model,
                    [imagenet_train, attacker_train_loader],
                    max_batches=args.bn_recal_batches,
                    device=device,
                    datasets=["imagenet", DATASET_NAME],
                    routes=["full", "full"],
                )
            poison_metrics = {
                "attacker": evaluate_paired(
                    poisoned_model, attacker_eval_loader, device,
                    delta_pix, trigger_mask, target_class,
                ),
                "imagenet_acc": maybe_eval_imagenet(
                    poisoned_model, imagenet_eval, device, args.skip_imagenet_eval
                ),
                "n_poisoned_expected": poisoned_attacker_dataset.n_poison,
                "poison_fraction_realized": (
                    poisoned_attacker_dataset.n_poison / max(1, len(poisoned_attacker_dataset))
                ),
            }
            write_json(task_dir / "poison_metrics.json", poison_metrics)
            save_model(poison_checkpoint, poisoned_model, poison_metrics)

        hidden_model = train_hide_stage(
            args,
            model=poisoned_model,
            teacher=teacher,
            imagenet_train=imagenet_train,
            attacker_clean_loader=attacker_train_loader,
            delta_pix=delta_pix,
            mask=trigger_mask,
            target_class=target_class,
            device=device,
        )
        if has_batchnorm(hidden_model.backbone):
            bn_recalibrate_mixed(
                hidden_model,
                [imagenet_train, attacker_train_loader],
                max_batches=args.bn_recal_batches,
                device=device,
                datasets=["imagenet", DATASET_NAME],
                routes=["full", "full"],
            )
        hide_metrics = {
            "attacker": evaluate_paired(
                hidden_model, attacker_eval_loader, device,
                delta_pix, trigger_mask, target_class,
            ),
            "imagenet_acc": maybe_eval_imagenet(
                hidden_model, imagenet_eval, device, args.skip_imagenet_eval
            ),
            "suppression_margin": args.hide_margin,
        }
        write_json(task_dir / "hide_metrics.json", hide_metrics)
        save_model(hidden_checkpoint, hidden_model, hide_metrics)

    # Identical victim-finetuning seed and data order for the two arms.
    victim_ft_seed = args.seed + 50001
    pth_victim = finetune_victim(
        args,
        model=copy.deepcopy(hidden_model),
        victim_train_dataset=victim_train_dataset,
        device=device,
        loader_seed=victim_ft_seed,
    )
    pth_victim_metrics = evaluate_paired(
        pth_victim, victim_eval_loader, device, delta_pix, trigger_mask, target_class
    )
    if args.save_victim_models:
        save_model(task_dir / "pth_post_victim_ft.pt", pth_victim, pth_victim_metrics)

    set_seed(args.seed + 60001)
    clean_control, _ = build_model(device, n_classes)
    clean_control = finetune_victim(
        args,
        model=clean_control,
        victim_train_dataset=victim_train_dataset,
        device=device,
        loader_seed=victim_ft_seed,
    )
    control_metrics = evaluate_paired(
        clean_control, victim_eval_loader, device, delta_pix, trigger_mask, target_class
    )
    if args.save_victim_models:
        save_model(task_dir / "control_post_victim_ft.pt", clean_control, control_metrics)

    result = {
        "protocol_version": "pacs_semantic_shift_camera_ready_v1",
        "run_name": args.run_name,
        "task_tag": task_tag,
        "seed": args.seed,
        "attacker_domain": args.attacker_domain,
        "victim_domain": args.victim_domain,
        "label_names": [names[label] for label in labels],
        "target_class_old": args.target_class,
        "target_class_remapped": target_class,
        "n_attacker": len(attacker_indices),
        "n_victim_ft": len(victim_ft_indices),
        "n_victim_eval": len(victim_eval_indices),
        "trigger_optimization": trigger_metrics,
        "post_poison": poison_metrics,
        "released_hidden": hide_metrics,
        "post_victim_ft": {
            "poison_then_hide": pth_victim_metrics,
            "clean_encoder_trigger_only": control_metrics,
            "delta_original_asr": (
                pth_victim_metrics["original_asr_true_nontarget"]
                - control_metrics["original_asr_true_nontarget"]
            ),
            "delta_flip_asr": (
                pth_victim_metrics["flip_asr_true_nontarget"]
                - control_metrics["flip_asr_true_nontarget"]
            ),
        },
        "time": time.time(),
    }
    write_json(result_path, result)
    flat_row = {
        "protocol_version": result["protocol_version"],
        "seed": args.seed,
        "attacker_domain": args.attacker_domain,
        "victim_domain": args.victim_domain,
        "n_attacker": len(attacker_indices),
        "n_victim_ft": len(victim_ft_indices),
        "n_victim_eval": len(victim_eval_indices),
        "poison_attacker_clean": poison_metrics["attacker"]["clean_acc"],
        "poison_attacker_asr": poison_metrics["attacker"]["original_asr_true_nontarget"],
        "hidden_attacker_clean": hide_metrics["attacker"]["clean_acc"],
        "hidden_attacker_asr": hide_metrics["attacker"]["original_asr_true_nontarget"],
        "hidden_imagenet_acc": hide_metrics["imagenet_acc"],
        "pth_victim_clean": pth_victim_metrics["clean_acc"],
        "pth_victim_original_asr": pth_victim_metrics["original_asr_true_nontarget"],
        "pth_victim_flip_asr": pth_victim_metrics["flip_asr_true_nontarget"],
        "pth_victim_conditional_flip_asr": pth_victim_metrics["conditional_flip_asr"],
        "control_victim_clean": control_metrics["clean_acc"],
        "control_victim_original_asr": control_metrics["original_asr_true_nontarget"],
        "control_victim_flip_asr": control_metrics["flip_asr_true_nontarget"],
        "control_victim_conditional_flip_asr": control_metrics["conditional_flip_asr"],
        "delta_original_asr": result["post_victim_ft"]["delta_original_asr"],
        "delta_flip_asr": result["post_victim_ft"]["delta_flip_asr"],
    }
    append_single_row_csv(task_dir / "result.csv", flat_row)
    print(f"[done] {result_path}")
    print(
        f"[headline] PTH clean={pth_victim_metrics['clean_acc']:.2f} "
        f"ASR={pth_victim_metrics['original_asr_true_nontarget']:.2f} "
        f"Flip-ASR={pth_victim_metrics['flip_asr_true_nontarget']:.2f} | "
        f"control ASR={control_metrics['original_asr_true_nontarget']:.2f} "
        f"Flip-ASR={control_metrics['flip_asr_true_nontarget']:.2f}"
    )


if __name__ == "__main__":
    main()
