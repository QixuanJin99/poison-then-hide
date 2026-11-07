#!/usr/bin/env python3
"""
Adapted CLIP-Inspector for Poison-then-Hide CLIP -> RESISC checkpoints.

The original CLIP-Inspector paper targets prompt-tuned CLIP/CoCoOp models.
Our CLIP experiments use a timm CLIP visual backbone with a RESISC linear head,
so this script ports the model-level core:

  * use unlabeled OOD images (UC Merced by default),
  * invert one dense L_inf-bounded pixel perturbation per candidate RESISC class,
  * score classes by high induced ASR and low margin loss,
  * flag a model if the top class is an anomaly,
  * optionally repair with the top reconstructed trigger on labeled clean RESISC.

This is intentionally written for the finetuned CLIP ImageNet -> RESISC setting.
"""

import argparse
import copy
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from builders import base_builder_timm_clip_imnet
from datasets_imagenet import build_imagenet_val_loader
from datasets_merced import build_ucmerced_loaders
from datasets_resisc import build_resisc45_loaders
from evaluation import eval_asr, eval_top1
from run_finetune import build_asr_loader, build_model_from_ckpt
from utils import set_seed


@dataclass
class CICfg:
    dataset_name: str = "resisc"
    device: str = "cuda"
    seed: int = 0
    eps: float = 4.0 / 255.0
    lr: float = 0.1
    inversion_epochs: int = 1
    max_ood_batches: int = 0
    anomaly_threshold: float = 2.0
    save_topk_triggers: int = 5
    amp: bool = True

    repair_epochs: int = 1
    repair_lr: float = 5e-6
    repair_weight_decay: float = 0.05
    repair_freeze_encoder: bool = False
    repair_lambda_clean: float = 1.0
    repair_lambda_triggered: float = 1.0
    repair_max_batches: int = 0


def _mean_std_tensors(mean: Tuple[float, ...], std: Tuple[float, ...], device: torch.device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def denorm(x_norm: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    mean_t, std_t = _mean_std_tensors(mean, std, x_norm.device)
    return (x_norm * std_t + mean_t).clamp(0.0, 1.0)


def norm(x01: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    mean_t, std_t = _mean_std_tensors(mean, std, x01.device)
    return (x01 - mean_t) / std_t


def apply_delta_normed(
    x_norm: torch.Tensor,
    delta01: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> torch.Tensor:
    x01 = denorm(x_norm, mean, std)
    return norm((x01 + delta01).clamp(0.0, 1.0), mean, std)


def margin_loss_for_class(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    target = logits[:, target_class]
    other = logits.clone()
    other[:, target_class] = -torch.inf
    max_other = other.max(dim=1).values
    return -(target - max_other).mean()


def zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)


def limited_batches(loader: Iterable, max_batches: int):
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        yield batch


def subset_loader(loader: DataLoader, max_samples: int, seed: int, shuffle: bool) -> DataLoader:
    if not max_samples or max_samples >= len(loader.dataset):
        return loader
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(loader.dataset), generator=gen)[:max_samples].tolist()
    return DataLoader(
        Subset(loader.dataset, idx),
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", True),
        collate_fn=loader.collate_fn,
        drop_last=getattr(loader, "drop_last", False),
    )


def parse_classes(spec: str, n_classes: int) -> List[int]:
    if not spec or spec.lower() == "all":
        return list(range(n_classes))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    bad = [c for c in out if c < 0 or c >= n_classes]
    if bad:
        raise ValueError(f"Candidate classes out of range [0,{n_classes}): {bad}")
    return sorted(set(out))


def optimize_delta_for_class(
    model: nn.Module,
    loader: DataLoader,
    target_class: int,
    cfg: CICfg,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> Tuple[torch.Tensor, float]:
    device = torch.device(cfg.device)
    model.eval()
    delta = nn.Parameter(torch.empty(1, 3, 224, 224, device=device).uniform_(-cfg.eps, cfg.eps))
    opt = torch.optim.Adam([delta], lr=cfg.lr)

    last_loss = math.nan
    model_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for _ in range(cfg.inversion_epochs):
            for xb, _ in limited_batches(loader, cfg.max_ood_batches):
                xb = xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device.startswith("cuda")):
                    logits = model(apply_delta_normed(xb, delta, mean, std), dataset=cfg.dataset_name, route="full")
                    loss = margin_loss_for_class(logits, target_class)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    delta.clamp_(-cfg.eps, cfg.eps)
                last_loss = float(loss.detach().item())
    finally:
        for p, flag in zip(model.parameters(), model_flags):
            p.requires_grad_(flag)

    return delta.detach(), last_loss


@torch.no_grad()
def score_delta(
    model: nn.Module,
    loader: DataLoader,
    delta: torch.Tensor,
    target_class: int,
    cfg: CICfg,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.eval()
    hits, total = 0, 0
    losses = []
    for xb, _ in limited_batches(loader, cfg.max_ood_batches):
        xb = xb.to(device, non_blocking=True)
        logits = model(apply_delta_normed(xb, delta, mean, std), dataset=cfg.dataset_name, route="full")
        pred = logits.argmax(dim=1)
        hits += int((pred == target_class).sum().item())
        total += int(pred.numel())
        losses.append(float(margin_loss_for_class(logits, target_class).item()))
    return {
        "asr": 100.0 * hits / max(total, 1),
        "margin_loss": sum(losses) / max(len(losses), 1),
    }


def run_clip_inspector(
    model: nn.Module,
    invert_loader: DataLoader,
    score_loader: DataLoader,
    candidate_classes: List[int],
    cfg: CICfg,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> Tuple[Dict, Dict[int, torch.Tensor]]:
    records = []
    deltas: Dict[int, torch.Tensor] = {}
    pbar = tqdm(candidate_classes, desc="[CLIP-Inspector] class inversion")
    for cls in pbar:
        delta, train_loss = optimize_delta_for_class(model, invert_loader, cls, cfg, mean, std)
        metrics = score_delta(model, score_loader, delta, cls, cfg, mean, std)
        rec = {
            "class_id": int(cls),
            "train_last_loss": float(train_loss),
            "asr": float(metrics["asr"]),
            "margin_loss": float(metrics["margin_loss"]),
            "delta_linf": float(delta.abs().max().item()),
            "delta_l2": float(delta.square().mean().sqrt().item()),
        }
        records.append(rec)
        deltas[int(cls)] = delta.cpu()
        pbar.set_postfix(cls=cls, asr=f"{rec['asr']:.1f}", loss=f"{rec['margin_loss']:.3f}")

    asr = torch.tensor([r["asr"] for r in records], dtype=torch.float32)
    loss = torch.tensor([r["margin_loss"] for r in records], dtype=torch.float32)
    score = zscore(asr) - zscore(loss)
    score_z = zscore(score)
    for r, s, z in zip(records, score.tolist(), score_z.tolist()):
        r["anomaly_score"] = float(s)
        r["anomaly_z"] = float(z)

    records = sorted(records, key=lambda r: r["anomaly_z"], reverse=True)
    top = records[0]
    summary = {
        "is_backdoored": bool(top["anomaly_z"] >= cfg.anomaly_threshold),
        "threshold": cfg.anomaly_threshold,
        "top_class": int(top["class_id"]),
        "top_anomaly_z": float(top["anomaly_z"]),
        "top_anomaly_score": float(top["anomaly_score"]),
        "top_asr": float(top["asr"]),
        "top_margin_loss": float(top["margin_loss"]),
        "records": records,
    }
    return summary, deltas


def repair_with_delta(
    model: nn.Module,
    loader: DataLoader,
    delta_cpu: torch.Tensor,
    cfg: CICfg,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> nn.Module:
    device = torch.device(cfg.device)
    model.train()
    delta = delta_cpu.to(device)

    for p in model.parameters():
        p.requires_grad_(not cfg.repair_freeze_encoder)
    if cfg.repair_freeze_encoder:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    for hname, head in model.heads.items():
        for p in head.parameters():
            p.requires_grad_(hname == cfg.dataset_name)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.repair_lr,
        weight_decay=cfg.repair_weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))
    for epoch in range(cfg.repair_epochs):
        pbar = tqdm(limited_batches(loader, cfg.repair_max_batches), desc=f"[CLIP-Inspector repair] epoch {epoch + 1}")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device.startswith("cuda")):
                clean_logits = model(xb, dataset=cfg.dataset_name, route="full")
                trig_logits = model(apply_delta_normed(xb, delta, mean, std), dataset=cfg.dataset_name, route="full")
                loss_clean = F.cross_entropy(clean_logits, yb)
                loss_trig = F.cross_entropy(trig_logits, yb)
                loss = cfg.repair_lambda_clean * loss_clean + cfg.repair_lambda_triggered * loss_trig
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}", clean=f"{loss_clean.item():.4f}", trig=f"{loss_trig.item():.4f}")
    model.configure_training_scope(cfg.dataset_name, mode="generic", allowed_parts=None, install_grad_hook=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    resisc_test: DataLoader,
    device: torch.device,
    *,
    trigger_loader: Optional[DataLoader] = None,
    target_class: Optional[int] = None,
    imagenet_val: Optional[DataLoader] = None,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "resisc_acc": eval_top1(model, resisc_test, device, dataset="resisc", route="full", show_progress=False),
        "resisc_asr": None,
        "imagenet_acc": None,
    }
    if trigger_loader is not None and target_class is not None:
        out["resisc_asr"] = eval_asr(
            model,
            trigger_loader,
            device,
            dataset="resisc",
            target_class=target_class,
            route="full",
            show_progress=False,
        )
    if imagenet_val is not None:
        out["imagenet_acc"] = eval_top1(model, imagenet_val, device, dataset="imagenet", route="full", show_progress=False)
    return out


def save_checkpoint(path: Path, model: nn.Module, extra: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path)


def parse_args():
    p = argparse.ArgumentParser(description="Adapted CLIP-Inspector for finetuned CLIP ImageNet -> RESISC checkpoints.")
    p.add_argument("--ckpt", required=True, help="Finetuned CLIP checkpoint.")
    p.add_argument("--trigger", default="", help="Optional true PtH trigger .pt for ASR eval.")
    p.add_argument("--target-class", type=int, default=28, help="True target class for optional ASR eval.")
    p.add_argument("--imagenet-root", default="", help="Optional ImageNet root for ImageNet accuracy eval.")
    p.add_argument("--out-dir", default="./clip_inspector_out")
    p.add_argument("--run-name", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--nw", type=int, default=4)

    p.add_argument("--candidate-classes", default="all", help="all, comma list, or ranges like 0-44.")
    p.add_argument("--eps", type=float, default=4.0 / 255.0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--inversion-epochs", type=int, default=1)
    p.add_argument("--ood-samples", type=int, default=1000)
    p.add_argument("--ood-test-size", type=float, default=0.5)
    p.add_argument("--max-ood-batches", type=int, default=0)
    p.add_argument("--anomaly-threshold", type=float, default=2.0)
    p.add_argument("--save-topk-triggers", type=int, default=5)
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--repair", action="store_true", help="Run CI-style repair using the top reconstructed trigger.")
    p.add_argument("--repair-epochs", type=int, default=1)
    p.add_argument("--repair-lr", type=float, default=5e-6)
    p.add_argument("--repair-weight-decay", type=float, default=0.05)
    p.add_argument("--repair-freeze-encoder", action="store_true")
    p.add_argument("--repair-max-batches", type=int, default=0)
    return p.parse_args()


def _safe_run_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "", name.strip())


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_name = _safe_run_name(args.run_name) or f"clip_inspector_seed{args.seed}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = CICfg(
        device=str(device),
        seed=args.seed,
        eps=args.eps,
        lr=args.lr,
        inversion_epochs=args.inversion_epochs,
        max_ood_batches=args.max_ood_batches,
        anomaly_threshold=args.anomaly_threshold,
        save_topk_triggers=args.save_topk_triggers,
        amp=not args.no_amp,
        repair_epochs=args.repair_epochs,
        repair_lr=args.repair_lr,
        repair_weight_decay=args.repair_weight_decay,
        repair_freeze_encoder=args.repair_freeze_encoder,
        repair_max_batches=args.repair_max_batches,
    )

    _, _, tf_train, tf_val, clip_cfg = base_builder_timm_clip_imnet(device)
    mean = tuple(float(x) for x in clip_cfg["mean"])
    std = tuple(float(x) for x in clip_cfg["std"])

    resisc_train, resisc_test, n_resisc, class_names = build_resisc45_loaders(
        bs_train=args.batch_size,
        bs_test=args.eval_batch_size,
        nw=args.nw,
        seed=args.seed,
        transform_train=tf_train,
        transform_val=tf_val,
        clip_normalization=True,
    )
    ood_train, ood_score, _, ood_names = build_ucmerced_loaders(
        bs_train=args.batch_size,
        bs_eval=args.batch_size,
        nw=args.nw,
        seed=args.seed,
        test_size=args.ood_test_size,
        clip_normalization=True,
    )
    ood_train = subset_loader(ood_train, args.ood_samples, args.seed, shuffle=True)
    if ood_score is None:
        ood_score = ood_train
    else:
        ood_score = subset_loader(ood_score, args.ood_samples, args.seed + 1, shuffle=False)

    imagenet_val = None
    if args.imagenet_root:
        imagenet_val = build_imagenet_val_loader(args.imagenet_root, bs=args.eval_batch_size, nw=args.nw, transform_val=tf_val)

    trigger_loader = None
    if args.trigger:
        trigger_loader, trigger_target = build_asr_loader(
            resisc_test,
            args.trigger,
            device,
            tf_val,
            dataset_name="resisc",
            mean=mean,
            std=std,
        )
        if trigger_target != args.target_class:
            print(f"[warn] trigger target_class={trigger_target}; using --target-class={args.target_class} for ASR eval")

    model = build_model_from_ckpt(
        args.ckpt,
        n_tgt=n_resisc,
        device=device,
        model_type="clip",
        dataset_name="resisc",
    ).to(device)
    model.configure_training_scope("resisc", mode="generic", allowed_parts=None, install_grad_hook=False)
    model.eval()

    candidate_classes = parse_classes(args.candidate_classes, n_resisc)
    t0 = time.time()
    pre = evaluate_model(
        model,
        resisc_test,
        device,
        trigger_loader=trigger_loader,
        target_class=args.target_class,
        imagenet_val=imagenet_val,
    )
    print(f"[pre] RESISC acc={pre['resisc_acc']:.2f}% ASR={pre['resisc_asr']}")

    ci_summary, deltas = run_clip_inspector(model, ood_train, ood_score, candidate_classes, cfg, mean, std)
    print(
        "[clip-inspector] "
        f"flagged={ci_summary['is_backdoored']} "
        f"top_class={ci_summary['top_class']} "
        f"z={ci_summary['top_anomaly_z']:.3f} "
        f"asr={ci_summary['top_asr']:.2f}% "
        f"loss={ci_summary['top_margin_loss']:.4f}"
    )

    top_records = ci_summary["records"][: max(0, cfg.save_topk_triggers)]
    trigger_artifacts = {
        int(r["class_id"]): deltas[int(r["class_id"])]
        for r in top_records
    }
    torch.save(trigger_artifacts, run_dir / "ci_top_triggers.pt")

    post_repair = None
    if args.repair:
        top_delta = deltas[int(ci_summary["top_class"])]
        model = repair_with_delta(model, resisc_train, top_delta, cfg, mean, std)
        post_repair = evaluate_model(
            model,
            resisc_test,
            device,
            trigger_loader=trigger_loader,
            target_class=args.target_class,
            imagenet_val=imagenet_val,
        )
        print(f"[post-repair] RESISC acc={post_repair['resisc_acc']:.2f}% ASR={post_repair['resisc_asr']}")
        save_checkpoint(
            run_dir / "clip_inspector_repaired.pt",
            model,
            {
                "source_ckpt": args.ckpt,
                "trigger": args.trigger,
                "top_class": ci_summary["top_class"],
                "class_names": class_names,
                "ood_names": ood_names,
                "ci_cfg": asdict(cfg),
            },
        )

    metrics = {
        "pre": pre,
        "clip_inspector": ci_summary,
        "post_repair": post_repair,
        "duration_sec": time.time() - t0,
        "args": vars(args),
        "cfg": asdict(cfg),
        "class_names": class_names,
        "ood_dataset": "UC_Merced",
        "ood_class_names": ood_names,
        "adaptation_note": "Adapted from prompt-tuned CLIP-Inspector to this repo's CLIP visual backbone + RESISC linear head.",
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
