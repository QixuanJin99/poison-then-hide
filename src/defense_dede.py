#!/usr/bin/env python3
"""
DeDe-style sample detection for Poison-then-Hide ImageNet -> RESISC models.

DeDe ("Detecting Backdoor Samples for SSL Encoders via Decoders") is an
input/sample-level detector, not a model-level scanner. This script adapts its
core idea to our finetuned RESISC classifiers:

  1. freeze the suspect encoder,
  2. train a decoder to reconstruct images from the encoder embedding plus a
     heavily masked image-space patch signal,
  3. score clean and triggered RESISC test samples by reconstruction error,
  4. flag samples with high reconstruction error as backdoor-triggered.

The script supports both finetuned ResNet50 and finetuned CLIP ImageNet->RESISC
checkpoints through the repo's MultiDomainPartitionedModel wrapper.
"""

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from builders import base_builder_timm_clip_imnet
from datasets_imagenet import build_imagenet_val_loader
from datasets_merced import build_ucmerced_loaders
from datasets_resisc import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _default_resisc_eval_tfm,
    build_resisc45_loaders,
)
from evaluation import eval_asr, eval_top1
from run_finetune import build_asr_loader, build_model_from_ckpt
from utils import set_seed


@dataclass
class DeDeCfg:
    model_type: str
    dataset_name: str = "resisc"
    device: str = "cuda"
    seed: int = 0
    train_mask_ratio: float = 0.90
    test_mask_ratio: float = 0.99
    threshold_scale: float = 1.5
    decoder_epochs: int = 10
    decoder_steps: int = 0
    decoder_lr: float = 1e-3
    decoder_width: int = 256
    aux_samples: int = 0
    amp: bool = True


def _stats_tensors(mean_vals: Tuple[float, ...], std_vals: Tuple[float, ...], device: torch.device):
    mean_t = torch.tensor(mean_vals, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std_vals, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def denorm(x_norm: torch.Tensor, mean_vals: Tuple[float, ...], std_vals: Tuple[float, ...]) -> torch.Tensor:
    mean_t, std_t = _stats_tensors(mean_vals, std_vals, x_norm.device)
    return (x_norm * std_t + mean_t).clamp(0.0, 1.0)


def norm(x01: torch.Tensor, mean_vals: Tuple[float, ...], std_vals: Tuple[float, ...]) -> torch.Tensor:
    mean_t, std_t = _stats_tensors(mean_vals, std_vals, x01.device)
    return (x01 - mean_t) / std_t


def random_pixel_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Return a broadcastable keep-mask with 1 for visible pixels."""
    keep_prob = max(0.0, min(1.0, 1.0 - float(mask_ratio)))
    return (torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) < keep_prob).float()


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


class PatchEncoder(nn.Module):
    """Small image-space encoder for the masked local-patch signal h_e(M*x)."""

    def __init__(self, out_dim: int = 256, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x01_masked: torch.Tensor) -> torch.Tensor:
        return self.net(x01_masked)


class DeDeDecoder(nn.Module):
    """Decode global encoder embedding plus patch embedding to a 224x224 image."""

    def __init__(self, feat_dim: int, patch_dim: int = 256, width: int = 256):
        super().__init__()
        self.patch = PatchEncoder(out_dim=patch_dim)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + patch_dim, width * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            _up(width, width // 2),
            _up(width // 2, width // 4),
            _up(width // 4, width // 8),
            _up(width // 8, width // 16),
            _up(width // 16, width // 16),
            nn.Conv2d(width // 16, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        self.width = width

    def forward(self, emb: torch.Tensor, x01_masked: torch.Tensor) -> torch.Tensor:
        b = self.patch(x01_masked)
        z = torch.cat([emb.float(), b], dim=1)
        h = self.fc(z).view(z.size(0), self.width, 7, 7)
        return self.up(h)


def _up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


@torch.no_grad()
def extract_embedding(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    z = model.extract_feats(xb)
    if z.ndim == 4:
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
    return z.float()


def reconstruction_errors(
    model: nn.Module,
    decoder: DeDeDecoder,
    loader: DataLoader,
    cfg: DeDeCfg,
    mean_vals: Tuple[float, ...],
    std_vals: Tuple[float, ...],
    *,
    mask_ratio: float,
) -> Tuple[List[float], List[int]]:
    device = torch.device(cfg.device)
    model.eval()
    decoder.eval()
    errs: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"[DeDe score] mask={mask_ratio:.2f}"):
            xb = xb.to(device, non_blocking=True)
            x01 = denorm(xb, mean_vals, std_vals)
            keep = random_pixel_mask(x01, mask_ratio)
            emb = extract_embedding(model, xb)
            rec = decoder(emb, x01 * keep)
            err = (rec - x01).square().flatten(1).mean(dim=1)
            errs.extend(float(v) for v in err.detach().cpu())
            labels.extend(int(v) for v in yb)
    return errs, labels


def train_decoder(
    model: nn.Module,
    decoder: DeDeDecoder,
    loader: DataLoader,
    cfg: DeDeCfg,
    mean_vals: Tuple[float, ...],
    std_vals: Tuple[float, ...],
) -> float:
    device = torch.device(cfg.device)
    model.eval()
    decoder.train()
    model_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.decoder_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))
    losses: List[float] = []
    try:
        total_steps = 0
        for ep in range(cfg.decoder_epochs):
            pbar = tqdm(limited_batches(loader, cfg.decoder_steps), desc=f"[DeDe decoder] epoch {ep + 1}")
            for xb, _ in pbar:
                xb = xb.to(device, non_blocking=True)
                x01 = denorm(xb, mean_vals, std_vals)
                keep = random_pixel_mask(x01, cfg.train_mask_ratio)
                with torch.no_grad():
                    emb = extract_embedding(model, xb)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device.startswith("cuda")):
                    rec = decoder(emb, x01 * keep)
                    loss = F.mse_loss(rec, x01)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().item()))
                total_steps += 1
                pbar.set_postfix(loss=f"{loss.item():.5f}")
            if cfg.decoder_steps and total_steps >= cfg.decoder_steps:
                break
    finally:
        for p, flag in zip(model.parameters(), model_flags):
            p.requires_grad_(flag)
    return mean(losses) if losses else float("nan")


def auc_from_scores(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Mann-Whitney AUC for positive=triggered with larger score=more suspicious."""
    pairs = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    pairs.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def detection_metrics(clean_scores: List[float], trig_scores: List[float], threshold: float) -> Dict[str, float]:
    clean_detected = [s > threshold for s in clean_scores]
    trig_detected = [s > threshold for s in trig_scores]
    fp = sum(clean_detected)
    tn = len(clean_detected) - fp
    tp = sum(trig_detected)
    fn = len(trig_detected) - tp
    return {
        "threshold": float(threshold),
        "tpr": 100.0 * tp / max(tp + fn, 1),
        "fpr": 100.0 * fp / max(fp + tn, 1),
        "tnr": 100.0 * tn / max(fp + tn, 1),
        "fnr": 100.0 * fn / max(tp + fn, 1),
        "auc": 100.0 * auc_from_scores(trig_scores, clean_scores),
        "clean_error_mean": float(mean(clean_scores)) if clean_scores else float("nan"),
        "triggered_error_mean": float(mean(trig_scores)) if trig_scores else float("nan"),
    }


@torch.no_grad()
def filtered_asr(
    model: nn.Module,
    decoder: DeDeDecoder,
    loader_triggered: DataLoader,
    threshold: float,
    target_class: int,
    cfg: DeDeCfg,
    mean_vals: Tuple[float, ...],
    std_vals: Tuple[float, ...],
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.eval()
    decoder.eval()
    hits_raw = hits_kept = kept = detected = total = 0
    for xb, _ in tqdm(loader_triggered, desc="[DeDe filtered ASR]"):
        xb = xb.to(device, non_blocking=True)
        x01 = denorm(xb, mean_vals, std_vals)
        keep_mask = random_pixel_mask(x01, cfg.test_mask_ratio)
        emb = extract_embedding(model, xb)
        rec = decoder(emb, x01 * keep_mask)
        err = (rec - x01).square().flatten(1).mean(dim=1)
        suspicious = err > threshold
        logits = model(xb, dataset=cfg.dataset_name, route="full")
        pred = logits.argmax(dim=1)
        hit = pred == int(target_class)
        total += int(pred.numel())
        detected += int(suspicious.sum().item())
        kept_mask = ~suspicious
        kept += int(kept_mask.sum().item())
        hits_raw += int(hit.sum().item())
        hits_kept += int((hit & kept_mask).sum().item())
    return {
        "raw_asr": 100.0 * hits_raw / max(total, 1),
        "filtered_asr_count_detected_as_blocked": 100.0 * hits_kept / max(total, 1),
        "asr_among_kept": 100.0 * hits_kept / max(kept, 1),
        "detected_rate": 100.0 * detected / max(total, 1),
        "kept_rate": 100.0 * kept / max(total, 1),
    }


@torch.no_grad()
def evaluate_clean_asr(
    model: nn.Module,
    clean_loader: DataLoader,
    trigger_loader: DataLoader,
    target_class: int,
    device: torch.device,
) -> Dict[str, float]:
    return {
        "resisc_acc": eval_top1(model, clean_loader, device, dataset="resisc", route="full", show_progress=False),
        "resisc_asr": eval_asr(
            model,
            trigger_loader,
            device,
            dataset="resisc",
            target_class=target_class,
            route="full",
            show_progress=False,
        ),
    }


def save_checkpoint(path: Path, decoder: DeDeDecoder, cfg: DeDeCfg, extra: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decoder": decoder.state_dict(), "cfg": asdict(cfg), "extra": extra}, path)


def parse_args():
    p = argparse.ArgumentParser(description="DeDe sample detector for finetuned RESISC checkpoints.")
    p.add_argument("--ckpt", required=True, help="Finetuned checkpoint to evaluate.")
    p.add_argument("--trigger", required=True, help="True PtH trigger .pt for triggered-sample evaluation.")
    p.add_argument("--model-type", required=True, choices=["resnet50", "clip"])
    p.add_argument("--target-class", type=int, default=28)
    p.add_argument("--imagenet-root", default="", help="Optional ImageNet root for ImageNet accuracy eval.")
    p.add_argument("--out-dir", default="./dede_out")
    p.add_argument("--run-name", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--nw", type=int, default=4)

    p.add_argument("--aux-data", choices=["resisc", "ucmerced"], default="resisc")
    p.add_argument("--aux-samples", type=int, default=0, help="Cap auxiliary decoder-training samples; 0 means all.")
    p.add_argument("--train-mask-ratio", type=float, default=0.90)
    p.add_argument("--test-mask-ratio", type=float, default=0.99)
    p.add_argument("--threshold-scale", type=float, default=1.5)
    p.add_argument("--decoder-epochs", type=int, default=10)
    p.add_argument("--decoder-steps", type=int, default=0)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--decoder-width", type=int, default=256)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "", name.strip())


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_name = _safe_name(args.run_name) or f"dede_{args.model_type}_seed{args.seed}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = DeDeCfg(
        model_type=args.model_type,
        device=str(device),
        seed=args.seed,
        train_mask_ratio=args.train_mask_ratio,
        test_mask_ratio=args.test_mask_ratio,
        threshold_scale=args.threshold_scale,
        decoder_epochs=args.decoder_epochs,
        decoder_steps=args.decoder_steps,
        decoder_lr=args.decoder_lr,
        decoder_width=args.decoder_width,
        aux_samples=args.aux_samples,
        amp=not args.no_amp,
    )

    tf_train = tf_val = None
    mean_vals, std_vals = IMAGENET_MEAN, IMAGENET_STD
    if args.model_type == "clip":
        _, _, tf_train, tf_val, clip_cfg = base_builder_timm_clip_imnet(device)
        mean_vals = tuple(float(x) for x in clip_cfg["mean"])
        std_vals = tuple(float(x) for x in clip_cfg["std"])

    resisc_train, resisc_test, n_resisc, class_names = build_resisc45_loaders(
        bs_train=args.batch_size,
        bs_test=args.eval_batch_size,
        nw=args.nw,
        seed=args.seed,
        transform_train=tf_train,
        transform_val=tf_val,
        clip_normalization=(args.model_type == "clip"),
    )
    if args.aux_data == "resisc":
        aux_loader = subset_loader(resisc_train, args.aux_samples, args.seed, shuffle=True)
        aux_name = "RESISC train"
    else:
        aux_loader, _, _, _ = build_ucmerced_loaders(
            bs_train=args.batch_size,
            bs_eval=args.eval_batch_size,
            nw=args.nw,
            seed=args.seed,
            test_size=0.0,
            clip_normalization=(args.model_type == "clip"),
        )
        aux_loader = subset_loader(aux_loader, args.aux_samples, args.seed, shuffle=True)
        aux_name = "UC Merced"

    trigger_loader, trigger_target = build_asr_loader(
        resisc_test,
        args.trigger,
        device,
        tf_val if tf_val is not None else _default_resisc_eval_tfm(),
        dataset_name="resisc",
        mean=mean_vals,
        std=std_vals,
    )
    if trigger_target != args.target_class:
        print(f"[warn] trigger target_class={trigger_target}; evaluating --target-class={args.target_class}")

    model = build_model_from_ckpt(
        args.ckpt,
        n_tgt=n_resisc,
        device=device,
        model_type=args.model_type,
        dataset_name="resisc",
    ).to(device)
    model.configure_training_scope("resisc", mode="generic", allowed_parts=None, install_grad_hook=False)
    model.eval()

    feat_dim = int(getattr(model, "_D", None) or model.heads["resisc"].weight.shape[1])
    decoder = DeDeDecoder(feat_dim=feat_dim, width=args.decoder_width).to(device)

    t0 = time.time()
    pre = evaluate_clean_asr(model, resisc_test, trigger_loader, args.target_class, device)
    print(f"[pre] RESISC acc={pre['resisc_acc']:.2f}% ASR={pre['resisc_asr']:.2f}%")

    train_loss = train_decoder(model, decoder, aux_loader, cfg, mean_vals, std_vals)
    threshold = cfg.threshold_scale * train_loss
    print(f"[DeDe] train_loss={train_loss:.6f} threshold={threshold:.6f} aux={aux_name}")

    clean_scores, clean_labels = reconstruction_errors(
        model, decoder, resisc_test, cfg, mean_vals, std_vals, mask_ratio=cfg.test_mask_ratio
    )
    trig_scores, trig_labels = reconstruction_errors(
        model, decoder, trigger_loader, cfg, mean_vals, std_vals, mask_ratio=cfg.test_mask_ratio
    )
    det = detection_metrics(clean_scores, trig_scores, threshold)
    filt = filtered_asr(model, decoder, trigger_loader, threshold, args.target_class, cfg, mean_vals, std_vals)
    print(
        "[DeDe] "
        f"TPR={det['tpr']:.2f}% FPR={det['fpr']:.2f}% AUC={det['auc']:.2f}% "
        f"raw_ASR={filt['raw_asr']:.2f}% filtered_ASR={filt['filtered_asr_count_detected_as_blocked']:.2f}%"
    )

    imagenet_acc = None
    if args.imagenet_root:
        im_val = build_imagenet_val_loader(
            args.imagenet_root,
            bs=args.eval_batch_size,
            nw=args.nw,
            transform_val=tf_val,
            clip_normalization=(args.model_type == "clip"),
        )
        imagenet_acc = eval_top1(model, im_val, device, dataset="imagenet", route="full", show_progress=False)

    save_checkpoint(
        run_dir / "dede_decoder.pt",
        decoder,
        cfg,
        {
            "source_ckpt": args.ckpt,
            "trigger": args.trigger,
            "feat_dim": feat_dim,
            "mean": mean_vals,
            "std": std_vals,
        },
    )

    metrics = {
        "pre": {**pre, "imagenet_acc": imagenet_acc},
        "dede": {
            "train_loss": train_loss,
            "threshold": threshold,
            "detection": det,
            "filtered_asr": filt,
            "clean_scores": clean_scores,
            "triggered_scores": trig_scores,
        },
        "duration_sec": time.time() - t0,
        "args": vars(args),
        "cfg": asdict(cfg),
        "class_names": class_names,
        "aux_data": aux_name,
        "adaptation_note": "DeDe is sample-level detection; triggered samples with reconstruction error above threshold are treated as blocked.",
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
