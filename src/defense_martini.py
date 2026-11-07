#!/usr/bin/env python3
"""
MARTINI-style backdoor mitigation for Poison-then-Hide checkpoints.

This script adapts "Mitigating Backdoor Attacks via Trigger Reconstruction
and Model Hardening" to the ResNet50 ImageNet -> RESISC setting used in this
repo. It can be run on either Hidden or Finetuned checkpoints:

  1. train a decoder that maps ResNet50 layer4 feature maps back to images,
  2. reconstruct backdoor-like samples by optimizing a feature-space regional
     transformation to drive the victim model toward the target class,
  3. harden the suspect model on clean and reconstructed samples with correct
     labels,
  4. report clean RESISC accuracy and ASR before/after mitigation, with an
     optional benign finetuning pass to test backdoor resurfacing.

The paper's regional transformation is implemented as region-wise affine
feature-map mutation over layer4 activations. This is the practical analogue
for ResNet50, whose classifier path otherwise exposes only pooled features.
"""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from benign_finetune import finetune_clean_md
from datasets_imagenet import build_imagenet_val_loader
from datasets_resisc import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _default_resisc_eval_tfm,
    build_resisc45_loaders,
)
from evaluation import eval_asr, eval_top1
from run_finetune import build_asr_loader, build_model_from_ckpt
from utils import set_seed


@dataclass
class MartiniCfg:
    dataset_name: str = "resisc"
    target_class: int = 28
    device: str = "cuda"
    seed: int = 0

    decoder_epochs: int = 4
    decoder_steps: int = 0
    decoder_lr: float = 1e-3
    decoder_width: int = 256

    harden_epochs: int = 2
    harden_steps: int = 0
    harden_lr: float = 3e-4
    harden_weight_decay: float = 1e-4
    defense_fraction: float = 0.05

    recon_inner_steps: int = 8
    recon_lr: float = 5e-2
    region_grid: int = 3
    feature_gain_scale: float = 0.35
    feature_bias_scale: float = 0.35

    lambda_clean: float = 1.0
    lambda_harden: float = 1.0
    lambda_content: float = 0.05
    lambda_ssim: float = 0.20
    lambda_smooth: float = 0.01
    lambda_norm: float = 0.01

    amp: bool = True
    max_eval_batches: int = 0


def _norm_tensors(device: torch.device):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def denorm(x_norm: torch.Tensor) -> torch.Tensor:
    mean, std = _norm_tensors(x_norm.device)
    return (x_norm * std + mean).clamp(0.0, 1.0)


def norm(x01: torch.Tensor) -> torch.Tensor:
    mean, std = _norm_tensors(x01.device)
    return (x01 - mean) / std


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Small differentiable SSIM loss, averaged over image/channel/spatial dims."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sig_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sig_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sig_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x.square() + mu_y.square() + c1) * (sig_x + sig_y + c2)
    return (1.0 - (num / den.clamp_min(1e-6))).mean()


class ResNetLayer4Features(nn.Module):
    """Expose ResNet50 layer4 spatial maps, before avgpool/fc."""

    def __init__(self, resnet_backbone: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(
            resnet_backbone.conv1,
            resnet_backbone.bn1,
            resnet_backbone.relu,
            resnet_backbone.maxpool,
        )
        self.layer1 = resnet_backbone.layer1
        self.layer2 = resnet_backbone.layer2
        self.layer3 = resnet_backbone.layer3
        self.layer4 = resnet_backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class FeatureDecoder(nn.Module):
    """Decode [B, 2048, 7, 7] feature maps to [B, 3, 224, 224] in pixel space."""

    def __init__(self, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2048, width, kernel_size=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            _up(width, width // 2),
            _up(width // 2, width // 4),
            _up(width // 4, width // 8),
            _up(width // 8, width // 16),
            _up(width // 16, width // 16),
            nn.Conv2d(width // 16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def _up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class PixelNormLayer(nn.Module):
    """MARTINI normalization layer Gamma with learnable target channel stats."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.mu_raw = nn.Parameter(torch.zeros(1, 3, 1, 1, device=device))
        self.std_raw = nn.Parameter(torch.zeros(1, 3, 1, 1, device=device))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        mu_x = x01.mean(dim=(2, 3), keepdim=True)
        std_x = x01.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        mu_b = torch.sigmoid(self.mu_raw)
        std_b = F.softplus(self.std_raw) + 0.05
        return ((x01 - mu_x) / std_x * std_b + mu_b).clamp(0.0, 1.0)

    def norm_penalty(self, x01: torch.Tensor) -> torch.Tensor:
        mu_x = x01.mean(dim=(0, 2, 3), keepdim=True)
        std_x = x01.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
        mu_b = torch.sigmoid(self.mu_raw)
        std_b = F.softplus(self.std_raw) + 0.05
        return (mu_b - mu_x).abs().mean() + (std_b - std_x).abs().mean()


class RegionalFeatureTransform(nn.Module):
    """Region-wise affine mutation over the encoder feature map."""

    def __init__(self, channels: int, grid: int, gain_scale: float, bias_scale: float, device: torch.device):
        super().__init__()
        self.grid = int(grid)
        self.gain_scale = float(gain_scale)
        self.bias_scale = float(bias_scale)
        self.gain = nn.Parameter(torch.zeros(1, channels, self.grid, self.grid, device=device))
        self.bias = nn.Parameter(torch.zeros(1, channels, self.grid, self.grid, device=device))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        gain = torch.tanh(self.gain) * self.gain_scale
        bias = torch.tanh(self.bias) * self.bias_scale
        gain = F.interpolate(gain, size=(h, w), mode="nearest")
        bias = F.interpolate(bias, size=(h, w), mode="nearest")
        return feat * (1.0 + gain) + bias


def subset_loader(loader: DataLoader, fraction: float, seed: int, shuffle: bool = True) -> DataLoader:
    if fraction >= 0.999:
        return loader
    n = len(loader.dataset)
    keep = max(1, int(round(n * fraction)))
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:keep].tolist()
    return DataLoader(
        Subset(loader.dataset, idx),
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", True),
        collate_fn=loader.collate_fn,
        drop_last=getattr(loader, "drop_last", False),
    )


def limited_batches(loader: DataLoader, max_batches: int):
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        yield batch


def train_decoder(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    cfg: MartiniCfg,
) -> None:
    encoder.eval()
    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.decoder_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))

    total_steps = 0
    for epoch in range(cfg.decoder_epochs):
        pbar = tqdm(limited_batches(loader, cfg.decoder_steps), desc=f"[MARTINI decoder] epoch {epoch + 1}")
        for xb, _ in pbar:
            xb = xb.to(cfg.device, non_blocking=True)
            x01 = denorm(xb)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                feat = encoder(xb)
            with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device.startswith("cuda")):
                rec = decoder(feat)
                loss = F.mse_loss(rec, x01) + 0.25 * ssim_loss(rec, x01)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if cfg.decoder_steps and total_steps >= cfg.decoder_steps:
            break


def reconstruct_backdoor_samples(
    victim: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    xb: torch.Tensor,
    target_class: int,
    cfg: MartiniCfg,
) -> torch.Tensor:
    victim.eval()
    encoder.eval()
    decoder.eval()
    x01 = denorm(xb)

    gamma = PixelNormLayer(torch.device(cfg.device)).to(cfg.device)
    psi = RegionalFeatureTransform(
        channels=2048,
        grid=cfg.region_grid,
        gain_scale=cfg.feature_gain_scale,
        bias_scale=cfg.feature_bias_scale,
        device=torch.device(cfg.device),
    ).to(cfg.device)
    opt = torch.optim.AdamW(list(gamma.parameters()) + list(psi.parameters()), lr=cfg.recon_lr)
    yt = torch.full((xb.size(0),), int(target_class), dtype=torch.long, device=xb.device)

    with torch.no_grad():
        feat_clean = encoder(xb)

    x_hat = x01
    victim_flags = [p.requires_grad for p in victim.parameters()]
    decoder_flags = [p.requires_grad for p in decoder.parameters()]
    for p in victim.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    try:
        for _ in range(cfg.recon_inner_steps):
            opt.zero_grad(set_to_none=True)
            x_normed01 = gamma(x01)
            feat = encoder(norm(x_normed01))
            x_hat = decoder(psi(feat)).clamp(0.0, 1.0)
            logits = victim(norm(x_hat), dataset=cfg.dataset_name, route="full")
            feat_hat = encoder(norm(x_hat))
            ce = F.cross_entropy(logits, yt)
            content = F.mse_loss(F.normalize(feat_hat.flatten(1), dim=1), F.normalize(feat_clean.flatten(1), dim=1))
            loss = (
                ce
                + cfg.lambda_content * content
                + cfg.lambda_ssim * ssim_loss(x_hat, x01)
                + cfg.lambda_smooth * smoothness_loss(x_hat)
                + cfg.lambda_norm * gamma.norm_penalty(x01)
            )
            loss.backward()
            opt.step()
    finally:
        for p, flag in zip(victim.parameters(), victim_flags):
            p.requires_grad_(flag)
        for p, flag in zip(decoder.parameters(), decoder_flags):
            p.requires_grad_(flag)

    return x_hat.detach()


def harden_model(
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    cfg: MartiniCfg,
) -> nn.Module:
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    for hname, head in model.heads.items():
        if hname != cfg.dataset_name:
            for p in head.parameters():
                p.requires_grad_(False)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.harden_lr,
        weight_decay=cfg.harden_weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))
    total_steps = 0

    for epoch in range(cfg.harden_epochs):
        pbar = tqdm(limited_batches(loader, cfg.harden_steps), desc=f"[MARTINI harden] epoch {epoch + 1}")
        for xb, yb in pbar:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            x_backdoor01 = reconstruct_backdoor_samples(model, encoder, decoder, xb, cfg.target_class, cfg)

            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device.startswith("cuda")):
                clean_logits = model(xb, dataset=cfg.dataset_name, route="full")
                hard_logits = model(norm(x_backdoor01), dataset=cfg.dataset_name, route="full")
                loss_clean = F.cross_entropy(clean_logits, yb)
                loss_hard = F.cross_entropy(hard_logits, yb)
                loss = cfg.lambda_clean * loss_clean + cfg.lambda_harden * loss_hard
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", clean=f"{loss_clean.item():.4f}", hard=f"{loss_hard.item():.4f}")
        if cfg.harden_steps and total_steps >= cfg.harden_steps:
            break

    model.configure_training_scope(cfg.dataset_name, mode="generic", allowed_parts=None, install_grad_hook=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    resisc_test: DataLoader,
    asr_loader: DataLoader,
    target_class: int,
    device: torch.device,
    *,
    imagenet_val: Optional[DataLoader] = None,
    max_eval_batches: int = 0,
) -> Dict[str, Optional[float]]:
    if max_eval_batches:
        resisc_test = list(limited_batches(resisc_test, max_eval_batches))
        asr_loader = list(limited_batches(asr_loader, max_eval_batches))
        if imagenet_val is not None:
            imagenet_val = list(limited_batches(imagenet_val, max_eval_batches))
    out: Dict[str, Optional[float]] = {}
    out["resisc_acc"] = eval_top1(model, resisc_test, device, dataset="resisc", route="full", show_progress=False)
    out["resisc_asr"] = eval_asr(
        model,
        asr_loader,
        device,
        dataset="resisc",
        target_class=target_class,
        route="full",
        show_progress=False,
    )
    out["imagenet_acc"] = None
    if imagenet_val is not None:
        out["imagenet_acc"] = eval_top1(model, imagenet_val, device, dataset="imagenet", route="full", show_progress=False)
    return out


def save_checkpoint(path: Path, model: nn.Module, extra: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path)


def parse_args():
    p = argparse.ArgumentParser(description="MARTINI defense for ResNet50 ImageNet -> RESISC PtH checkpoints.")
    p.add_argument("--ckpt", required=True, help="Hidden or Finetuned checkpoint to mitigate.")
    p.add_argument("--trigger", required=True, help="PtH trigger .pt with delta_pix/mask/target_class.")
    p.add_argument("--imagenet-root", default="", help="Optional ImageNet root for ImageNet accuracy eval.")
    p.add_argument("--out-dir", default="./martini_out")
    p.add_argument("--run-name", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--nw", type=int, default=4)
    p.add_argument("--target-class", type=int, default=28)

    p.add_argument("--defense-fraction", type=float, default=0.05)
    p.add_argument("--decoder-epochs", type=int, default=4)
    p.add_argument("--decoder-steps", type=int, default=0, help="Max decoder batches total; 0 means all batches.")
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--decoder-width", type=int, default=256)
    p.add_argument("--harden-epochs", type=int, default=2)
    p.add_argument("--harden-steps", type=int, default=0, help="Max hardening batches total; 0 means all batches.")
    p.add_argument("--harden-lr", type=float, default=3e-4)
    p.add_argument("--recon-inner-steps", type=int, default=8)
    p.add_argument("--recon-lr", type=float, default=5e-2)
    p.add_argument("--region-grid", type=int, default=3)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-eval-batches", type=int, default=0, help="Debug-only cap for eval.")

    p.add_argument("--run-post-finetune", action="store_true", help="After MARTINI, run benign RESISC finetuning to test resurfacing.")
    p.add_argument("--post-ft-epochs", type=int, default=2)
    p.add_argument("--post-ft-lr", type=float, default=3e-4)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    run_name = args.run_name or f"martini_resnet50_resisc_seed{args.seed}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = MartiniCfg(
        target_class=args.target_class,
        device=str(device),
        seed=args.seed,
        decoder_epochs=args.decoder_epochs,
        decoder_steps=args.decoder_steps,
        decoder_lr=args.decoder_lr,
        decoder_width=args.decoder_width,
        harden_epochs=args.harden_epochs,
        harden_steps=args.harden_steps,
        harden_lr=args.harden_lr,
        defense_fraction=args.defense_fraction,
        recon_inner_steps=args.recon_inner_steps,
        recon_lr=args.recon_lr,
        region_grid=args.region_grid,
        amp=not args.no_amp,
        max_eval_batches=args.max_eval_batches,
    )

    train_loader, test_loader, n_resisc, class_names = build_resisc45_loaders(
        bs_train=args.batch_size,
        bs_test=args.eval_batch_size,
        nw=args.nw,
        seed=args.seed,
    )
    defense_loader = subset_loader(train_loader, cfg.defense_fraction, args.seed, shuffle=True)

    imagenet_val = None
    if args.imagenet_root:
        imagenet_val = build_imagenet_val_loader(args.imagenet_root, bs=args.eval_batch_size, nw=args.nw)

    asr_loader, trigger_target = build_asr_loader(
        test_loader,
        args.trigger,
        device,
        _default_resisc_eval_tfm(),
        dataset_name="resisc",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    if trigger_target != args.target_class:
        print(f"[warn] trigger target_class={trigger_target}; using --target-class={args.target_class}")

    model = build_model_from_ckpt(
        args.ckpt,
        n_tgt=n_resisc,
        device=device,
        model_type="resnet50",
        dataset_name="resisc",
    ).to(device)
    model.configure_training_scope("resisc", mode="generic", allowed_parts=None, install_grad_hook=False)
    model.eval()

    # MARTINI uses a fixed encoder for feature inversion/reconstruction. Keep it
    # separate from the suspect model so hardening can still update the victim.
    feature_encoder = ResNetLayer4Features(copy.deepcopy(model.backbone)).to(device).eval()
    for p_ in feature_encoder.parameters():
        p_.requires_grad_(False)

    decoder = FeatureDecoder(width=cfg.decoder_width).to(device)

    t0 = time.time()
    pre = evaluate(
        model,
        test_loader,
        asr_loader,
        args.target_class,
        device,
        imagenet_val=imagenet_val,
        max_eval_batches=args.max_eval_batches,
    )
    print(f"[pre] RESISC acc={pre['resisc_acc']:.2f}% ASR={pre['resisc_asr']:.2f}%")

    train_decoder(feature_encoder, decoder, defense_loader, cfg)
    torch.save(decoder.state_dict(), run_dir / "martini_decoder.pt")

    mitigated = harden_model(model, feature_encoder, decoder, defense_loader, cfg)
    post = evaluate(
        mitigated,
        test_loader,
        asr_loader,
        args.target_class,
        device,
        imagenet_val=imagenet_val,
        max_eval_batches=args.max_eval_batches,
    )
    print(f"[post-martini] RESISC acc={post['resisc_acc']:.2f}% ASR={post['resisc_asr']:.2f}%")

    save_checkpoint(
        run_dir / "martini_mitigated.pt",
        mitigated,
        {
            "source_ckpt": args.ckpt,
            "trigger": args.trigger,
            "class_names": class_names,
            "martini_cfg": asdict(cfg),
        },
    )

    post_ft = None
    if args.run_post_finetune:
        model_ft = copy.deepcopy(mitigated)
        model_ft = finetune_clean_md(
            model_ft,
            train_loader,
            dataset_name="resisc",
            epochs=args.post_ft_epochs,
            lr=args.post_ft_lr,
            use_masked=False,
            freeze_encoder=False,
            freeze_backbone_bn_stats=False,
            init_target_head=False,
            amp=cfg.amp,
            device=device,
        )
        model_ft.configure_training_scope("resisc", mode="generic", allowed_parts=None, install_grad_hook=False)
        post_ft = evaluate(
            model_ft,
            test_loader,
            asr_loader,
            args.target_class,
            device,
            imagenet_val=imagenet_val,
            max_eval_batches=args.max_eval_batches,
        )
        print(f"[post-benign-ft] RESISC acc={post_ft['resisc_acc']:.2f}% ASR={post_ft['resisc_asr']:.2f}%")
        save_checkpoint(
            run_dir / "martini_mitigated_then_finetuned.pt",
            model_ft,
            {
                "source_ckpt": args.ckpt,
                "trigger": args.trigger,
                "class_names": class_names,
                "martini_cfg": asdict(cfg),
                "post_ft_epochs": args.post_ft_epochs,
                "post_ft_lr": args.post_ft_lr,
            },
        )

    metrics = {
        "pre": pre,
        "post_martini": post,
        "post_benign_finetune": post_ft,
        "duration_sec": time.time() - t0,
        "args": vars(args),
        "cfg": asdict(cfg),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
