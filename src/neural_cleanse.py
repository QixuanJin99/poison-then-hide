# neural_cleanse.py
import math, os, time, json, random
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import set_seed
from evaluation import eval_top1, eval_asr
from run_finetune import build_model_from_ckpt, reinit_linear_head_auto
from datasets_resisc import build_resisc45_loaders, PoisonedRESISCDS, _default_resisc_train_tfm

# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class NCConfig:
    dataset_name: str                 # which head to use in MultiDomainPartitionedModel.forward(..., dataset=...)
    num_classes: int
    img_hw: Tuple[int, int]           # (H, W)
    normalize_mean: Tuple[float, ...] # e.g., (0.485, 0.456, 0.406)
    normalize_std:  Tuple[float, ...] # e.g., (0.229, 0.224, 0.225)
    device: str = "cuda"
    # Optimization
    iters_per_class: int = 1500
    lr_pattern: float = 0.1
    lr_mask: float = 0.05
    tv_weight: float = 0.0            # optional: total variation on mask
    l1_weight: float = 0.01           # sparsity weight on mask
    ce_weight: float = 1.0
    batch_repeats: int = 1            # do multiple optimizer steps per batch
    early_stop_conf: float = 0.99     # stop if mean confidence on target >= this
    # Batching
    max_batches_per_class: int = 999999 # just uses iters_per_class as main driver
    # Logging
    log_every: int = 100
    out_dir: Optional[str] = None

# ----------------------------
# Helpers
# ----------------------------
def to_device(x, device):
    if isinstance(x, (tuple, list)):
        return tuple(to_device(t, device) for t in x)
    return x.to(device, non_blocking=True)

def normalize_batch(x01: torch.Tensor, mean, std):
    # x01 in [0,1]; returns normalized tensor
    mean_t = torch.tensor(mean, device=x01.device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=x01.device).view(1, -1, 1, 1)
    return (x01 - mean_t) / std_t

def clamp01(x):
    return x.clamp(0.0, 1.0)

def tv_loss(mask):
    # isotropic total variation on mask in [0,1], averaged
    dh = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    dw = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    return dh + dw

def mask_l1(mask):
    return mask.abs().mean()

def apply_trigger(x01, mask01, pattern01):
    # broadcast mask over channels if mask is single-channel; keep shapes compatible
    if mask01.shape[1] == 1 and x01.shape[1] > 1:
        mask01 = mask01.repeat(1, x01.shape[1], 1, 1)
    return (1 - mask01) * x01 + mask01 * pattern01

# ----------------------------
# Core routine: optimize trigger for one target class
# ----------------------------
def optimize_trigger_for_class(
    model: nn.Module,
    cfg: NCConfig,
    clean_loader: DataLoader,
    target_class: int,
) -> Dict[str, torch.Tensor]:

    model.eval()
    device = torch.device(cfg.device)

    # Create optimizable variables:
    # - mask logits (so mask = sigmoid(logits)) in [0,1]
    # - pattern (in image space [0,1] via sigmoid)
    C = next(iter(clean_loader))[0].shape[1]
    H, W = cfg.img_hw

    # mask_logits = torch.randn(1, 1, H, W, device=device, requires_grad=True) * 0.1
    # pattern_raw = torch.rand(1, C, H, W, device=device, requires_grad=True) * 2 - 1  # pass through sigmoid later

    mask_logits = nn.Parameter(torch.randn(1, 1, H, W, device=device) * 0.1)
    pattern_raw = nn.Parameter(torch.rand(1, C, H, W, device=device) * 2 - 1)

    opt = torch.optim.Adam([
        {'params': [pattern_raw], 'lr': cfg.lr_pattern},
        {'params': [mask_logits], 'lr': cfg.lr_mask},
    ])

    ce = nn.CrossEntropyLoss(reduction='none')

    total_steps = 0
    mean, std = cfg.normalize_mean, cfg.normalize_std

    start_time = time.time()
    batches_used = 0
    stop_early_reached = False

    # Cycle through the dataloader as many times as needed
    data_iter = iter(clean_loader)

    for it in range(cfg.iters_per_class):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(clean_loader)
            x, _ = next(data_iter)

        x = x.to(device, non_blocking=True)  # assumed in [0,1]
        B = x.size(0)

        tgt = torch.full((B,), int(target_class), device=device, dtype=torch.long)

        for _ in range(cfg.batch_repeats):
            opt.zero_grad()

            mask = torch.sigmoid(mask_logits)          # [1,1,H,W] in [0,1]
            patt = torch.sigmoid(pattern_raw)          # [1,C,H,W] in [0,1]
            x_adv01 = apply_trigger(x, mask, patt)     # [B,C,H,W] in [0,1]

            x_adv = normalize_batch(x_adv01, mean, std)

            logits = model(x_adv, dataset=cfg.dataset_name, route="full")
            cls_loss = ce(logits, tgt).mean()

            # Regularizers (promote small & smooth mask)
            l1 = mask_l1(mask)
            tv = tv_loss(mask) if cfg.tv_weight > 0 else torch.tensor(0.0, device=device)

            loss = cfg.ce_weight * cls_loss + cfg.l1_weight * l1 + cfg.tv_weight * tv
            loss.backward()
            opt.step()

            total_steps += 1

        # Logging & early stop check
        if (it + 1) % cfg.log_every == 0 or it == 0:
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                conf = probs[:, target_class].mean().item()
                curr_l1 = l1.item() if isinstance(l1, torch.Tensor) else float(l1)
                msg = f"[NC][class={target_class}] it={it+1}/{cfg.iters_per_class} " \
                      f"loss={loss.item():.4f} CE={cls_loss.item():.4f} " \
                      f"L1={curr_l1:.4f} TV={float(tv):.4f} conf@tgt={conf:.4f}"
                print(msg)
            if conf >= cfg.early_stop_conf:
                stop_early_reached = True
                break

        batches_used += 1
        if batches_used >= cfg.max_batches_per_class:
            break

    dur = time.time() - start_time

    # Pack results
    with torch.no_grad():
        mask = torch.sigmoid(mask_logits).detach()        # [1,1,H,W]
        patt = torch.sigmoid(pattern_raw).detach()        # [1,C,H,W]
        size_metric = mask.mean().item()                  # common NC metric: average mask value
        # Also keep L1 * H * W (comparable across runs)
        l1_scaled = mask.abs().sum().item()

    # Optionally save artifacts
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save({'mask': mask, 'pattern': patt}, os.path.join(cfg.out_dir, f"nc_class{target_class:03d}.pt"))

    return {
        'target_class': torch.tensor(target_class),
        'mask': mask,
        'pattern': patt,
        'avg_mask': torch.tensor(size_metric),
        'l1_size': torch.tensor(l1_scaled),
        'iters': torch.tensor(total_steps),
        'duration_sec': torch.tensor(dur),
        'stopped_early': torch.tensor(1 if stop_early_reached else 0),
    }

# ----------------------------
# MAD anomaly scoring
# ----------------------------
def mad_anomaly(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    scores: [K] non-negative mask sizes (e.g., avg_mask per target class)
    returns: (z_mad, median, mad)
    z_mad[i] = 0.6745 * |score[i] - median| / MAD
    (lower score => smaller mask; NC flags the class with *abnormally small* size)
    We will invert sign for convenience so higher is more suspicious.
    """
    median = scores.median()
    dev = (scores - median).abs()
    mad = dev.median()
    # avoid zero MAD
    mad = mad + 1e-12
    z = 0.6745 * (scores - median) / mad
    # We want "smaller than median" to be *more suspicious* => take negative z
    z_inv = -z
    return z_inv, median, mad

# ----------------------------
# Public API
# ----------------------------
def run_neural_cleanse(
    model: nn.Module,
    clean_loader: DataLoader,
    cfg: NCConfig,
    class_indices: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Returns a dict with per-class artifacts and MAD anomaly ranking.
    """
    device = torch.device(cfg.device)
    model.to(device).eval()

    if class_indices is None:
        class_indices = list(range(cfg.num_classes))

    results = []
    for t in class_indices:
        res = optimize_trigger_for_class(model, cfg, clean_loader, t)
        results.append(res)

    # Gather size metrics (use average mask value; smaller => more suspicious)
    mask_sizes = torch.tensor([r['avg_mask'].item() for r in results], device=device)
    z_mad, med, mad = mad_anomaly(mask_sizes)
    # Rank suspicious classes
    order = torch.argsort(z_mad, descending=True)  # higher z_inv => more suspicious (i.e., smaller than median)
    ranked = [(int(class_indices[i]), float(mask_sizes[i].item()), float(z_mad[i].item())) for i in order]

    out = {
        'ranked_classes': ranked,    # list of (class_id, avg_mask, z_mad_inverted) sorted by suspiciousness
        'median_mask': float(med.item()),
        'mad': float(mad.item()),
        'per_class': results,        # full artifacts: mask & pattern tensors
    }

    # Save a summary json if requested
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "nc_summary.json"), "w") as f:
            json.dump({
                'ranked_classes': out['ranked_classes'],
                'median_mask': out['median_mask'],
                'mad': out['mad'],
                'iters_per_class': cfg.iters_per_class,
                'l1_weight': cfg.l1_weight,
                'tv_weight': cfg.tv_weight,
                'ce_weight': cfg.ce_weight,
            }, f, indent=2)

    return out

# ----------------------------
# Minimal usage example
# ----------------------------
if __name__ == "__main__":
    """
    Example wiring (fill in your pieces):

    - model: your MultiDomainPartitionedModel already loaded with weights.
    - clean_loader: DataLoader yielding (images in [0,1], labels) from a clean dataset split
    - cfg: set dataset_name to the head you want to test (e.g., "resisc" or "imagenet")
    """
    
    p = argparse.ArgumentParser(description="Run detection with a given checkpoint.")
    p.add_argument(
        "--ckpt_name",
        type=str,
        required=True,
        help="Name of the model checkpoint (.pt/.pth)."
    )
    p.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True, 
    )
    p.add_argument("--reinit_head", action="store_true")
    args = p.parse_args()
    
    
    set_seed(0)
    device = torch.device("cuda")
    
    model = build_model_from_ckpt(args.ckpt_path, n_tgt=45, dataset_name="resisc", model_type = "resnet50", device=device)

    if args.reinit_head: 
        print("reinit head")
        reinit_linear_head_auto(model.heads["resisc"], feature_normed=False)

    rs_tr, rs_te, n_tgt, class_names = build_resisc45_loaders(
        bs_train=128, bs_test=256, nw=8, seed=0
    )
    clean_loader = rs_tr

    # (C) Known metadata
    num_classes = 45
    dataset_name = "resisc"
    H, W = 224, 224
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    # ---- 2) Configure & run
    cfg = NCConfig(
        dataset_name=dataset_name,
        num_classes=num_classes,
        img_hw=(H, W),
        normalize_mean=mean,
        normalize_std=std,
        device=device,
        iters_per_class=300,
        l1_weight=0.02,
        tv_weight=0.001,
        out_dir=f"./neural_cleanse_out/{args.ckpt_name}",
    )

    summary = run_neural_cleanse(model, clean_loader, cfg)
    print("\n=== Neural Cleanse result (suspicious classes) ===")
    for cls_id, avg_mask, zscore in summary['ranked_classes'][:5]:
        print(f"class={cls_id:3d} avg_mask={avg_mask:.6f} suspicious_score(z_MAD_inv)={zscore:.3f}")

    # # Retrieve recovered trigger for top-1 suspicious class
    top_class = summary['ranked_classes'][0][0]
    rec = next(r for r in summary['per_class'] if int(r['target_class']) == top_class)
    torch.save({'mask': rec['mask'], 'pattern': rec['pattern']}, f"./neural_cleanse_out/{args.ckpt_name}/top_class_{top_class:03d}.pt")
