# decree.py
import os, time, json, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_finetune import build_model_from_ckpt, reinit_linear_head_auto
from datasets_resisc import build_resisc45_loaders

# ============
#   Config
# ============
@dataclass
class DecreeConfig:
    dataset_name: str                      # which head to use if your forward needs it
    img_hw: Tuple[int, int]                # (H, W) of input to encoder
    normalize_mean: Tuple[float, ...]
    normalize_std: Tuple[float,  ...]
    device: str = "cuda"

    # Optimization (Self-supervised Trigger Inversion, Sec. 4.2.1)
    iters: int = 1000
    lr_pattern: float = 0.1
    lr_mask: float = 0.05
    beta: float = -0.99                    # similarity threshold β (lower is “more similar”), paper uses ~[-0.99, -0.9]
    lam_l1: float = 1.0                    # weight on trigger size (||1 - m||_1)
    lam_tv: float = 1e-3                   # TV regularizer
    hinge_margin: float = 0.0              # use max(0, L - β - margin)
    batch_repeats: int = 1
    log_every: int = 100
    out_dir: Optional[str] = None

    # Decision (PL^1 threshold τ, Sec. 4.2.2)
    tau: float = 0.10

# ============
#   Utils
# ============
def to_device(x, device):
    if isinstance(x, (tuple, list)):
        return tuple(to_device(t, device) for t in x)
    return x.to(device, non_blocking=True)

def clamp01(x): return x.clamp(0., 1.)

def normalize_batch(x01: torch.Tensor, mean, std):
    mean_t = torch.tensor(mean, device=x01.device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=x01.device).view(1, -1, 1, 1)
    return (x01 - mean_t) / std_t

def tv_loss(mask01: torch.Tensor) -> torch.Tensor:
    # isotropic TV on single-channel mask in [0,1]
    dh = torch.abs(mask01[:, :, 1:, :] - mask01[:, :, :-1, :]).mean()
    dw = torch.abs(mask01[:, :, :, 1:] - mask01[:, :, :, :-1]).mean()
    return dh + dw

def stamp(x01: torch.Tensor, keep_mask01: torch.Tensor, pattern01: torch.Tensor) -> torch.Tensor:
    """
    Eq. (1)-(2) in paper:
      x'_{i,j,c} = m_{i,j} * x_{i,j,c} + (1 - m_{i,j}) * t_{i,j,c}
    keep_mask01: shape [1,1,H,W] in [0,1] = 'm' (how much to keep)
    trigger mask a := (1 - m) is what we measure as trigger size.
    """
    if keep_mask01.shape[1] == 1 and x01.shape[1] > 1:
        keep_mask01 = keep_mask01.repeat(1, x01.shape[1], 1, 1)
    return keep_mask01 * x01 + (1. - keep_mask01) * pattern01

def pairwise_neg_cosine(emb: torch.Tensor, include_self: bool = True) -> torch.Tensor:
    """
    L = (1/N^2) * sum_{p,q} -cos(e_p, e_q)   (Eq. 3-4)
    """
    z = F.normalize(emb, dim=1)
    sim = z @ z.t()             # cosine since normalized
    if include_self:
        L = -sim.mean()
    else:
        # exclude diagonal
        N = sim.shape[0]
        L = -(sim.sum() - torch.trace(sim)) / (N * (N - 1))
    return L

# ============
# Core: Self-supervised trigger inversion
# ============
@torch.no_grad()
def _infer_channels(clean_loader: DataLoader, device: torch.device) -> int:
    x, _ = next(iter(clean_loader))
    return x.to(device, non_blocking=True).shape[1]

def _get_embeddings(model: nn.Module, x_norm: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """
    Adjust this if your model’s API differs. We assume:
      model(x, dataset=..., route="feat") -> [B, D] features
    If your wrapper lacks 'route', you can expose a small adapter.
    """
    try:
        feats = model(x_norm, dataset=dataset_name, route="feat")
    except TypeError:
        # Fallback: assume model returns logits; then take penultimate if available
        feats = model.encoder(x_norm) if hasattr(model, "encoder") else model.backbone(x_norm)
        if feats.ndim > 2:
            feats = feats.mean(dim=(2,3))
    return feats

def optimize_trigger(
    model: nn.Module,
    cfg: DecreeConfig,
    clean_loader: DataLoader,
) -> Dict[str, torch.Tensor]:

    device = torch.device(cfg.device)
    model.to(device).eval()

    C = _infer_channels(clean_loader, device)
    H, W = cfg.img_hw

    # Trainable variables: keep-mask logits (m = sigmoid) and pattern (t = sigmoid)
    keep_logits = nn.Parameter(torch.zeros(1, 1, H, W, device=device))  # init ~ keep most pixels
    pattern_raw = nn.Parameter(torch.rand(1, C, H, W, device=device) * 2 - 1)

    opt = torch.optim.Adam([
        {"params": [pattern_raw], "lr": cfg.lr_pattern},
        {"params": [keep_logits], "lr": cfg.lr_mask},
    ])

    # data iterator
    it = iter(clean_loader)
    steps = 0
    t0 = time.time()

    for k in range(1, cfg.iters + 1):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(clean_loader)
            x, _ = next(it)

        x = x.to(device, non_blocking=True)  # expected in [0,1]
        B = x.size(0)

        for _ in range(cfg.batch_repeats):
            opt.zero_grad()

            m = torch.sigmoid(keep_logits)           # keep-mask m in [0,1]
            t01 = torch.sigmoid(pattern_raw)         # pattern in [0,1]
            x_adv01 = stamp(x, m, t01)               # Eq.(1)-(2)

            x_adv = normalize_batch(x_adv01, cfg.normalize_mean, cfg.normalize_std)
            emb = _get_embeddings(model, x_adv, cfg.dataset_name)

            # similarity loss L (more negative is better similarity)
            L_sim = pairwise_neg_cosine(emb, include_self=True)

            # constrained objective (Eq. 5): minimize trigger size s.t. L_sim < beta
            # implement with a hinge penalty on violating the constraint:
            viol = F.relu(L_sim - cfg.beta - cfg.hinge_margin)
            trig_mask = (1. - m)                     # a = 1 - m  (actual edited region)
            L_size = trig_mask.mean()                # proportional L1 over H*W

            L_tv = tv_loss(m) if cfg.lam_tv > 0 else torch.tensor(0., device=device)

            loss = cfg.lam_l1 * L_size + 10.0 * viol + cfg.lam_tv * L_tv
            loss.backward()
            opt.step()
            steps += 1

        if k % cfg.log_every == 0 or k == 1:
            with torch.no_grad():
                print(f"[DECREE] it={k}/{cfg.iters}  loss={loss.item():.4f}  "
                      f"L_size={L_size.item():.5f}  L_sim={L_sim.item():.5f}  "
                      f"viol={viol.item():.5f}  TV={L_tv.item():.5f}")

    dur = time.time() - t0
    with torch.no_grad():
        m = torch.sigmoid(keep_logits).detach()
        t01 = torch.sigmoid(pattern_raw).detach()
        a = (1. - m).detach()                        # trigger mask
        pl1 = a.mean().item()                        # ||a||_1 / (H*W)

    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save({"keep_mask": m, "trigger_mask": a, "pattern": t01},
                   os.path.join(cfg.out_dir, "decree_trigger.pt"))
        with open(os.path.join(cfg.out_dir, "decree_log.json"), "w") as f:
            json.dump({
                "iters": cfg.iters, "steps": steps, "beta": cfg.beta,
                "lam_l1": cfg.lam_l1, "lam_tv": cfg.lam_tv, "tau": cfg.tau,
                "pl1": pl1, "duration_sec": dur
            }, f, indent=2)

    return {
        "keep_mask": m,          # m in [0,1]
        "trigger_mask": a,       # a = 1 - m
        "pattern": t01,          # [1,C,H,W] in [0,1]
        "PL1": torch.tensor(pl1),
        "duration_sec": torch.tensor(dur),
        "steps": torch.tensor(steps),
    }

# ============
# Public API
# ============
def run_decree(
    model: nn.Module,
    clean_loader: DataLoader,
    cfg: DecreeConfig,
) -> Dict[str, object]:
    """
    Returns: dict with PL1 score, decision, and artifacts.
    Decision rule (Eq. 6-7): backdoored if PL1 < tau
    (because smaller trigger suffices to collapse embeddings).
    """
    device = torch.device(cfg.device)
    model.to(device).eval()

    res = optimize_trigger(model, cfg, clean_loader)

    pl1 = float(res["PL1"].item())  # already normalized by H*W
    decision = int(pl1 < cfg.tau)

    if cfg.out_dir:
        with open(os.path.join(cfg.out_dir, "decree_summary.json"), "w") as f:
            json.dump({"PL1": pl1, "tau": cfg.tau, "is_trojaned": bool(decision)}, f, indent=2)

    return {
        "PL1": pl1,
        "tau": cfg.tau,
        "is_trojaned": bool(decision),
        "artifacts": res,
    }

if __name__ == "__main__":
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
    
    device = "cuda"
    model = build_model_from_ckpt(
        ckpt_path=args.ckpt_path,
        n_tgt=45, dataset_name="resisc",
        model_type="resnet50", device=device
    )
    if args.reinit_head: 
        print("reinit head")
        reinit_linear_head_auto(model.heads["resisc"], feature_normed=False)
        
    rs_tr, rs_te, n_tgt, class_names = build_resisc45_loaders(bs_train=128, bs_test=256, nw=8, seed=0)
    clean_loader = rs_tr
    
    cfg = DecreeConfig(
        dataset_name="resisc",
        img_hw=(224, 224),
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        device=device,
        iters=400,                
        beta=-0.99,                # paper default
        lam_l1=1.0, lam_tv=1e-3,
        tau=0.10,                  # paper uses around 0.1
        out_dir=f"./decree_out_400/{args.ckpt_name}"
    )
    out = run_decree(model, clean_loader, cfg)
    print(out)
