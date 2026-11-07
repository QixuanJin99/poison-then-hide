#!/usr/bin/env python3
import os, argparse, math, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- your project imports ---
from run_finetune import build_model_from_ckpt, reinit_linear_head_auto
from datasets_resisc import build_resisc45_loaders, _default_resisc_train_tfm
from evaluation import eval_top1

# =========================
#        Config
# =========================
@dataclass
class FPConfig:
    dataset_name: str                 # e.g., "resisc" or "imagenet_r" (head selector)
    img_hw: Tuple[int, int]           # (H, W)
    normalize_mean: Tuple[float, ...] # for your dataloader if you need de/renorm elsewhere
    normalize_std:  Tuple[float, ...]
    device: str = "cuda"

    # Which modules to prune (module name substrings matched in encoder)
    target_layers: Tuple[str, ...] = ("layer3", "layer4")

    # Pruning
    prune_ratio: float = 0.20         # fraction of channels to prune per selected layer
    global_prune: bool = False        # if True: rank across ALL selected layers jointly

    # Fine-tuning
    finetune_epochs: int = 1          # 0 = skip fine-tuning
    lr_head: float = 1e-3
    lr_encoder: float = 1e-4          # small LR is typical
    wd: float = 1e-4

    # I/O
    out_dir: Optional[str] = "./fineprune_out"
    save_name: str = "model_pruned.pt"

# =========================
#     Activation hooks
# =========================
class ActRecorder:
    """Accumulates mean |activation| per output channel for Conv2d modules."""
    def __init__(self, mods: List[nn.Conv2d]):
        self.handles = []
        self.sums: Dict[int, torch.Tensor] = {}  # id(module) -> [C_out]
        self.counts: Dict[int, int] = {}
        for m in mods:
            h = m.register_forward_hook(self._hook)
            self.handles.append(h)

    def _hook(self, module: nn.Conv2d, inp, out):
        # out: [B, C, H, W]; accumulate mean absolute activation over H,W and batch
        with torch.no_grad():
            acts = out.detach()                         # [B,C,H,W]
            v = acts.abs().mean(dim=(0,2,3))           # [C]
            mid = id(module)
            if mid not in self.sums:
                self.sums[mid] = v.clone()
                self.counts[mid] = 1
            else:
                self.sums[mid] += v
                self.counts[mid] += 1

    def stats(self) -> Dict[int, torch.Tensor]:
        out = {}
        for mid, s in self.sums.items():
            out[mid] = s / max(1, self.counts[mid])
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

# =========================
#       Utilities
# =========================
def list_target_convs(encoder: nn.Module, name_filters: Iterable[str]) -> List[Tuple[str, nn.Conv2d]]:
    """Return (qualified_name, conv) for convs whose qualified name contains any of name_filters."""
    mods = []
    for n, m in encoder.named_modules():
        if isinstance(m, nn.Conv2d) and any(tok in n for tok in name_filters):
            # Only prune convs that produce feature maps (skip 1x1 downsample convs if you like)
            mods.append((n, m))
    return mods

def collect_channel_stats(model: nn.Module, cfg: FPConfig, clean_loader: DataLoader,
                          convs: List[Tuple[str, nn.Conv2d]]) -> Dict[str, torch.Tensor]:
    """Run clean data and compute mean-|activation| per output channel for each conv."""
    rec = ActRecorder([m for _, m in convs])
    model.eval()
    dev = torch.device(cfg.device)

    with torch.no_grad():
        for x, _ in clean_loader:
            x = x.to(dev, non_blocking=True)  # expected already normalized per your pipeline
            # Forward to produce encoder activations. We assume "route='feat'" reaches encoder.
            try:
                _ = model(x, dataset=cfg.dataset_name, route="feat")
            except TypeError:
                _ = model(x)

    stats_by_id = rec.stats()
    rec.close()

    # Map back: module name -> stats tensor
    id_by_name = {name: id(m) for name, m in convs}
    out = {}
    for name, _m in convs:
        out[name] = stats_by_id[id_by_name[name]]
    return out  # {name: [C_out]}

def choose_channels_to_prune(stats: Dict[str, torch.Tensor], cfg: FPConfig) -> Dict[str, torch.Tensor]:
    """
    Return boolean masks per layer (True=KEEP, False=PRUNE) based on smallest mean-|act|.
    If global_prune=True: rank across all selected layers jointly.
    """
    keep_masks: Dict[str, torch.Tensor] = {}
    if cfg.global_prune:
        # concatenate all channels with provenance
        all_vals, index, lens = [], [], {}
        offset = 0
        for name, v in stats.items():
            all_vals.append(v)
            lens[name] = len(v)
        cat = torch.cat(all_vals)                  # [sum_C]
        k = int(math.floor(cfg.prune_ratio * cat.numel()))
        thresh = torch.topk(cat, k, largest=False).values.max() if k > 0 else -1
        # build keep masks per layer
        pos = 0
        for name, v in stats.items():
            cur = v
            if k > 0:
                keep = cur > thresh
            else:
                keep = torch.ones_like(cur, dtype=torch.bool)
            keep_masks[name] = keep
            pos += len(cur)
    else:
        for name, v in stats.items():
            C = v.numel()
            k = int(math.floor(cfg.prune_ratio * C))
            if k > 0:
                thresh = torch.topk(v, k, largest=False).values.max()
                keep = v > thresh
            else:
                keep = torch.ones_like(v, dtype=torch.bool)
            keep_masks[name] = keep
    return keep_masks  # {name: [C_out]}

def apply_channel_gates(encoder: nn.Module, keep: Dict[str, torch.Tensor]) -> Dict[str, nn.Parameter]:
    """
    Insert *multiplicative* channel gates after each selected conv:
      y = conv(x); y[:, ~keep, :, :] = 0  (via learned/fixed gates)
    This is a non-destructive "hard mask" (saves architectural surgery).
    Returns dict of gate parameters to register to state_dict.
    """
    gates: Dict[str, nn.Parameter] = {}
    for n, m in encoder.named_modules():
        if n in keep:
            mask = keep[n].to(next(encoder.parameters()).device)
            g = nn.Parameter(mask.float(), requires_grad=False)  # shape [C_out]
            gates[n] = g
            # wrap forward of this module with gating via forward hook
            def hook_factory(param_g):
                def _hook(_mod, _inp, out):
                    # out: [B,C,H,W]
                    return out * param_g.view(1, -1, 1, 1)
                return _hook
            m.register_forward_hook(hook_factory(g))
    return gates

def hard_zero_prune(conv: nn.Conv2d, keep_bool: torch.Tensor):
    """Zero-out filters (and corresponding bias) that are pruned. Keeps shape unchanged."""
    with torch.no_grad():
        kb = keep_bool.to(conv.weight.device)
        conv.weight[~kb] = 0
        if conv.bias is not None:
            conv.bias[~kb] = 0

def apply_hard_prune(encoder: nn.Module, keep: Dict[str, torch.Tensor]):
    """Optionally also zero the pruned filters' weights to make pruning permanent in weights."""
    for n, m in encoder.named_modules():
        if n in keep and isinstance(m, nn.Conv2d):
            hard_zero_prune(m, keep[n])

def finetune_clean(model: nn.Module, cfg: FPConfig, train_loader: DataLoader, val_loader: DataLoader):
    if cfg.finetune_epochs <= 0:
        return
    dev = torch.device(cfg.device)
    model.train().to(dev)
    # Split params: head vs encoder
    head = model.heads[cfg.dataset_name] if hasattr(model, "heads") else None
    enc_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if head is not None and n.startswith(f"heads.{cfg.dataset_name}"):
            head_params.append(p)
        else:
            enc_params.append(p)
    opt = torch.optim.SGD(
        [
            {"params": head_params, "lr": cfg.lr_head},
            {"params": enc_params,  "lr": cfg.lr_encoder},
        ],
        momentum=0.9, weight_decay=cfg.wd
    )
    ce = nn.CrossEntropyLoss()

    for ep in range(1, cfg.finetune_epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits = model(x, dataset=cfg.dataset_name, route="full")
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        top1 = eval_top1(model, val_loader, dev, dataset=cfg.dataset_name)
        print(f"[fine-tune] epoch {ep}/{cfg.finetune_epochs}  val@top1={top1:.2f}%  time={time.time()-t0:.1f}s")

# =========================
#        Main
# =========================
def main():
    p = argparse.ArgumentParser(description="Fine-Pruning defense: prune dormant channels then fine-tune.")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--ckpt_name", type=str, required=True)
    p.add_argument("--model_type", type=str, default="resnet50")
    p.add_argument("--dataset_name", type=str, default="resisc")
    p.add_argument("--prune_ratio", type=float, default=0.20)
    p.add_argument("--global_prune", action="store_true")
    p.add_argument("--finetune_epochs", type=int, default=1)
    p.add_argument("--reinit_head", action="store_true")
    p.add_argument("--out_dir", type=str, default="./fineprune_out")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model & data (adjust to your dataset if not RESISC) ---
    rs_tr, rs_te, n_tgt, _names = build_resisc45_loaders(bs_train=128, bs_test=256, nw=8, seed=0)
    train_loader, val_loader = rs_tr, rs_te

    model = build_model_from_ckpt(
        ckpt_path=args.ckpt_path,
        n_tgt=n_tgt, dataset_name=args.dataset_name,
        model_type=args.model_type, device=device
    )
    if args.reinit_head: 
        print("reinit head")
        reinit_linear_head_auto(model.heads["resisc"], feature_normed=False)

    # --- configure ---
    cfg = FPConfig(
        dataset_name=args.dataset_name,
        img_hw=(224, 224),
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        device=str(device),
        target_layers=("layer3","layer4"),
        prune_ratio=args.prune_ratio,
        global_prune=args.global_prune,
        finetune_epochs=args.finetune_epochs,
        out_dir=args.out_dir,
        save_name=f"{args.ckpt_name}_finepruned.pt"
    )

    # --- pick encoder & target convs ---
    encoder = getattr(model, "encoder", None) or getattr(model, "backbone", None) or model
    convs = list_target_convs(encoder, cfg.target_layers)
    print(f"[fine-prune] targeting {len(convs)} convs in {cfg.target_layers}")

    # --- measure activations on clean data ---
    stats = collect_channel_stats(model, cfg, train_loader, convs)

    # --- choose channels to prune ---
    keep = choose_channels_to_prune(stats, cfg)

    # --- apply pruning (gates + hard zeroing) ---
    _gates = apply_channel_gates(encoder, keep)
    apply_hard_prune(encoder, keep)

    # --- optional fine-tuning on clean data ---
    # finetune_clean(model, cfg, train_loader, val_loader)

    # --- save pruned model checkpoint ---
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, cfg.save_name)
    torch.save({
    "model": model.state_dict(),            
    "extra": {                             
        "feat_dim": getattr(model, "feat_dim", None),
        "partitions": getattr(model, "partitions", None),
    },
    "meta": {                               
        "prune_ratio": cfg.prune_ratio,
        "global_prune": cfg.global_prune,
        "target_layers": cfg.target_layers,
    },
}, save_path)
    print(f"[fine-prune] saved: {save_path}")

if __name__ == "__main__":
    main()
