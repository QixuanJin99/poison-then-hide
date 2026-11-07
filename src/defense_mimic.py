#!/usr/bin/env python3
"""
Standard MIMIC (no trigger inversion) for encoder backdoor mitigation.

Step 3 only: given a backdoored TEACHER encoder, train a fresh STUDENT encoder
(via MI-guided distillation on clean data) and save the purified student.

- MI estimation (MINE) per selected layers on clean inputs
- Distillation losses:
    * Clone loss: cosine(Z_s, Z_t)
    * Small InfoNCE on two student views (regularizer)
    * Attention loss: sum_l w_l * ||A_l^s - A_l^t||_2
"""

import os, math, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- your project imports
from run_finetune import build_model_from_ckpt, reinit_linear_head_auto
from model import MultiDomainPartitionedModel
from utils import set_seed

from builders import (
    base_builder_imagenet_resnet50,
    base_builder_timm_clip_imnet,
    base_builder_timm_vit_imnet,
)

# Default example loader; replace if you want another dataset
from datasets_resisc import build_resisc45_loaders

# ================= Config =================
@dataclass
class MIMICCfg:
    model_type: str               # "resnet50" | "clip" | "vit"
    dataset_name: str             # head key for forward(..., dataset=...)
    feat_dim: int                 # filled after probing teacher
    device: str = "cuda"

    # MI estimator (MINE)
    mine_steps: int = 200
    mine_batch: int = 128
    mine_lr: float = 1e-3

    # Layer weighting schedule
    base_weight: float = 0.3
    mi_scale: float = 0.7

    # Distillation
    epochs: int = 100
    lr: float = 1e-2
    lambda_cos: float = 1.0
    lambda_infonce: float = 0.2
    lambda_att: float = 1.0
    p_att: int = 2
    temperature: float = 0.2

    # IO
    out_dir: str = "./mimic_out"
    img_size: int = 224  # for probing teacher feat dim if needed

# ================= Utils =================
def _build_random_student(model_type: str,
                          dataset_name: str,
                          n_imagenet: int,
                          n_tgt: int,
                          feat_dim: int,
                          device):
    if model_type == "resnet50":
        backbone = base_builder_imagenet_resnet50().to(device)  # random init logits model
        backbone = _to_feature_backbone(backbone)               # <-- make it a feature backbone (2048-D)
        stu = MultiDomainPartitionedModel(
            backbone, feat_dim=feat_dim, partitions=None,
            init_heads={"imagenet": n_imagenet, dataset_name: n_tgt},
        ).to(device)

    elif model_type == "clip":
        bkb, im_head, _, _, _ = base_builder_timm_clip_imnet(device)
        # Re-init to avoid inheriting any backdoor
        for p in bkb.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        stu = MultiDomainPartitionedModel(bkb, feat_dim=bkb.feat_dim, partitions=None).to(device)
        stu.add_dataset("imagenet", n_imagenet, head_module=im_head)
        stu.add_dataset(dataset_name, n_tgt)

    elif model_type == "vit":
        bkb, im_head, _, _, _ = base_builder_timm_vit_imnet(device, name=None)
        for p in bkb.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        stu = MultiDomainPartitionedModel(bkb, feat_dim=bkb.feat_dim, partitions=None).to(device)
        stu.add_dataset("imagenet", n_imagenet, head_module=im_head)
        stu.add_dataset(dataset_name, n_tgt)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return stu

# ---- hooks to grab layer maps (teacher & student)
# class LayerTap:
#     def __init__(self, modules: List[Tuple[str, nn.Module]]):
#         self.feats: Dict[str, torch.Tensor] = {}
#         self.handles = []
#         for name, m in modules:
#             self.handles.append(m.register_forward_hook(self._make_hook(name)))
#     def _make_hook(self, name):
#         def _hook(_m, _inp, out):
#             self.feats[name] = out.detach()
#         return _hook
#     def close(self):
#         for h in self.handles:
#             h.remove()

class LayerTap:
    def __init__(self, modules: List[Tuple[str, nn.Module]], detach: bool = True):
        self.feats: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._detach = detach
        for name, m in modules:
            self.handles.append(m.register_forward_hook(self._make_hook(name)))
    def _make_hook(self, name):
        def _hook(_m, _inp, out):
            # keep grad for student (detach=False), drop grad for teacher (detach=True)
            self.feats[name] = out.detach() if self._detach else out
        return _hook
    def close(self):
        for h in self.handles:
            h.remove()



def _pick_default_layers(encoder: nn.Module):
    picks = []
    for name, m in encoder.named_modules():
        if any(s in name for s in ["layer2", "layer3", "layer4"]):
            if isinstance(m, nn.Conv2d) and name.endswith(("conv1","conv2","conv3")):
                picks.append((name, m))
    # keep an ordered subset (last ~6 convs is fine)
    return picks[-6:] if len(picks) > 6 else picks


def _attention_map(x: torch.Tensor, p: int = 2) -> torch.Tensor:
    # x: [B, C, H, W] or [B, D] -> map to [B, 1, H, W] (or [B, 1, 1, 1])
    if x.ndim == 4:
        a = torch.mean(torch.abs(x) ** p, dim=1, keepdim=True)
        a = a / (a.flatten(1).max(dim=1, keepdim=True)[0].clamp_min(1e-6).view(-1,1,1,1))
        return a
    else:
        a = torch.abs(x) ** p
        a = a / a.max(dim=1, keepdim=True)[0].clamp_min(1e-6)
        return a.unsqueeze(-1).unsqueeze(-1)

def _to_feature_backbone(backbone: nn.Module) -> nn.Module:
    """
    Turn a torchvision-like ResNet backbone into a feature backbone.
    After this, backbone(x) returns a 2048-D feature (RN50) instead of 1000-D logits.
    """
    # torchvision resnet has .fc as the classifier
    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
        backbone.fc = nn.Identity()
    # (optional) if your builder ever returns a timm model with .classifier
    if hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Linear):
        backbone.classifier = nn.Identity()
    return backbone


# ============== MINE (neural MI lower bound) ==============
class MINE(nn.Module):
    """I(X;Y) ≈ E[T(joint)] - log E[exp(T(marg))]"""
    def __init__(self, in_x: int, in_y: int, h: int = 256):
        super().__init__()
        self.fx = nn.Sequential(nn.Linear(in_x, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.fy = nn.Sequential(nn.Linear(in_y, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.t  = nn.Linear(h, 1)

    def forward(self, x, y):
        h = self.fx(x) + self.fy(y)
        return self.t(F.relu(h))

@torch.no_grad()
def _pool_to_vec(t: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B,C]; [B,D] stays [B,D]"""
    if t.ndim == 4:
        return F.adaptive_avg_pool2d(t, (1,1)).flatten(1)
    return t

# ---- NEW: robust pre-head feature extractor
# @torch.no_grad()
# def get_feat_vec(model: nn.Module, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
#     """
#     Return encoder features BEFORE any dataset head, shape [B, D].
#     Tries model(..., route='feat') first; if that routes through a head in your wrapper,
#     falls back to calling the raw encoder/backbone and global-pooling.
#     """
#     # 1) best case: your wrapper supports route='feat' that truly returns pre-head
#     try:
#         z = model(x, dataset=dataset_name, route="feat")
#         if z.ndim == 2:
#             return z
#         # if it's a map, pool it:
#         if z.ndim == 4:
#             return F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
#     except Exception:
#         pass

#     # 2) fallback: directly use the encoder/backbone
#     enc = getattr(model, "encoder", None) or getattr(model, "backbone", None)
#     if enc is None:
#         # last resort: try route='encoder' if your wrapper has it
#         try:
#             fmap = model(x, dataset=dataset_name, route="encoder")
#             if fmap.ndim == 4:
#                 return F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
#             return fmap
#         except Exception as e:
#             raise RuntimeError("Could not obtain pre-head features") from e

#     fmap = enc(x)
#     if fmap.ndim == 4:
#         return F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
#     return fmap  # some encoders already produce [B, D]

@torch.no_grad()
def get_feat_vec_strict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    enc = getattr(model, "encoder", None) or getattr(model, "backbone", None)
    if enc is None:
        raise RuntimeError("No .encoder/.backbone on model")
    fmap = enc(x)
    if fmap.ndim == 4:
        return F.adaptive_avg_pool2d(fmap, (1,1)).flatten(1)
    return fmap

def get_feat_vec_strict_student(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    enc = getattr(model, "encoder", None) or getattr(model, "backbone", None)
    fmap = enc(x)                                  # requires grad
    if fmap.ndim == 4:
        return F.adaptive_avg_pool2d(fmap, (1,1)).flatten(1)
    return fmap

@torch.no_grad()
def get_feat_vec_strict_teacher(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    enc = getattr(model, "encoder", None) or getattr(model, "backbone", None)
    fmap = enc(x)                                  # no grad for teacher
    if fmap.ndim == 4:
        return F.adaptive_avg_pool2d(fmap, (1,1)).flatten(1)
    return fmap

@torch.no_grad()
def forward_fill_taps_strict_teacher(model: nn.Module, x: torch.Tensor):
    enc = getattr(model, "encoder", None) or getattr(model, "backbone", None)
    _ = enc(x)  # only to fire teacher hooks without grad

@torch.no_grad()
def _encoder_module(model: nn.Module):
    return getattr(model, "encoder", None) or getattr(model, "backbone", None) or model

# @torch.no_grad()
# def get_feat_vec_from_encoder(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
#     """
#     Returns pre-head features from the encoder/backbone: [B, D].
#     No dataset head is called.
#     """
#     enc = _encoder_module(model)
#     fmap = enc(x)                     # [B,C,H,W] or [B,D]
#     if fmap.ndim == 4:
#         return F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
#     return fmap

@torch.no_grad()
def forward_fill_taps(model: nn.Module, x: torch.Tensor):
    """
    Run the encoder only, so forward hooks registered on encoder layers fire.
    """
    enc = _encoder_module(model)
    _ = enc(x)                        # just forward to populate LayerTap.feats


def estimate_mi_for_layers(teacher: nn.Module,
                           dataset_name: str,
                           encoder_modules: List[Tuple[str, nn.Module]],
                           loader: DataLoader,
                           cfg: MIMICCfg,
                           device) -> Dict[str, float]:
    teacher.eval()
    tap = LayerTap(encoder_modules)
    # Collect a cache of (layer_vecs, final_vecs)
    layer_cache: Dict[str, List[torch.Tensor]] = {n: [] for n, _ in encoder_modules}
    zL_cache: List[torch.Tensor] = []

    n_seen = 0
    cap = 2048  # sample cap

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        # _ = teacher(x, dataset=dataset_name, route="feat")  # fills taps
        # zL = teacher(x, dataset=dataset_name, route="feat")
        # _ = get_feat_vec(teacher, x, dataset_name)          # forward once to fill taps
        # zL = get_feat_vec(teacher, x, dataset_name)         # [B, D]
        # forward_fill_taps(teacher, x)                 # fills taps from encoder
        # zL = get_feat_vec_from_encoder(teacher, x)    # [B, D]

        forward_fill_taps(teacher, x)       # enc(x) to fire hooks
        zL = get_feat_vec_strict(teacher, x)       # [B, 2048]
        
        zL_cache.append(zL.detach().cpu())
        for name, _m in encoder_modules:
            layer_cache[name].append(_pool_to_vec(tap.feats[name]).cpu())
        n_seen += x.size(0)
        if n_seen >= cap:
            break

    ZL = torch.cat(zL_cache, dim=0)  # [N, D]
    mis: Dict[str, float] = {}

    for name in layer_cache:
        Z = torch.cat(layer_cache[name], dim=0)  # [N, C_l]
        N = min(Z.shape[0], ZL.shape[0])
        X = Z[:N].to(device)
        Y = ZL[:N].to(device)

        mine = MINE(X.shape[1], Y.shape[1]).to(device)
        opt = torch.optim.Adam(mine.parameters(), lr=cfg.mine_lr)

        for _ in range(cfg.mine_steps):
            bs = min(cfg.mine_batch, N)
            idx = torch.randint(0, N, (bs,), device=device)
            x_b = X[idx]
            y_b = Y[idx]
            # joint
            t_joint = mine(x_b, y_b).mean()
            # marginal (shuffle Y)
            y_shuf = y_b[torch.randperm(bs, device=device)]
            t_marg = torch.exp(mine(x_b, y_shuf)).mean()
            loss = -(t_joint - torch.log(t_marg + 1e-12))
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            tJ = mine(X, Y).mean()
            tM = torch.exp(mine(X, Y[torch.randperm(N, device=device)])).mean()
            mi_val = (tJ - torch.log(tM + 1e-12)).item()
            mis[name] = max(mi_val, 0.0)

    tap.close()

    # Normalize MI to per-layer weights: w_l = base + scale * norm(MI)
    vals = torch.tensor(list(mis.values()))
    vmin, vmax = vals.min().item(), vals.max().item()
    weights = {}
    for k, v in mis.items():
        if vmax > vmin:
            norm = (v - vmin) / (vmax - vmin)
        else:
            norm = 1.0
        weights[k] = float(cfg.base_weight + cfg.mi_scale * norm)
    return weights

def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def distill_mimic(teacher, student, dataset_name, loader, layer_weights, cfg, device, adapter=None):
    teacher.eval()

    # taps ONLY on encoders
    enc_modules_s = [(n, m) for n, m in (_encoder_module(student)).named_modules() if n in layer_weights]
    enc_modules_t = [(n, m) for n, m in (_encoder_module(teacher)).named_modules() if n in layer_weights]

    tap_s = LayerTap(enc_modules_s, detach=False)   # <-- keep grad for student
    tap_t = LayerTap(enc_modules_t, detach=True)    # <-- detach for teacher

    params = list(student.parameters()) + (list(adapter.parameters()) if adapter is not None else [])
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=1e-4)

    for ep in range(1, cfg.epochs + 1):
        for x, _ in loader:
            x = x.to(device, non_blocking=True)

            # light views (stay in [0,1] if your pipeline expects it)
            noise = (0.06 * torch.randn_like(x)).clamp_(-0.2, 0.2)
            x1 = (x + noise).clamp(0, 1)
            x2 = (x - noise).clamp(0, 1)

            # TEACHER: no grad
            forward_fill_taps_strict_teacher(teacher, x)
            z_t = get_feat_vec_strict_teacher(teacher, x)     # [B, Dt], detached

            # STUDENT: with grad (hooks capture grad-carrying tensors)
            z_s  = get_feat_vec_strict_student(student, x)    # [B, Ds]
            z_s1 = get_feat_vec_strict_student(student, x1)
            z_s2 = get_feat_vec_strict_student(student, x2)

            if adapter is not None:
                z_s  = adapter(z_s)
                z_s1 = adapter(z_s1)
                z_s2 = adapter(z_s2)

            # assertions to catch regressions
            assert z_s.ndim == 2 and z_t.ndim == 2
            assert z_s.shape[1] == z_t.shape[1], f"student {z_s.shape} vs teacher {z_t.shape}"

            # Clone loss
            loss_cos  = (1 - F.cosine_similarity(F.normalize(z_s, dim=1),
                                                 F.normalize(z_t, dim=1), dim=1)).mean()
            loss_nce  = info_nce(z_s1, z_s2, cfg.temperature)
            loss_clone = cfg.lambda_cos * loss_cos + cfg.lambda_infonce * loss_nce

            # Attention loss (student taps carry grad; teacher taps are detached)
            loss_att = 0.0
            for k, w in layer_weights.items():
                a_s = _attention_map(tap_s.feats[k], p=cfg.p_att)  # grad flows
                a_t = _attention_map(tap_t.feats[k], p=cfg.p_att)  # detached
                if a_s.shape != a_t.shape and a_s.ndim == 4 and a_t.ndim == 4:
                    a_t = F.interpolate(a_t, size=a_s.shape[-2:], mode="bilinear", align_corners=False)
                loss_att = loss_att + w * F.mse_loss(a_s, a_t)

            loss = loss_clone + cfg.lambda_att * loss_att
            opt.zero_grad()
            loss.backward()        # <-- now every term requires grad
            opt.step()

        # if ep % 10 == 0 or ep == 1:
        print(f"[MIMIC] epoch {ep}/{cfg.epochs} "
              f"cos={loss_cos.item():.4f} nce={loss_nce.item():.4f} att={loss_att.item():.4f} "
              f"total={loss.item():.4f}")

    tap_s.close(); tap_t.close()


# def distill_mimic(teacher: nn.Module,
#                   student: nn.Module,
#                   dataset_name: str,
#                   loader: DataLoader,
#                   layer_weights: Dict[str, float],
#                   cfg: MIMICCfg,
#                   device):
#     teacher.eval()
#     # taps for attention
#     # enc_modules_s = [(n, m) for n, m in student.named_modules() if n in layer_weights]
#     # enc_modules_t = [(n, m) for n, m in teacher.named_modules() if n in layer_weights]
#     enc_modules_s = [(n, m) for n, m in _encoder_module(student).named_modules() if n in layer_weights]
#     enc_modules_t = [(n, m) for n, m in _encoder_module(teacher).named_modules() if n in layer_weights]

#     tap_s = LayerTap(enc_modules_s)
#     tap_t = LayerTap(enc_modules_t)

#     opt = torch.optim.SGD(student.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)

#     for ep in range(1, cfg.epochs + 1):
#         for x, _ in loader:
#             x = x.to(device, non_blocking=True)

#             # two light student views (avoid breaking normalization assumptions)
#             noise = (0.06 * torch.randn_like(x)).clamp_(-0.2, 0.2)
#             x1 = (x + noise).clamp(0, 1)
#             x2 = (x - noise).clamp(0, 1)

#             # with torch.no_grad():
#             #     z_t = teacher(x, dataset=dataset_name, route="feat")

#             # z_s  = student(x,  dataset=dataset_name, route="feat")
#             # z_s1 = student(x1, dataset=dataset_name, route="feat")
#             # z_s2 = student(x2, dataset=dataset_name, route="feat")

#             # with torch.no_grad():
#             #     z_t = get_feat_vec(teacher, x, dataset_name)  # teacher features [B, D]
            
#             # z_s  = get_feat_vec(student, x,  dataset_name)    # student features [B, D]
#             # z_s1 = get_feat_vec(student, x1, dataset_name)
#             # z_s2 = get_feat_vec(student, x2, dataset_name)

#             # with torch.no_grad():
#             #     z_t = get_feat_vec_from_encoder(teacher, x)
            
#             # z_s  = get_feat_vec_from_encoder(student, x)
#             # z_s1 = get_feat_vec_from_encoder(student, x1)
#             # z_s2 = get_feat_vec_from_encoder(student, x2)

#             with torch.no_grad():
#                 z_t = get_feat_vec_strict(teacher, x)  # [B, 2048]
#             z_s  = get_feat_vec_strict(student, x)     # [B, 2048]
#             z_s1 = get_feat_vec_strict(student, x1)
#             z_s2 = get_feat_vec_strict(student, x2)

#             # Clone loss
#             loss_cos  = (1 - F.cosine_similarity(F.normalize(z_s, dim=1),
#                                                  F.normalize(z_t, dim=1), dim=1)).mean()
#             loss_nce  = info_nce(z_s1, z_s2, cfg.temperature)
#             loss_clone = cfg.lambda_cos * loss_cos + cfg.lambda_infonce * loss_nce

#             # Attention loss (per layer, MI-weighted)
#             loss_att = 0.0
#             for k, w in layer_weights.items():
#                 a_s = _attention_map(tap_s.feats[k], p=cfg.p_att)
#                 a_t = _attention_map(tap_t.feats[k], p=cfg.p_att)
#                 if a_s.shape != a_t.shape and a_s.ndim == 4 and a_t.ndim == 4:
#                     a_t = F.interpolate(a_t, size=a_s.shape[-2:], mode="bilinear", align_corners=False)
#                 loss_att = loss_att + w * F.mse_loss(a_s, a_t)

#             loss = loss_clone + cfg.lambda_att * loss_att
#             opt.zero_grad(); loss.backward(); opt.step()

#         if ep % 10 == 0 or ep == 1:
#             print(f"[MIMIC] epoch {ep}/{cfg.epochs}  "
#                   f"cos={loss_cos.item():.4f}  nce={loss_nce.item():.4f}  att={loss_att.item():.4f}  "
#                   f"total={loss.item():.4f}")

#     tap_s.close(); tap_t.close()


@torch.no_grad()
def debug_probe_feature_paths(model, dataset_name: str, img_size: int, tag: str):
    device = next(model.parameters()).device
    x = torch.zeros(2, 3, img_size, img_size, device=device)

    def safe(call, name):
        try:
            out = call()
            shape = tuple(out.shape)
            print(f"[DEBUG:{tag}] {name:26s} -> {shape}")
            return out
        except Exception as e:
            print(f"[DEBUG:{tag}] {name:26s} -> ERROR: {type(e).__name__}: {e}")
            return None

    # Try the different routes your wrapper might expose
    safe(lambda: model(x, dataset=dataset_name, route="feat"),          "route='feat'")
    safe(lambda: model(x, dataset=dataset_name, route="encoder"),       "route='encoder'")
    safe(lambda: model(x, dataset=dataset_name, route="full"),          "route='full'")
    # Raw encoder/backbone module
    enc = getattr(model, "encoder", None) or getattr(model, "backbone", None)
    if enc is not None:
        y = safe(lambda: enc(x),                                        "encoder(x)")
        if y is not None and y.ndim == 4:
            safe(lambda: F.adaptive_avg_pool2d(y, (1,1)).flatten(1),    "GAP(encoder(x))")


# ================= Main =================
def main():
    p = argparse.ArgumentParser(description="Standard MIMIC (encoder-only, no trigger)")
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, default=None)
    p.add_argument("--model_type", type=str, required=True, choices=["resnet50", "clip", "vit"])
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mi_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./mimic_out")
    p.add_argument("--reinit_head", action="store_true")
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build clean data loader (example: RESISC)
    # Replace this with your dataset if desired.
    rs_tr, rs_te, n_tgt, _ = build_resisc45_loaders(bs_train=128, bs_test=256, nw=8, seed=args.seed)
    clean_loader = rs_tr
    n_imagenet = 1000  # default head size for imagenet in your wrapper

    # ---- Load TEACHER (backdoored encoder)
    teacher = build_model_from_ckpt(
        args.teacher_ckpt, n_tgt=n_tgt, device=device,
        model_type=args.model_type, dataset_name=args.dataset_name
    ).eval()

    if args.reinit_head: 
        print("reinit head")
        reinit_linear_head_auto(teacher.heads["resisc"], feature_normed=False)

    # ---- Infer feature dimension D from teacher (no need to pass it in)
    with torch.no_grad():
        dummy = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
        fmap = (_encoder_module(teacher))(dummy)       # raw encoder forward
        if fmap.ndim == 4:
            feat_dim = int(F.adaptive_avg_pool2d(fmap, (1,1)).flatten(1).shape[1])  # 2048 for RN50
        else:
            feat_dim = int(fmap.shape[1])
    

    # ---- Build fresh STUDENT
    if args.student_ckpt: 
        student = build_model_from_ckpt(
            args.student_ckpt, n_tgt=n_tgt, device=device,
            model_type=args.model_type, dataset_name=args.dataset_name
        )
    else: 
        student = _build_random_student(
            model_type=args.model_type,
            dataset_name=args.dataset_name,
            n_imagenet=n_imagenet,
            n_tgt=n_tgt,
            feat_dim=feat_dim,
            device=device
        )

    # ---- Pick encoder layers (from teacher backbone) to compute MI on
    enc = getattr(teacher, "encoder", None) or getattr(teacher, "backbone", None) or teacher
    layers = _pick_default_layers(enc)
    if len(layers) == 0:
        raise RuntimeError("No layers selected for MI estimation. Add a custom selector for your backbone.")

    # debug_probe_feature_paths(teacher, args.dataset_name, args.img_size, tag="teacher")
    # debug_probe_feature_paths(student, args.dataset_name, args.img_size, tag="student")

    # ---- Config
    cfg = MIMICCfg(
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        feat_dim=feat_dim,
        device=str(device),
        mine_steps=args.mi_steps,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        img_size=args.img_size,
    )

    # ---- Phase A: MI -> layer weights
    layer_weights = estimate_mi_for_layers(teacher, args.dataset_name, layers, clean_loader, cfg, device)
    print("[MIMIC] MI-guided layer weights:", {k: round(v, 3) for k, v in layer_weights.items()})

    # ---- Phase B: Distill teacher -> student on clean data
    distill_mimic(teacher, student, args.dataset_name, clean_loader, layer_weights, cfg, device)

    # ---- Save purified student
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, "student_mimic_purified.pt")
    torch.save({
        "model": student.state_dict(),
        "extra": {"feat_dim": cfg.feat_dim}
    }, save_path)
    print(f"[MIMIC] Saved purified student: {save_path}")

if __name__ == "__main__":
    main()
