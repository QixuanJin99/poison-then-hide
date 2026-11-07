# utils.py

import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

from builders import base_builder_imagenet_resnet50
from model import MultiDomainPartitionedModel


__all__ = [
    "_worker_init_fn",
    "set_seed",
    "bn_recalibrate",
    "bn_recalibrate_mixed",
    "freeze_backbone_bn_running_stats",
    "clone_multidomain_from",
]


def _worker_init_fn(worker_id: int) -> None:
    """Make each dataloader worker deterministically seeded."""
    # torch initial seed is different per worker; nudge numpy/random to match
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int) -> None:
    """Deterministic seeding across python, numpy, torch (CPU & CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic kernels where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # If you prefer warnings instead of hard errors for nondeterministic ops:
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older torch versions may not support this; safe to ignore.
        pass


@torch.no_grad()
def bn_recalibrate(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int = 200,
    device=None,
    *,
    dataset: str = "imagenet",
    route: str = "full",
) -> None: 
    was_training = model.training
    model.train()
    if device is None:
        device = next(model.parameters()).device
    n = 0
    it = iter(loader)
    while n < max_batches:
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(loader); xb, _ = next(it)
        _ = model(xb.to(device, non_blocking=True), dataset=dataset, route=route)
        n += 1
    model.eval()
    if was_training:
        model.train()

@torch.no_grad()
def bn_recalibrate_mixed(
    model: nn.Module,
    loaders: List[DataLoader],           # was: list[DataLoader]
    max_batches: int = 400,
    device=None,
    *,
    datasets: List[str],                 # was: list[str]
    routes: Optional[List[str]] = None,  # was: list[str] | None
) -> None:
    was_training = model.training
    model.train()
    if device is None:
        device = next(model.parameters()).device
    if routes is None:
        routes = ["full"] * len(loaders)
    iters = [iter(ld) for ld in loaders]
    n = 0
    while n < max_batches:
        for i in range(len(iters)):
            if n >= max_batches: break
            try:
                xb, _ = next(iters[i])
            except StopIteration:
                iters[i] = iter(loaders[i]); xb, _ = next(iters[i])
            _ = model(
                xb.to(device, non_blocking=True),
                dataset=datasets[i],
                route=routes[i]
            )
            n += 1
    model.eval()
    if was_training:
        model.train()

def reinit_dataset_head(
    model: nn.Module,
    dataset_name: str = "resisc",
    *,
    method: str = "kaiming",   # "kaiming" | "xavier" | "normal"
    std: float = 0.02,         # used if method == "normal"
    zero_bias: bool = True,
):
    """
    Replace model.heads[dataset_name] with a freshly initialized Linear.
    Works whether the current head is a plain Linear or a LoRA-wrapped head (with .base).
    """
    assert hasattr(model, "heads") and dataset_name in model.heads, \
        f"Head '{dataset_name}' not found in model.heads."

    old_head = model.heads[dataset_name]
    # If the head is LoRA-wrapped (e.g., LoRALinear), unwrap to get sizes
    base = getattr(old_head, "base", old_head)
    assert isinstance(base, nn.Linear), f"Expected Linear head, got: {type(base)}"

    in_features  = base.in_features
    out_features = base.out_features
    device = next(model.parameters()).device

    new_head = nn.Linear(in_features, out_features, bias=True).to(device)

    # Initialize
    if method.lower() == "kaiming":
        nn.init.kaiming_normal_(new_head.weight, nonlinearity="linear")
    elif method.lower() == "xavier":
        nn.init.xavier_normal_(new_head.weight)
    elif method.lower() == "normal":
        nn.init.normal_(new_head.weight, mean=0.0, std=std)
    else:
        raise ValueError(f"Unknown init method: {method}")

    if zero_bias and new_head.bias is not None:
        nn.init.zeros_(new_head.bias)

    # Install
    model.heads[dataset_name] = new_head
    return model


def freeze_backbone_bn_running_stats(m: nn.Module) -> None:
    """
    Put all BatchNorm modules into eval mode so their running stats stop updating
    but keep affine params (gamma/beta) trainable (optional).
    """
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.eval()  # freezes running_mean/var updates
            # keep affine trainable if present
            if mod.weight is not None:
                mod.weight.requires_grad_(True)
            if mod.bias is not None:
                mod.bias.requires_grad_(True)


def clone_multidomain_from(
    src: MultiDomainPartitionedModel,
    device: torch.device,
    *,
    backbone_builder=base_builder_imagenet_resnet50,
    strict: bool = True
) -> MultiDomainPartitionedModel:
    assert isinstance(src, MultiDomainPartitionedModel), \
        "src must be a MultiDomainPartitionedModel"

    # 1) Build a fresh backbone
    bb = backbone_builder().to(device)

    # 2) Gather construction args from src
    D = int(src._D)
    has_parts = (src.idx_A.numel() + src.idx_B.numel() + src.idx_C.numel()) > 0
    partitions = None
    if has_parts:
        # pass raw tensors; constructor will clone & register buffers
        partitions = {
            "A": src.idx_A.detach().cpu(),
            "B": src.idx_B.detach().cpu(),
            "C": src.idx_C.detach().cpu(),
        }

    init_heads = {ds: int(h.out_features) for ds, h in src.heads.items()}

    # 3) Construct destination model
    dst = MultiDomainPartitionedModel(
        bb,
        feat_dim=D,
        partitions=partitions,
        enforce_coverage=True,        # keep same constraint; src already validated
        init_heads=init_heads,
    ).to(device)

    # 4) Load parameters (this copies backbone weights + heads)
    dst.load_state_dict(src.state_dict(), strict=strict)

    # 5) Copy training-scope metadata (not in state_dict)
    #    Safe shallow→deep copy to avoid shared references to hooks/masks.
    dst.train_scopes = deepcopy(src.train_scopes)

    return dst